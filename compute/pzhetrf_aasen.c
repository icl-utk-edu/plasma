/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> s d c
 *
 **/
#include <math.h>

#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_types.h"
#include "plasma_workspace.h"
#include <plasma_core_blas.h>

#define A(m, n) ((plasma_complex64_t*)plasma_tile_addr(A, (m), (n)))
#define L(m, n) ((plasma_complex64_t*)plasma_tile_addr(A, (m), (n)-1))
#define U(m, n) ((plasma_complex64_t*)plasma_tile_addr(A, (m)-1, (n)))

#define T(m, n) ((plasma_complex64_t*)(W4((m)+((m)-(n))*A.mt)))
#define Tgb(m, n) ((plasma_complex64_t*)plasma_tile_addr(T, (m), (n)))

// W(m): nb*nb used to compute T(k,k)
#define W(m)    ((plasma_complex64_t*)plasma_tile_addr(W,  (m), 0))
// W2(m): mt*(nb*nb) to store H
#define W2(m)   ((plasma_complex64_t*)plasma_tile_addr(W2, (m), 0))
// W3(m): mt*(nb*nb) used to form T(k,k)
#define W3(m)   ((plasma_complex64_t*)plasma_tile_addr(W3, (m), 0))
// W4(m): mt*(nb*nb) used to store off-diagonal T
#define W4(m)   ((plasma_complex64_t*)plasma_tile_addr(W4, (m), 0))
// W5(m): wmt used to update L(:,k)
#define W5(m)   ((plasma_complex64_t*)plasma_tile_addr(W5, (m), 0))

#define H(m, n) (uplo == PlasmaLower ? W2((m)) : W2((n)))
#define IPIV(i) (ipiv + (i)*(A.mb))

/***************************************************************************//**
 *  Parallel tile LDLt factorization.
 * @see plasma_omp_zhetrf_aasen
 ******************************************************************************/
void plasma_pzhetrf_aasen(plasma_enum_t uplo,
                          plasma_desc_t A, int *ipiv,
                          plasma_desc_t T,
                          plasma_desc_t W,
                          plasma_sequence_t *sequence,
                          plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    // Read parameters from the context.
    plasma_context_t *plasma = plasma_context_self();
    int ib = plasma->ib;
    int wmt = W.mt-(1+4*A.mt);

    // Creaet views for the workspaces
    plasma_desc_t W2
         = plasma_desc_view(W, A.mb,            0,   A.mt*A.mb, A.nb);
    plasma_desc_t W3
         = plasma_desc_view(W, (1+1*A.mt)*A.mb, 0,   A.mt*A.mb, A.nb);
    plasma_desc_t W4
         = plasma_desc_view(W, (1+2*A.mt)*A.mb, 0, 2*A.mt*A.mb, A.nb);
    plasma_desc_t W5
         = plasma_desc_view(W, (1+4*A.mt)*A.mb, 0,    wmt*A.mb, A.nb);

    //==============
    // PlasmaLower
    //==============
    // NOTE: In old PLASMA, we used priority.
    if (uplo == PlasmaLower) {
        for (int k = 0; k < A.mt; k++) {
            int nvak = plasma_tile_nview(A, k);
            int mvak = plasma_tile_mview(A, k);
            int ldak = plasma_tile_mmain(A, k);
            int ldtk = plasma_tile_mmain(W4, k);

            // -- computing offdiagonals H(1:k-1, k) -- //
            for (int m=1; m < k; m++) {
                int mvam = plasma_tile_mview(A, m);
                int ldtm = plasma_tile_mmain(W4, m);
                plasma_core_omp_zgemm(
                    PlasmaNoTrans, PlasmaConjTrans,
                    mvam, mvak, mvam,
                    1.0, T(m, m), ldtm,
                         L(k, m), ldak,
                    0.0, H(m, k), A.mb,
                    sequence, request);
                if (m > 1) {
                    plasma_core_omp_zgemm(
                        PlasmaNoTrans, PlasmaConjTrans,
                        mvam, mvak, A.mb,
                        1.0, T(m, m-1), ldtm,
                             L(k, m-1), ldak,
                        1.0, H(m, k),   A.mb,
                        sequence, request);
                }
                int mvamp1 = plasma_tile_mview(A, m+1);
                int ldtmp1 = plasma_tile_mmain(W4, m+1);
                plasma_core_omp_zgemm(
                    PlasmaConjTrans, PlasmaConjTrans,
                    mvam, mvak, mvamp1,
                    1.0, T(m+1, m), ldtmp1,
                         L(k, m+1), ldak,
                    1.0, H(m, k),   A.mb,
                    sequence, request);
            }
            // ---- end of computing H(1:(k-1),k) -- //

            // -- computing diagonal T(k, k) -- //
            plasma_complex64_t beta;
            if (k > 1) {
                int num = k-1;
                for (int m = 1; m < k; m++) {
                    int mvam = plasma_tile_mview(A, m);
                    int id = (m-1) % num;
                    if (m < num+1)
                        beta = 0.0;
                    else
                        beta = 1.0;

                    plasma_core_omp_zgemm(
                        PlasmaNoTrans, PlasmaNoTrans,
                        mvak, mvak, mvam,
                        -1.0, L(k, m), ldak,
                              H(m, k), A.mb,
                        beta, W3(id),  A.mb,
                        sequence, request);
                }
                // all-reduce W3 using a binary tree                          //
                // NOTE: Old PLASMA had an option to reduce in a set of tiles //
                // num_players: number of players
                int num_players = num;
                // num_rounds : height of tournament
                int num_rounds = ceil( log10((double)num_players)/log10(2.0) );
                // base: intervals between brackets
                int base  = 2;
                for (int round = 1; round <= num_rounds; round++) {
                    int num_brackets = num_players / 2; // number of brackets
                    for (int bracket = 0; bracket < num_brackets; bracket++) {
                        // first contender
                        int m1 = base*bracket;
                        // second contender
                        int m2 = m1+base/2;
                        plasma_core_omp_zgeadd(
                            PlasmaNoTrans, mvak, mvak,
                            1.0, W3(m2), A.mb,
                            1.0, W3(m1), A.mb,
                            sequence, request);
                    }
                    num_players = ceil( ((double)num_players)/2.0 );
                    base = 2*base;
                }
                plasma_core_omp_zlacpy(
                    PlasmaLower, PlasmaNoTrans,
                    mvak, mvak,
                    A(k, k), ldak,
                    T(k, k), ldtk,
                    sequence, request);
                plasma_core_omp_zgeadd(
                    PlasmaNoTrans, mvak, mvak,
                    1.0, W3(0), A.mb,
                    1.0, T(k, k), ldtk,
                    sequence, request);
            }
            else { // k == 0 or 1
                plasma_core_omp_zlacpy(
                    PlasmaLower, PlasmaNoTrans,
                    mvak, mvak,
                    A(k, k), ldak,
                    T(k, k), ldtk,
                    sequence, request);
                // expanding to full matrix
                plasma_core_omp_zlacpy(
                        PlasmaLower, PlasmaConjTrans,
                        mvak, mvak,
                        T(k, k), ldtk,
                        T(k, k), ldtk,
                        sequence, request);
            }

            if (k > 0) {
                if (k > 1) {
                    plasma_core_omp_zgemm(
                        PlasmaNoTrans, PlasmaNoTrans,
                        mvak, A.mb, mvak,
                        1.0, L(k, k),   ldak,
                             T(k, k-1), ldtk,
                        0.0, W(0), A.mb,
                        sequence, request);
                    plasma_core_omp_zgemm(
                        PlasmaNoTrans, PlasmaConjTrans,
                        mvak, mvak, A.mb,
                        -1.0, W(0), A.mb,
                              L(k, k-1), ldak,
                         1.0, T(k, k), ldtk,
                        sequence, request);
                }

                // - symmetrically solve with L(k,k) //
                plasma_core_omp_zhegst(
                    1, PlasmaLower, mvak,
                    T(k, k), ldtk,
                    L(k, k), ldak,
                    sequence, request);
                // expand to full matrix
                plasma_core_omp_zlacpy(
                        PlasmaLower, PlasmaConjTrans,
                        mvak, mvak,
                        T(k, k), ldtk,
                        T(k, k), ldtk,
                        sequence, request);
            }

            // computing H(k, k) //
            beta = 0.0;
            if (k > 1) {
                plasma_core_omp_zgemm(
                    PlasmaNoTrans, PlasmaConjTrans,
                    mvak, mvak, A.nb,
                    1.0, T(k, k-1), ldtk,
                         L(k, k-1), ldak,
                    0.0, H(k, k), A.mb,
                    sequence, request);
                beta = 1.0;
            }

            // computing the (k+1)-th column of L //
            if (k+1 < A.nt) {
                int ldakp1 = plasma_tile_mmain(A, k+1);
                if (k > 0) {
                    // finish computing H(k, k) //
                    plasma_core_omp_zgemm(
                        PlasmaNoTrans, PlasmaConjTrans,
                        mvak, mvak, mvak,
                        1.0,  T(k, k), ldtk,
                              L(k, k), ldak,
                        beta, H(k, k), A.mb,
                        sequence, request);

                    // computing the (k+1)-th column of L //
                    // - update with the previous column
                    if (A.mt-k < plasma->max_threads && k > 0) {
                        int num = imin(k, wmt/(A.mt-k-1)); // workspace per row
                        for (int n = 1; n <= k; n++) {
                            // update A(k+1:mt-1, k) using L(:,n) //
                            plasma_complex64_t *a1, *a2, *b, *c;
                            int ma1 = (A.mt-k)*A.mb;
                            int ma2 = plasma_tile_mmain(A, A.mt-1);
                            int mc  = (A.mt-(k+1))*A.mb;
                            int na  = plasma_tile_nmain(A, k);
                            a1 = L(k, n);       // we need only L(k+1:mt-1, n)
                            a2 = L(k, A.mt-1);  // left-over tile
                            b = H(n, k);
                            c = W5(((n-1)%num)*(A.mt-k-1));

                            int mvan = plasma_tile_mview(A, n);
                            #pragma omp task depend(in:a1[0:ma1*na])   \
                                             depend(in:a2[0:ma2*na])   \
                                             depend(in:b[0:A.mb*mvak]) \
                                             depend(inout:c[0:mc*A.mb])
                            {
                                for (int m = k+1; m < A.mt; m++) {
                                    int mvam = plasma_tile_mview(A, m);
                                    int ldam = plasma_tile_mmain(A, m);

                                    int id = (m-k-1)+((n-1)%num)*(A.mt-k-1);
                                    if (n < num+1)
                                        beta = 0.0;
                                    else
                                        beta = 1.0;

                                    #pragma omp task
                                    {
                                        plasma_core_zgemm(
                                            PlasmaNoTrans, PlasmaNoTrans,
                                            mvam, mvak, mvan,
                                            -1.0, L(m, n), ldam,
                                                  H(n, k), A.mb,
                                            beta, W5(id),  A.mb);
                                    }
                                }
                                #pragma omp taskwait
                            }
                        }
                        // accumerate within workspace using a binary tree
                        // num_players: number of players
                        int num_players = num;
                        // num_rounds: height of tournament
                        int num_rounds = ceil( log10((double)num_players)/log10(2.0) );
                        // base: intervals between brackets
                        int base = 2;
                        for (int round = 1; round <= num_rounds; round++) {
                            int num_brackets = num_players / 2; // number of brackets
                            for (int bracket = 0; bracket < num_brackets; bracket++) {
                                // first contender
                                int m1 = base*bracket;
                                // second contender
                                int m2 = m1+base/2;

                                plasma_complex64_t *c1, *c2;
                                int mc = (A.mt-(k+1))*A.mb;
                                c1 = W5(m1*(A.mt-k-1));
                                c2 = W5(m2*(A.mt-k-1));
                                #pragma omp task depend(in:c2[0:mc*A.mb]) \
                                                 depend(inout:c1[0:mc*A.mb])
                                {
                                    for (int m = k+1; m < A.mt; m++) {
                                        int mvam = plasma_tile_mview(A, m);
                                        #pragma omp task
                                        {
                                            plasma_core_zgeadd(
                                                PlasmaNoTrans, mvam, mvak,
                                                1.0, W5((m-k-1)+m2*(A.mt-k-1)), A.mb,
                                                1.0, W5((m-k-1)+m1*(A.mt-k-1)), A.mb);
                                        }
                                    }
                                    #pragma omp taskwait
                                }
                            }
                            num_players = ceil( ((double)num_players)/2.0 );
                            base = 2*base;
                        }
                        // accumelate into A(k+1:mt-1, k)
                        {
                            plasma_complex64_t *c1;
                            plasma_complex64_t *c2_in;
                            plasma_complex64_t *c3_in;
                            plasma_complex64_t *c2_out;
                            int mc = (A.mt-(k+1))*A.mb;
                            int mc2_in  = (A.mt-k)*A.mb;
                            int mc3_in  = plasma_tile_mmain(A, A.mt-1);
                            int mc2_out = (A.mt-(k+1))*A.mb;
                            int nc2 = plasma_tile_nmain(A, k);
                            c1 = W5(0);
                            c2_in  = A(k, k);       // we write only A(k+1,k), but dependency from sym-swap
                            c3_in  = A(A.mt-1, k);  // left-over tile
                            c2_out = A(k+1, k);
                            #pragma omp task depend(in:c1[0:mc*A.mb])       \
                                             depend(in:c2_in[0:mc2_in*nc2]) \
                                             depend(in:c3_in[0:mc3_in*nc2]) \
                                             depend(out:c2_out[0:mc2_out*nc2])
                            {
                                for (int m = k+1; m < A.mt; m++) {
                                    int mvam = plasma_tile_mview(A, m);
                                    int ldam = plasma_tile_mmain(A, m);
                                    #pragma omp task
                                    {
                                        plasma_core_zgeadd(
                                            PlasmaNoTrans, mvam, mvak,
                                            1.0, W5(m-k-1), A.mb,
                                            1.0, A(m, k),   ldam);
                                    }
                                }
                                #pragma omp taskwait
                            }
                        }
                    }
                    else {
                        for (int n = 1; n <= k; n++) {
                            // update L(:,k+1) using L(:,n) //
                            plasma_complex64_t *a1, *a2;
                            plasma_complex64_t *b;
                            plasma_complex64_t *c1_in, *c2_in;
                            plasma_complex64_t *c_out;
                            int ma1 = (A.mt-k)*A.mb;
                            int ma2 = plasma_tile_mmain(A, A.mt-1);
                            int mc1_in = (A.mt-k)*A.mb;
                            int mc2_in = plasma_tile_mmain(A, A.mt-1);
                            int mc_out = (A.mt-(k+1))*A.mb;
                            int na = plasma_tile_nmain(A, k);
                            int nc = plasma_tile_nmain(A, k);
                            a1 = L(k, n);       // we read only L(k+1:mt-1, n), dependency from row-swap
                            a2 = L(A.mt-1, n);  // left-over tile
                            b = H(n, k);
                            c1_in = A(k, k);      // we write only A(k+1,k), but dependency from sym-swap
                            c2_in = A(A.mt-1, k); // left-over tile
                            c_out = A(k+1, k);

                            int mvan = plasma_tile_mview(A, n);
                            #pragma omp task depend(in:a1[0:ma1*na])     \
                                             depend(in:a2[0:ma2*na])     \
                                             depend(in:b[0:A.mb*mvak])   \
                                             depend(in:c1_in[0:mc1_in*nc]) \
                                             depend(in:c2_in[0:mc2_in*nc]) \
                                             depend(out:c_out[0:mc_out*nc])
                            {
                                for (int m = k+1; m < A.mt; m++) {
                                    int mvam = plasma_tile_mview(A, m);
                                    int ldam = plasma_tile_mmain(A, m);
                                    #pragma omp task
                                    {
                                        plasma_core_zgemm(
                                            PlasmaNoTrans, PlasmaNoTrans,
                                            mvam, mvak, mvan,
                                            -1.0, L(m, n), ldam,
                                                  H(n, k), A.mb,
                                             1.0, A(m, k), ldam);
                                    }
                                }
                                #pragma omp taskwait
                            }
                        }
                    }
                } // end of if (k > 0)

                // ============================= //
                // ==     PLASMA LU panel     == //
                // ============================= //
                // -- compute LU of the panel -- //
                plasma_complex64_t *a1, *a2;
                a1 = L(k+1, k+1);
                a2 = L(A.mt-1, k+1);

                int mlkk  = A.m - (k+1)*A.mb; // dimension
                int ma1 = (A.mt-(k+1)-1)*A.mb;
                int ma2 = plasma_tile_mmain(A, A.mt-1);
                int na  = plasma_tile_nmain(A, k);

                int k1 = 1+(k+1)*A.nb;
                int k2 = imin(mlkk, mvak)+(k+1)*A.nb;

                int num_panel_threads = imin(plasma->max_panel_threads,
                                             A.mt-(k+1));

                #pragma omp task depend(inout:a1[0:ma1*na]) \
                                 depend(inout:a2[0:ma2*na]) \
                                 depend(out:ipiv[k1-1:k2])
                {
                    volatile int *max_idx =
                        (int*)malloc(num_panel_threads*sizeof(int));
                    if (max_idx == NULL)
                        plasma_request_fail(sequence, request,
                                            PlasmaErrorOutOfMemory);

                    volatile plasma_complex64_t *max_val =
                        (plasma_complex64_t*)malloc(num_panel_threads*sizeof(
                                                    plasma_complex64_t));
                    if (max_val == NULL)
                        plasma_request_fail(sequence, request,
                                            PlasmaErrorOutOfMemory);

                    volatile int info = 0;

                    plasma_barrier_t barrier;
                    plasma_barrier_init(&barrier);

                    if (sequence->status == PlasmaSuccess) {
                        for (int rank = 0; rank < num_panel_threads; rank++) {
                            #pragma omp task shared(barrier)
                            {
                                plasma_desc_t view =
                                    plasma_desc_view(A,
                                                    (k+1)*A.mb, k*A.nb,
                                                     mlkk, mvak);

                                plasma_core_zgetrf(view, IPIV(k+1), ib,
                                            rank, num_panel_threads,
                                            max_idx, max_val, &info,
                                            &barrier);

                                if (info != 0)
                                    plasma_request_fail(sequence, request,
                                                        (k+1)*A.mb+info);
                            }
                        }
                    }
                    #pragma omp taskwait
                    free((void*)max_idx);
                    free((void*)max_val);
                    {
                        for (int i = 0; i < imin(mlkk, mvak); i++) {
                            IPIV(k+1)[i] += (k+1)*A.mb;
                        }
                    }
                }
                // ============================== //
                // ==  end of PLASMA LU panel  == //
                // ============================== //

                // -- apply pivoting to previous columns of L -- //
                for (int n = 1; n < k+1; n++) {
                    ma1 = (A.mt-k-1)*A.mb;
                    ma2 = plasma_tile_mmain(A, A.mt-1);
                    na  = plasma_tile_nmain(A, n-1);

                    a1 = L(k+1, n);
                    a2 = L(A.mt-1, n);
                    #pragma omp task depend(in:ipiv[(k1-1):k2]) \
                                     depend(inout:a1[0:ma1*na]) \
                                     depend(inout:a2[0:ma2*na])
                    {
                        if (sequence->status == PlasmaSuccess) {
                            plasma_desc_t view =
                                plasma_desc_view(A, 0, (n-1)*A.nb, A.m, na);
                            plasma_core_zgeswp(PlasmaRowwise, view, k1, k2, ipiv, 1);
                        }
                    }
                }

                // computing T(k+1, k) //
                int mvakp1 = plasma_tile_mview(A, k+1);
                int ldak_n = plasma_tile_nmain(A, k);
                int ldtkp1 = plasma_tile_mmain(W4, k+1);
                // copy upper-triangular part of L(k+1,k+1) to T(k+1,k)
                // and then zero it out
                plasma_core_omp_zlacpy(
                        PlasmaUpper, PlasmaNoTrans,
                        mvakp1, mvak,
                        L(k+1, k+1), ldakp1,
                        T(k+1, k  ), ldtkp1,
                        sequence, request);
                plasma_core_omp_zlaset(
                        PlasmaUpper,
                        ldakp1, ldak_n, 0, 0,
                        mvakp1, mvak,
                        0.0, 1.0,
                        L(k+1, k+1));
                if (k > 0) {
                    plasma_core_omp_ztrsm(
                        PlasmaRight, PlasmaLower,
                        PlasmaConjTrans, PlasmaUnit,
                        mvakp1, mvak,
                        1.0, L(k,   k), ldak,
                             T(k+1, k), ldtkp1,
                        sequence, request);
                }

                // -- symmetrically apply pivoting to trailing A -- //
                {
                    int mnt1, mnt2;
                    mnt2 = A.mt-1-(k+1);
                    ma2  = plasma_tile_mmain(A, A.mt-1);

                    mnt1 = (mnt2*(mnt2+1))/2; // # of remaining tiles in a1
                    mnt1 *= A.mb*A.mb;

                    mnt2 *= A.mb*ma2; // a2
                    mnt2 += ma2*ma2;  // last diagonal

                    a1 = A(k+1, k+1);
                    a2 = A(A.mt-1, k+1);

                    int num_swap_threads = imin(plasma->max_panel_threads,
                                                A.mt-(k+1));

                    #pragma omp task depend(in:ipiv[(k1-1):k2]) \
                                     depend(inout:a1[0:mnt1]) \
                                     depend(inout:a2[0:mnt2])
                    {
                        plasma_barrier_t barrier;
                        plasma_barrier_init(&barrier);
                        for (int rank = 0; rank < num_swap_threads; rank++) {
                            #pragma omp task shared(barrier)
                            {
                                plasma_core_zheswp(rank, num_swap_threads, PlasmaLower, A, k1, k2, ipiv, 1, &barrier);
                            }
                        }
                        #pragma omp taskwait
                    }
                }
            }

            // copy T(k+1, k) to T(k, k+1) for zgbtrf,
            // forcing T(k, k) and T(k+1, k) tiles are ready
            plasma_complex64_t *Tin10 = NULL;
            plasma_complex64_t *Tin11 = NULL;
            plasma_complex64_t *Tin21 = NULL;

            plasma_complex64_t *Tout01 = NULL;
            plasma_complex64_t *Tout11 = NULL;
            plasma_complex64_t *Tout21 = NULL;
            plasma_complex64_t *Tout   = NULL;

            int km1 = imax(0, k-1);
            int kp1 = imin(k+1, A.mt-1);
            int ldtkm1 = plasma_tile_mmain(W4, km1);
            int ldtkp1 = plasma_tile_mmain(W4, kp1);
            int nvakp1 = plasma_tile_nview(A, kp1);
            Tin10 = T(k, km1);
            Tin11 = T(k, k);
            Tin21 = T(kp1, k);

            Tout01 = Tgb(km1, k);
            Tout11 = Tgb(k,   k);
            Tout21 = Tgb(kp1, k);
            Tout   = Tgb(imax(0, k-T.kut+1), k);
            #pragma omp task depend(in:Tin10[0:A.mb*ldtk])     \
                             depend(in:Tin11[0:nvak*ldtk])     \
                             depend(in:Tin21[0:nvakp1*ldtkp1]) \
                             depend(out:Tout01[0:nvak*ldtkm1]) \
                             depend(out:Tout11[0:nvak*ldtk])   \
                             depend(out:Tout21[0:nvak*ldtkp1]) \
                             depend(out:Tout[0:nvak*ldtkp1])
            {
                plasma_core_zlacpy(
                    PlasmaGeneral, PlasmaNoTrans,
                    mvak, mvak,
                    T(k, k), ldtk,
                    Tgb(k, k), ldtk);
                if (k+1 < T.nt) {
                    int mvakp1 = plasma_tile_mview(A, k+1);
                    plasma_core_zlacpy(
                        PlasmaGeneral, PlasmaNoTrans,
                        mvakp1, mvak,
                        T(kp1, k), ldtkp1,
                        Tgb(kp1, k), ldtkp1);
                }
                if (k > 0) {
                    int nvakm1 = plasma_tile_nview(A, k-1);
                    plasma_core_zlacpy(
                        PlasmaGeneral, PlasmaConjTrans,
                        mvak, nvakm1,
                        T(k, km1), ldtk,
                        Tgb(km1, k), ldtkm1);
                }
            }
        }
    }
    //==============
    // PlasmaUpper
    //==============
    else {
        // TODO: Upper
    }
}
