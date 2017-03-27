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
#include "core_blas.h"

#define A(m, n) ((plasma_complex64_t*)plasma_tile_addr(A, (m), (n)))
#define T(m, n) ((plasma_complex64_t*)plasma_tile_addr(T, (m), (n)))
#define L(m, n) ((plasma_complex64_t*)plasma_tile_addr(A, (m), (n)-1))
#define U(m, n) ((plasma_complex64_t*)plasma_tile_addr(A, (m)-1, (n)))

#define W(j)    ((plasma_complex64_t*)plasma_tile_addr(W,  (j), 0)) // nb*nb used to compute T(k,k)
#define W2(j)   ((plasma_complex64_t*)plasma_tile_addr(W2, (j), 0)) // mt*(nb*nb) to store H
#define W3(j)   ((plasma_complex64_t*)plasma_tile_addr(W3, (j), 0)) // mt*(nb*nb) used to form T(k,k)
#define W4(j)   ((plasma_complex64_t*)plasma_tile_addr(W4, (j), 0)) // wmt used to update L(:,k)

#define H(m, n) (uplo == PlasmaLower ? W2((m)) : W2((n)))
#define IPIV(i) (ipiv + (i)*(A.mb))

/***************************************************************************//**
 *  Parallel tile LDLt factorization.
 * @see plasma_omp_zhetrf_aasen
 * TODO: Use nested-parallelisation to remove synchronization points?
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
    plasma_barrier_t *barrier = &plasma->barrier;
    int ib = plasma->ib;
    int max_panel_threads = plasma->max_panel_threads;
    int wmt = W.mt-(1+2*A.mt);

    // Creaet views for the workspaces
    plasma_desc_t W2 = plasma_desc_view(W, A.mb,            0, A.mt*A.mb, A.nb);
    plasma_desc_t W3 = plasma_desc_view(W, (1+A.mt)*A.mb,   0, A.mt*A.mb, A.nb);
    plasma_desc_t W4 = plasma_desc_view(W, (1+2*A.mt)*A.mb, 0, wmt,       A.nb);

    //==============
    // PlasmaLower
    //==============
    // NOTE: In old PLASMA, we used priority.
    if (uplo == PlasmaLower) {
        for (int k = 0; k < A.mt; k++) {
            int mvak = plasma_tile_mview(A, k);
            int ldak = plasma_tile_mmain(A, k);
            int ldtk = T.mb; //plasma_tile_mmain_band(T, k);

            // -- computing offdiagonals H(1:k-1, k) -- //
            for (int m=1; m < k; m++) {
                int mvam = plasma_tile_mview(A, m);
                int ldtm = T.mb; //plasma_tile_mmain_band(T, m);
                core_omp_zgemm(
                    PlasmaNoTrans, PlasmaConjTrans,
                    mvam, mvak, mvam,
                    1.0, T(m, m), ldtm,
                         L(k, m), ldak,
                    0.0, H(m, k), A.mb,
                    sequence, request);
                if (m > 1) {
                    core_omp_zgemm(
                        PlasmaNoTrans, PlasmaConjTrans,
                        mvam, mvak, A.mb,
                        1.0, T(m, m-1), ldtm,
                             L(k, m-1), ldak,
                        1.0, H(m, k),   A.mb,
                        sequence, request);
                }
                int mvamp1 = plasma_tile_mview(A, m+1);
                int ldtmp1 = A.mb; //plasma_tile_mmain_band(T, m+1);
                core_omp_zgemm(
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

                    core_omp_zgemm(
                        PlasmaNoTrans, PlasmaNoTrans,
                        mvak, mvak, mvam,
                        -1.0, L(k, m), ldak,
                              H(m, k), A.mb,
                        beta, W3(id),  A.mb,
                        sequence, request);
                }
                // all-reduce W3 using a binary tree                          //
                // NOTE: Old PLASMA had an option to reduce in a set of tiles //
                int num_players = num;                                           // number of players
                int num_rounds = ceil( log10((double)num_players)/log10(2.0) );  // height of tournament
                int base  = 2;                                                   // intervals between brackets
                for (int round = 1; round <= num_rounds; round++) {
                    int num_brackets = num_players / 2; // number of brackets
                    for (int bracket = 0; bracket < num_brackets; bracket++) {
                        // first contender
                        int m1 = base*bracket;
                        // second contender
                        int m2 = m1+base/2;
                        core_omp_zgeadd(
                            PlasmaNoTrans, mvak, mvak,
                            1.0, W3(m2), A.mb,
                            1.0, W3(m1), A.mb,
                            sequence, request);
                    }
                    num_players = ceil( ((double)num_players)/2.0 );
                    base = 2*base;
                }
                core_omp_zlacpy(
                    PlasmaLower, PlasmaNoTrans,
                    mvak, mvak,
                    A(k, k), ldak,
                    T(k, k), ldtk,
                    sequence, request);
                core_omp_zgeadd(
                    PlasmaNoTrans, mvak, mvak,
                    1.0, W3(0), A.mb,
                    1.0, T(k, k), ldtk,
                    sequence, request);
            }
            else { // k == 0 or 1
                core_omp_zlacpy(
                    PlasmaLower, PlasmaNoTrans,
                    mvak, mvak,
                    A(k, k), ldak,
                    T(k, k), ldtk,
                    sequence, request);
                // expanding to full matrix
                core_omp_zlacpy(
                        PlasmaLower, PlasmaConjTrans,
                        mvak, mvak,
                        T(k, k), ldtk,
                        T(k, k), ldtk,
                        sequence, request);
            }

            if (k > 0) {
                if (k > 1) {
                    core_omp_zgemm(
                        PlasmaNoTrans, PlasmaNoTrans,
                        mvak, A.mb, mvak,
                        1.0, L(k, k),   ldak,
                             T(k, k-1), ldtk,
                        0.0, W(0), A.mb,
                        sequence, request);
                    core_omp_zgemm(
                        PlasmaNoTrans, PlasmaConjTrans,
                        mvak, mvak, A.mb,
                        -1.0, W(0), A.mb,
                              L(k, k-1), ldak,
                         1.0, T(k, k), ldtk,
                        sequence, request);
                }

                // - symmetrically solve with L(k,k) //
                core_omp_zhegst(
                    1, PlasmaLower, mvak,
                    T(k, k), ldtk,
                    L(k, k), ldak,
                    sequence, request);
                // expand to full matrix
                core_omp_zlacpy(
                        PlasmaLower, PlasmaConjTrans,
                        mvak, mvak,
                        T(k, k), ldtk,
                        T(k, k), ldtk,
                        sequence, request);
            }

            // computing H(k, k) //
            beta = 0.0;
            if (k > 1) {
                core_omp_zgemm(
                    PlasmaNoTrans, PlasmaConjTrans,
                    mvak, mvak, A.nb,
                    1.0, T(k, k-1), ldtk,
                         L(k, k-1), ldak,
                    0.0, H(k, k), A.mb,
                    sequence, request);
                beta = 1.0;
            }

            if (k+1 < A.nt) {
                int ldakp1 = plasma_tile_mmain(A, k+1);
                if (k > 0) {
                    core_omp_zgemm(
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
                            int mvan = plasma_tile_mview(A, n);
                            for (int m = k+1; m < A.mt; m++) {
                                int mvam = plasma_tile_mview(A, m);
                                int ldam = plasma_tile_mmain(A, m);

                                int id = (m-k-1)*num+(n-1)%num;
                                if (n < num+1)
                                    beta = 0.0;
                                else
                                    beta = 1.0;

                                if (n < num+1 || n > k-num) {
                                    core_omp_zgemm(
                                        PlasmaNoTrans, PlasmaNoTrans,
                                        mvam, mvak, mvan,
                                        -1.0, L(m, n), ldam,
                                              H(n, k), A.mb,
                                        beta, W4(id),  A.mb,
                                        sequence, request);
                                }
                                else {
                                    core_omp_zgemm(
                                        PlasmaNoTrans, PlasmaNoTrans,
                                        mvam, mvak, mvan,
                                        -1.0, L(m, n), ldam,
                                              H(n, k), A.mb,
                                        beta, W4(id),  A.mb,
                                        sequence, request);
                                }
                            }
                        }
                        // accumerate within workspace using a binary tree
                        int num_players = num;                                           // number of players
                        int num_rounds = ceil( log10((double)num_players)/log10(2.0) );  // height of tournament
                        int base = 2;                                                    // intervals between brackets
                        for (int round = 1; round <= num_rounds; round++) {
                            int num_brackets = num_players / 2; // number of brackets
                            for (int bracket = 0; bracket < num_brackets; bracket++) {
                                // first contender
                                int m1 = base*bracket;
                                // second contender
                                int m2 = m1+base/2;

                                for (int m = k+1; m < A.mt; m++) {
                                    int mvam = plasma_tile_mview(A, m);
                                    core_omp_zgeadd(
                                        PlasmaNoTrans, mvam, mvak,
                                        1.0, W4((m-k-1)*num+m2), A.mb,
                                        1.0, W4((m-k-1)*num+m1), A.mb,
                                        sequence, request);
                                }
                            }
                            num_players = ceil( ((double)num_players)/2.0 );
                            base = 2*base;
                        }
                        // accumelate into L(:,k+1)
                        for (int m = k+1; m < A.mt; m++) {
                            int mvam = plasma_tile_mview(A, m);
                            int ldam = plasma_tile_mmain(A, m);
                            core_omp_zgeadd(
                                PlasmaNoTrans, mvam, mvak,
                                1.0, W4((m-k-1)*num), A.mb,
                                1.0, L(m, k+1), ldam,
                                sequence, request);
                        }
                    }
                    else {
                        for (int n = 1; n <= k; n++) {
                            int mvan = plasma_tile_mview(A, n);
                            for (int m = k+1; m < A.mt; m++) {
                                int mvam = plasma_tile_mview(A, m);
                                int ldam = plasma_tile_mmain(A, m);
                                core_omp_zgemm(
                                    PlasmaNoTrans, PlasmaNoTrans,
                                    mvam, mvak, mvan,
                                    -1.0, L(m, n), ldam,
                                          H(n, k), A.mb,
                                     1.0, L(m, k+1), ldam,
                                    sequence, request);
                            }
                        }
                    }
                } // end of if (k > 0)

                // ============================= //
                // ==     PLASMA LU panel     == //
                // ============================= //
                // -- compute LU of the panel -- //
                plasma_complex64_t *a00, *a20;
                a00 = L(k+1, k+1);
                a20 = L(A.mt-1, k+1);

                int mlkk  = A.m - (k+1)*A.mb; // dimension
                int ma00k = (A.mt-(k+1)-1)*A.mb;
                int na00k = plasma_tile_nmain(A, k);
                int lda20 = plasma_tile_mmain(A, A.mt-1);

                int k1 = 1+(k+1)*A.nb;
                int k2 = imin(mlkk, mvak)+(k+1)*A.nb;

                #pragma omp taskwait // make sure all the tiles in the column are ready
                #pragma omp task depend(inout:a00[0:ma00k*na00k]) \
                                 depend(inout:a20[0:lda20*mvak]) \
                                 depend(out:ipiv[k1-1:k2]) /*\
                                 priority(1) */
                {
                    if (sequence->status == PlasmaSuccess) {
                        for (int rank = 0; rank < max_panel_threads; rank++) {
                            #pragma omp task // priority(1)
                            {
                                plasma_desc_t view =
                                    plasma_desc_view(A,
                                                     (k+1)*A.mb, k*A.nb,
                                                     mlkk, mvak);

                                int info = core_zgetrf(view, IPIV(k+1), ib,
                                                       rank, max_panel_threads,
                                                       barrier);
                                if (info != 0)
                                    plasma_request_fail(sequence, request, (k+1)*A.mb+info);
                            }
                        }
                    }
                    #pragma omp taskwait
                    for (int i = 0; i < imin(mlkk, mvak); i++) {
                        IPIV(k+1)[i] += (k+1)*A.mb;
                    }
                }
                //#pragma omp taskwait
                // ============================== //
                // ==  end of PLASMA LU panel  == //
                // ============================== //

                // -- apply pivoting to previous columns of L -- //
                for (int n = 1; n < k+1; n++)
                {
                    plasma_complex64_t *akk = NULL;
                    int mlkn = (A.mt-k-1)*A.mb;
                    int nlkn = plasma_tile_nmain(A, n-1);
                    akk = L(k+1, n);
                    #pragma omp task depend(in:ipiv[(k1-1):k2]) \
                                      depend(inout:akk[0:mlkn*nlkn])
                    {
                        if (sequence->status == PlasmaSuccess) {
                            plasma_desc_t view =
                                plasma_desc_view(A, 0, (n-1)*A.nb, A.m, nlkn);
                            core_zgeswp(PlasmaRowwise, view, k1, k2, ipiv, 1);
                        }
                    }
                }

                // computing T(k+1, k) //
                int mvakp1 = plasma_tile_mview(A, k+1);
                int ldak_n = plasma_tile_nmain(A, k);
                int ldtkp1 = A.mb; //plasma_tile_mmain_band(T, k+1);
                // copy upper-triangular part of L(k+1,k+1) to T(k+1,k)
                // and then zero it out
                core_omp_zlacpy(
                        PlasmaUpper, PlasmaNoTrans,
                        mvakp1, mvak,
                        L(k+1, k+1), ldakp1,
                        T(k+1, k  ), ldtkp1,
                        sequence, request);
                core_omp_zlaset(
                        PlasmaUpper,
                        ldakp1, ldak_n, 0, 0,
                        mvakp1, mvak,
                        0.0, 1.0,
                        L(k+1, k+1));
                if (k > 0) {
                    core_omp_ztrsm(
                        PlasmaRight, PlasmaLower,
                        PlasmaConjTrans, PlasmaUnit,
                        mvakp1, mvak,
                        1.0, L(k,   k), ldak,
                             T(k+1, k), ldtkp1,
                        sequence, request);
                }
                // copy T(k+1, k) to T(k, k+1) for zgbtrf
                core_omp_zlacpy(
                        PlasmaGeneral, PlasmaConjTrans,
                        mvakp1, mvak,
                        T(k+1, k), ldtkp1,
                        T(k, k+1), ldtk,
                        sequence, request);

                // -- symmetrically apply pivoting to trailing A -- //
                plasma_complex64_t *akk = NULL;
                int mlkn = (A.mt-k-1)*A.mb;
                int nlkn = plasma_tile_nmain(A, k+1);
                akk = A(k+1, k+1);
                // TODO: calling core routine for now.
                #pragma omp task depend(in:ipiv[(k1-1):k2]) \
                                 depend(inout:akk[0:nlkn*mlkn])
                {
                    core_zheswp(PlasmaLower, A, k1, k2, ipiv, 1);
                }
                // synch the row-swap of previous column before the next update
                #pragma omp taskwait
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
