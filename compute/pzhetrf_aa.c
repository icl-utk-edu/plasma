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
#define T(m, n) ((plasma_complex64_t*)plasma_tile_addr(T, (m) , (n)))
#define L(m, n) ((plasma_complex64_t*)plasma_tile_addr(A, (m), (n)-1))
#define U(m, n) ((plasma_complex64_t*)plasma_tile_addr(A, (m)-1, (n)))
#define IPIV(i) (ipiv + (i)*(A.mb))

#define W(j)  ((plasma_complex64_t*)plasma_tile_addr(W, (j), 0))        // mt  x nb*nb
#define W2(j) ((plasma_complex64_t*)plasma_tile_addr(W, (j)+A.mt, 0))   // 2mt x nb*nb
#define W3(j) ((plasma_complex64_t*)plasma_tile_addr(W, (j)+3*A.mt, 0)) // tot
#define W4(j) ((plasma_complex64_t*)plasma_tile_addr(W, (j)+3*A.mt, 0)) // tot

#define H(m,n) (uplo == PlasmaLower ? W2(m) : W2(n))

/***************************************************************************//**
 *  Parallel tile LDLt factorization.
 * @see plasma_omp_zhetrf_aa
 ******************************************************************************/
void plasma_pzhetrf_aa(plasma_enum_t uplo, 
                       plasma_desc_t A,
                       plasma_desc_t T, int *ipiv,
                       plasma_desc_t W, int *iwork,
                       plasma_sequence_t *sequence, 
                       plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    plasma_complex64_t zzero =  0.0;
    plasma_complex64_t zone  =  1.0;
    plasma_complex64_t zmone = -1.0;

    // Read parameters from the context.
    plasma_context_t *plasma = plasma_context_self();
    plasma_barrier_t *barrier = &plasma->barrier;
    int ib = plasma->ib;
    int num_panel_threads = plasma->num_panel_threads;
    int tot = W.mt-(3*A.mt); //(lwork - 3*A.mt*(A.nb*A.nb))/(A.nb*A.nb); //max(2*A.nt, panel_thread_count); 


    int *perm  = &iwork[0];
    int *iperm = &perm[A.m];
    int *perm2work  = &iperm[A.m];
    int *iperm2work = &perm2work[A.m];

    //==============
    // PlasmaLower
    //==============
    if (uplo == PlasmaLower) {
        for (int k = 0; k < A.mt; k++) {
            int tempkn = k == A.nt-1 ? A.n-k*A.nb : A.nb;
            int ldak = plasma_tile_mmain(A, k);

            /* -- computing offdiagonals H(1:k-1, k) -- */
            for(int m=1; m<k; m++) {
                int tempmm = m == A.mt-1 ? A.m-m*A.mb : A.mb;
                int ldam = plasma_tile_mmain(A, m);

                #ifdef WITH_PRIORITY
                QUARK_Task_Flag_Set(&task_flags, TASK_PRIORITY, QUARK_TASK_MAX_PRIORITY-2);
                #endif
                core_omp_zgemm(
                    PlasmaNoTrans, PlasmaConjTrans,
                    tempmm, tempkn, tempmm,
                    zone,  T(m, m), A.mb,
                           L(k, m), ldak,
                    zzero, H(m, k), ldam,
                    sequence, request);
                if( m > 1 ) {
                    core_omp_zgemm(
                        PlasmaNoTrans, PlasmaConjTrans,
                        tempmm, tempkn, A.mb,
                        zone,  T(m, m-1), A.mb,
                               L(k, m-1), ldak,
                        zone,  H(m, k),   ldam,
                        sequence, request);
                }
                int tempmp1 = m+1 == A.mt-1 ? A.m-(m+1)*A.mb : A.mb;
                core_omp_zgemm(
                    PlasmaConjTrans, PlasmaConjTrans,
                    tempmm, tempkn, tempmp1,
                    zone,  T(m+1, m), A.mb,
                           L(k, m+1), ldak,
                    zone,  H(m, k),   ldam,
                    sequence, request);

                #ifdef WITH_PRIORITY
                QUARK_Task_Flag_Set(&task_flags, TASK_PRIORITY, QUARK_TASK_MIN_PRIORITY);
                #endif
            }
            /* ---- end of computing H(1:(k-1),k) -- */
#pragma omp taskwait
printf( " -- T(%d,%d) --\n",2,1 );
for (int i=0; i<tempkn; i++) {
  for (int j=0; j<tempkn; j++) printf( "%.2e ",T(2,1)[i+j*A.mb] );
  printf( "\n" );
}

            /* -- computing diagonal T(k, k) -- */
            if (k > 1) {
                int num = imin(tot, k-1);
                for (int m=1; m<k; m++) {
                    int tempmm = m == A.mt-1 ? A.m-m*A.mb : A.mb;
                    int ldam = plasma_tile_mmain(A, m);
                    int id = (m-1) % num;

                    plasma_complex64_t beta;
                    if( m < num+1 ) {
                        beta = zzero;
                    } else{
                        beta = zone;
                    }
                    #ifdef WITH_PRIORITY
                    QUARK_Task_Flag_Set(&task_flags, TASK_PRIORITY, QUARK_TASK_MAX_PRIORITY-2);
                    #endif
#pragma omp taskwait
printf( " W3(%d)\n",id );
printf( " L(%d,%d)\n",k,m );
for (int i=0; i<tempkn; i++) {
  for (int j=0; j<tempkn; j++) printf( "%.2e ",L(k,m)[i+j*A.mb] );
  printf( "\n" );
}
printf( " H(%d,%d)\n",m,k );
for (int i=0; i<tempkn; i++) {
  for (int j=0; j<tempkn; j++) printf( "%.2e ",H(m,k)[i+j*A.mb] );
  printf( "\n" );
}
printf( "\n" );
                    core_omp_zgemm(
                        PlasmaNoTrans, PlasmaNoTrans,
                        tempkn, tempkn, tempmm,
                        zmone, L(k, m), ldak,
                               H(m, k), ldam,
                        beta,  W3(id),  A.nb,
                        sequence, request);
                    #ifdef WITH_PRIORITY
                    QUARK_Task_Flag_Set(&task_flags, TASK_PRIORITY, QUARK_TASK_MIN_PRIORITY);
                    #endif
                }
#pragma omp taskwait
printf( " --- T(%d,%d) ---\n",2,1 );
for (int i=0; i<tempkn; i++) {
  for (int j=0; j<tempkn; j++) printf( "%.2e ",T(2,1)[i+j*A.mb] );
  printf( "\n" );
}

                //#define GROUPED_REDUCTION
                #if defined(GROUPED_REDUCTION)
                int group_size = 8; // each task performs group_size geadd
                if( sqrt((double)num) >= (A.nt-k-1)*(k+1)/num_panel_threads ) group_size = 1;
                int skip  = 2;                                                   /* intervals between brackets */
                int num_players = num;                                           /* number of players          */
                int num_rounds = ceil( log10((double)num_players)/log10(2.0) );  /* height of tournament       */
                for (int round=1; round<=num_rounds; round++) {
                    int num_brackets = num_players/2;
                    for (int bracket=0; bracket<num_brackets; bracket+=group_size) {
                        #ifdef WITH_PRIORITY
                        QUARK_Task_Flag_Set(&task_flags, TASK_PRIORITY, QUARK_TASK_MAX_PRIORITY-2);
                        #endif
                        QUARK_CORE_zhetrf_group_add(plasma->quark, &task_flags,
                                                    bracket, imin(bracket+group_size,num_brackets), skip,
                                                    tempkn, W, A.mt,
                                                    sequence, request);
                        #ifdef WITH_PRIORITY
                        QUARK_Task_Flag_Set(&task_flags, TASK_PRIORITY, QUARK_TASK_MIN_PRIORITY);
                        #endif
                    }
                    num_players = ceil( ((double)num_players)/2.0 );
                    skip = 2*skip;
                }
                #else
                int num_players = num;                                           /* number of players          */
                int skip  = 2;                                                   /* intervals between brackets */
                int num_rounds = ceil( log10((double)num_players)/log10(2.0) );  /* height of tournament       */
                for(int round=1; round<=num_rounds; round++) {
                    int num_brackets = num_players / 2; /* number of brackets */
                    for (int bracket=0; bracket<num_brackets; bracket++) {
                        /* first contendar */
                        int m1 = skip*bracket;
                        /* second contendar */
                        int m2 = skip*bracket+skip/2;
                        #ifdef WITH_PRIORITY
                        QUARK_Task_Flag_Set(&task_flags, TASK_PRIORITY, QUARK_TASK_MAX_PRIORITY-2);
                        #endif
#pragma omp taskwait
printf( " W3(%d)\n",m2 );
for (int i=0; i<tempkn; i++) {
  for (int j=0; j<tempkn; j++) printf( "%.2e ",W3(m2)[i+j*A.mb] );
  printf( "\n" );
}
printf( " W3(%d)\n",m1 );
for (int i=0; i<tempkn; i++) {
  for (int j=0; j<tempkn; j++) printf( "%.2e ",W3(m1)[i+j*A.mb] );
  printf( "\n" );
}
printf( "\n" );
                        core_omp_zgeadd(
                            PlasmaNoTrans, tempkn, tempkn,
                            zone, W3(m2), A.nb,
                            zone, W3(m1), A.nb,
                            sequence, request);
                        #ifdef WITH_PRIORITY
                        QUARK_Task_Flag_Set(&task_flags, TASK_PRIORITY, QUARK_TASK_MIN_PRIORITY);
                        #endif
                    }
                    num_players = ceil( ((double)num_players)/2.0 );
                    skip = 2*skip;
                }
                #endif

#pragma omp taskwait
printf( " -*- T(%d,%d) -*-\n",2,1 );
for (int i=0; i<tempkn; i++) {
  for (int j=0; j<tempkn; j++) printf( "%.2e ",T(2,1)[i+j*A.mb] );
  printf( "\n" );
}
                #ifdef WITH_PRIORITY
                QUARK_Task_Flag_Set(&task_flags, TASK_PRIORITY, QUARK_TASK_MAX_PRIORITY-2);
                #endif
                core_omp_zlacpy(
                    PlasmaLower,
                    tempkn, tempkn,
                    A(k, k), ldak,
                    T(k, k), A.mb,
                    sequence, request);
#pragma omp taskwait
printf( " -** T(%d,%d) **-\n",2,1 );
for (int i=0; i<tempkn; i++) {
  for (int j=0; j<tempkn; j++) printf( "%.2e ",T(2,1)[i+j*A.mb] );
  printf( "\n" );
}
printf( " * T(%d,%d)\n",k,k );
for (int i=0; i<tempkn; i++) {
  for (int j=0; j<tempkn; j++) printf( "%.2e ",T(k,k)[i+j*A.mb] );
  printf( "\n" );
}
printf( " * W:\n" );
for (int i=0; i<tempkn; i++) {
  for (int j=0; j<tempkn; j++) printf( "%.2e ",W3(0)[i+j*A.nb] );
  printf( "\n" );
}
                core_omp_zgeadd(
                    PlasmaNoTrans, tempkn, tempkn,
                    zone, W3(0), A.nb,
                    zone, T(k, k), A.mb,
                    sequence, request);
#pragma omp taskwait
printf( " ---- T(%d,%d) ----\n",2,1 );
for (int i=0; i<tempkn; i++) {
  for (int j=0; j<tempkn; j++) printf( "%.2e ",T(2,1)[i+j*A.mb] );
  printf( "\n" );
}
                #ifdef WITH_PRIORITY
                QUARK_Task_Flag_Set(&task_flags, TASK_PRIORITY, QUARK_TASK_MIN_PRIORITY);
                #endif
            } else { /* k == 0 or 1 */
                core_omp_zlacpy(
                    PlasmaLower,
                    tempkn, tempkn,
                    A(k, k), ldak,
                    T(k, k), A.mb,
                    sequence, request);
                #pragma omp taskwait
                for (int j=0; j<tempkn; j++) {
                    for (int i=0; i<j; i++) {
                        T(k, k)[i+j*A.mb] = conj(T(k, k)[j+i*A.mb]);
                    }
                }
            }
#pragma omp taskwait
printf( " init: T(%d,%d)\n",k,k );
for (int i=0; i<tempkn; i++) {
  for (int j=0; j<tempkn; j++) printf( "%.2e ",T(k,k)[i+j*A.mb] );
  printf( "\n" );
}

            if (k > 0) {
                #ifdef WITH_PRIORITY
                QUARK_Task_Flag_Set(&task_flags, TASK_PRIORITY, QUARK_TASK_MAX_PRIORITY-2);
                #endif
                if (k > 1) {
#pragma omp taskwait
printf( " * L(%d,%d)\n",k,k );
for (int i=0; i<tempkn; i++) {
  for (int j=0; j<tempkn; j++) printf( "%.2e ",L(k,k)[i+j*A.mb] );
  printf( "\n" );
}
printf( " * T(%d,%d):\n",k,k-1 );
for (int i=0; i<tempkn; i++) {
  for (int j=0; j<tempkn; j++) printf( "%.2e ",T(k,k-1)[i+j*A.nb] );
  printf( "\n" );
}
                    core_omp_zgemm(
                        PlasmaNoTrans, PlasmaNoTrans,
                        tempkn, A.mb, tempkn,
                        zone,  L(k, k),   ldak,
                               T(k, k-1), A.mb,
                        zzero, W(0), A.mb,
                        sequence, request);
                    core_omp_zgemm(
                        PlasmaNoTrans, PlasmaConjTrans,
                        tempkn, tempkn, A.mb,
                        zmone, W(0), A.mb,
                               L(k, k-1), ldak,
                        zone,  T(k, k), A.mb,
                        sequence, request);
                }
                #ifdef WITH_PRIORITY
                QUARK_Task_Flag_Set(&task_flags, TASK_PRIORITY, QUARK_TASK_MIN_PRIORITY);
                #endif

                /* - symmetrically solve with L(k,k) */
                #ifdef WITH_PRIORITY
                QUARK_Task_Flag_Set(&task_flags, TASK_PRIORITY, QUARK_TASK_MAX_PRIORITY-1);
                #endif
#pragma omp taskwait
printf( " T(%d,%d)\n",k,k );
for (int i=0; i<tempkn; i++) {
  for (int j=0; j<tempkn; j++) printf( "%.2e ",T(k,k)[i+j*A.mb] );
  printf( "\n" );
}
                core_omp_zhegst(
                    1, PlasmaLower, tempkn,
                    T(k, k), A.mb,
                    L(k, k), ldak,
                    sequence, request);
                #ifdef WITH_PRIORITY
                QUARK_Task_Flag_Set(&task_flags, TASK_PRIORITY, QUARK_TASK_MIN_PRIORITY);
                #endif
                /* expanding to full matrix */
                #pragma omp taskwait
                for (int j=0; j<tempkn; j++) {
                    for (int i=0; i<j; i++) {
                        T(k, k)[i+j*A.mb] = conj(T(k, k)[j+i*A.mb]);
                    }
                }
            }

            /* computing H(k, k) */
            plasma_complex64_t beta = zzero;
#pragma omp taskwait
printf( " H(%d,%d)\n",k,k );
            if (k > 1) {
printf( " > T(%d,%d)\n",k,k-1 );
for (int i=0; i<A.mb; i++) {
  for (int j=0; j<A.mb; j++) printf( "%.2e ",T(k,k-1)[i+j*A.mb] );
  printf( "\n" );
}
printf( " > L(%d,%d)\n",k,k-1 );
for (int i=0; i<A.mb; i++) {
  for (int j=0; j<A.mb; j++) printf( "%.2e ",L(k,k-1)[i+j*A.mb] );
  printf( "\n" );
}
                core_omp_zgemm(
                    PlasmaNoTrans, PlasmaConjTrans,
                    tempkn, tempkn, A.nb,
                    zone,  T(k, k-1), A.mb,
                           L(k, k-1), ldak,
                    zzero, H(k, k), ldak,
                    sequence, request);
                beta = zone;
            }

            if (k+1 < A.nt) {
                if (k > 0) {
#pragma omp taskwait
printf( " beta = %.2e\n",beta );
printf( " > T(%d,%d)\n",k,k );
for (int i=0; i<A.mb; i++) {
  for (int j=0; j<A.mb; j++) printf( "%.2e ",T(k,k)[i+j*A.mb] );
  printf( "\n" );
}
printf( " > L(%d,%d)\n",k,k );
for (int i=0; i<A.mb; i++) {
  for (int j=0; j<A.mb; j++) printf( "%.2e ",L(k,k)[i+j*A.mb] );
  printf( "\n" );
}
                    core_omp_zgemm(
                        PlasmaNoTrans, PlasmaConjTrans,
                        tempkn, tempkn, tempkn,
                        zone,  T(k, k), A.mb,
                               L(k, k), ldak,
                        beta , H(k, k), ldak,
                        sequence, request);
                }
#pragma omp taskwait
printf( " >> H(%d,%d)\n",k,k );
for (int i=0; i<A.mb; i++) {
  for (int j=0; j<A.mb; j++) printf( "%.2e ",H(k,k)[i+j*ldak] );
  printf( "\n" );
}

                /* computing L(k+1:nt, k+1) from A(k+1:nt, k) *
                 * so the number of the column stays the same */
                int ldakp1 = plasma_tile_mmain(A, k+1);

                /* computing the (k+1)-th column of L */
                /* - update with the previous column */
printf( " mt-k=%d, max_threads=%d\n",A.mt-k,plasma->max_threads );
                if (A.mt-k < plasma->max_threads) {
                    int num = imin(k, tot/(A.mt-k-1)); /* workspace per row */
                    for (int j=1; j<=k; j++) {
                        int ldaj = plasma_tile_mmain(A, j);
                        int tempjn = j == A.nt-1 ? A.n-j*A.nb : A.nb;
                        for (int m = k+1; m < A.mt; m++) {
                            int tempmm = m == A.mt-1 ? A.m-m*A.mb : A.mb;
                            int ldam = plasma_tile_mmain(A, m);

                            int id = (m-k-1)*num+(j-1)%num;
                            plasma_complex64_t beta;
                            if (j < num+1) {
                                beta = zzero;
                            } else{
                                beta = zone;
                            }
                            if (j < num+1 || j > k-num) {
                                core_omp_zgemm(
                                    PlasmaNoTrans, PlasmaNoTrans,
                                    tempmm, tempkn, tempjn,
                                    zmone, L(m, j), ldam,
                                           H(j, k), ldaj,
                                    beta,  W4(id),  A.nb,
                                    sequence, request);
                            } else {
                                core_omp_zgemm(
                                    PlasmaNoTrans, PlasmaNoTrans,
                                    tempmm, tempkn, tempjn,
                                    zmone, L(m, j), ldam,
                                           H(j, k), ldaj,
                                    beta,  W4(id),  A.nb,
                                    sequence, request);
                            }
                        }
                    }
                    /* accumeration within workspace */
                    int skip  = 2;                                                   /* intervals between brackets */
                    int num_players = num;                                           /* number of players          */
                    int num_rounds = ceil( log10((double)num_players)/log10(2.0) );  /* height of tournament       */
                    for (int round=1; round<=num_rounds; round++) {
                        int num_brackets = num_players / 2; /* number of brackets */
                        for (int bracket=0; bracket<num_brackets; bracket++) {
                            /* first contendar */
                            int m1 = skip*bracket;
                            /* second contendar */
                            int m2 = skip*bracket+skip/2;

                            for (int m = k+1; m < A.mt; m++) {
                                int tempmm = m == A.mt-1 ? A.m-m*A.mb : A.mb;
                                core_omp_zgeadd(
                                    PlasmaNoTrans, tempmm, tempkn,
                                    zone, W4((m-k-1)*num+m2), A.nb,
                                    zone, W4((m-k-1)*num+m1), A.nb,
                                    sequence, request);
                            }
                         }
                        num_players = ceil( ((double)num_players)/2.0 );
                        skip = 2*skip;
                    }

                    /* accumelate into L(:,k+1) */
                    for (int m = k+1; m < A.mt; m++) {
                        int tempmm = m == A.mt-1 ? A.m-m*A.mb : A.mb;
                        int ldam = plasma_tile_mmain(A, m);
                        core_omp_zgeadd(
                            PlasmaNoTrans, tempmm, tempkn,
                            zone, W4((m-k-1)*num), A.nb,
                            zone, L(m, k+1), ldam,
                            sequence, request);
                    }
                } else {
                    for (int j=1; j<=k; j++) {
                        int ldaj = plasma_tile_mmain(A, j);
                        int tempjn = j == A.nt-1 ? A.n-j*A.nb : A.nb;
                        for (int m = k+1; m < A.mt; m++) {
                            int tempmm = m == A.mt-1 ? A.m-m*A.mb : A.mb;
                            int ldam = plasma_tile_mmain(A, m);
printf( " > L(%d,%d)\n",m,k+1 );
printf( " L(%d,%d)\n",m,j );
for (int ii=0; ii<A.mb; ii++) {
  for (int jj=0; jj<A.mb; jj++) printf( "%.2e ",L(m,j)[ii+jj*A.mb] );
  printf( "\n" );
}
printf( " H(%d,%d)\n",j,k );
for (int ii=0; ii<A.mb; ii++) {
  for (int jj=0; jj<A.mb; jj++) printf( "%.2e ",H(j,k)[ii+jj*A.mb] );
  printf( "\n" );
}
                            core_omp_zgemm(
                               PlasmaNoTrans, PlasmaNoTrans,
                               tempmm, tempkn, tempjn,
                               zmone, L(m, j),   ldam,
                                      H(j, k),   ldaj,
                               zone,  L(m, k+1), ldam,
                               sequence, request);
                        }
                    }
                }
                /* =========================== */
                /* ==  PLASMA recursive LU  == */
                /* =========================== */
                /* -- compute LU of the panel -- */
                int tempi = (k+1)*A.mb, tempj = k*A.nb;   // offset
                int tempn = tempkn, tempm = A.m - (k+1)*A.mb; // dimension

                #ifdef WITH_PRIORITY
                QUARK_Task_Flag_Set(&task_flagsP, TASK_PRIORITY, QUARK_TASK_MAX_PRIORITY);
                #endif

                plasma_complex64_t *a00, *a20;
                a00 = A(k, k);
                a20 = A(A.mt-1, k);

                int ma00k = (A.mt-k-1)*A.mb;
                int na00k = plasma_tile_nmain(A, k);
                int lda20 = plasma_tile_mmain(A, A.mt-1);

                int nvak = plasma_tile_nview(A, k);
                int mvak = plasma_tile_mview(A, k);
                int ldak = plasma_tile_mmain(A, k);

                #pragma omp taskwait
printf( " %d: zgetrf(%d:%d, %dx%d), threads=%d\n",k,tempi,tempj,tempm,tempn,num_panel_threads );
for (int i=0; i<nvak; i++) {
  for (int j=0; j<nvak; j++) printf( "%.2e ",A(k+1,k)[i+j*ldak] );
  printf( "\n" );
}

                #pragma omp task depend(inout:a00[0:ma00k*na00k]) \
                                 depend(inout:a20[0:lda20*nvak]) \
                                 depend(out:ipiv[k*A.mb:mvak]) /*\
                                 priority(1) */
                {
                    if (sequence->status == PlasmaSuccess) {
                        for (int rank = 0; rank < num_panel_threads; rank++) {
                            #pragma omp task // priority(1)
                            {
                                plasma_desc_t view =
                                    plasma_desc_view(A,
                                                     tempi, tempj,
                                                     tempm, tempn);

                                int info = core_zgetrf(view, IPIV(k+1), ib,
                                                       rank, num_panel_threads,
                                                       barrier);
                                if (info != 0)
                                    plasma_request_fail(sequence, request, k*A.mb+info);
                            }
                        }
                    }
                    #pragma omp taskwait
                }
                #pragma omp taskwait
                for (int i = 0; i < imin(tempm, tempn); i++) {
                    printf( " IPIV(%d)[%d] = %d\n",k+1,i,IPIV(k+1)[i] );
                    IPIV(k+1)[i] += tempi;
                }
                for (int i = 0; i < A.m; i++) printf( "ipiv[%d]=%d\n",i,ipiv[i] );

                if (sequence->status == PlasmaSuccess) {
                    int ii;
                    /*for (ii=0; ii<min(A.m,A.n); ii++) {
                        IPIV(k+1)[ii] += (iinfo-A.i);
                    }*/

                    for( ii=0; ii<tempm; ii++ ) perm[tempi+ii] = tempi+ii;
                    for( ii=0; ii<imin(tempm, tempn); ii++ ) {
                        int piv = perm[IPIV(k+1)[ii]-1];
                        perm[IPIV(k+1)[ii]-1] = perm[tempi+ii];
                        perm[tempi+ii] = piv;
                    }
                    int npiv = 0;
                    for( ii=0; ii<tempm; ii++ ) {
                        if( perm[tempi+ii] != tempi+ii ) {
                            perm2work[tempi+ii] = npiv;
                            npiv ++;
                        } else {
                            perm2work[tempi+ii] = -1;
                        }
                        iperm[perm[tempi+ii]] = tempi+ii;
                    }
                    for( ii=0; ii<tempm; ii++ ) iperm2work[tempi+ii] = perm2work[iperm[tempi+ii]];
                }
                #ifdef WITH_PRIORITY
                QUARK_Task_Flag_Set(&task_flagsP, TASK_PRIORITY, QUARK_TASK_MIN_PRIORITY);
                #endif
                /* -- apply pivoting to previous columns of L -- */
                for (int n = 1; n < k+1; n++)
                {
                   int *ipivk = IPIV(k+1);
                   plasma_complex64_t *akk = L(k+1, n);
                   int k1 = 1+(k+1)*A.nb;
                   int k2 = imin(tempm,tempn)+(k+1)*A.nb;
                   int tempi  = 0;
                   int tempj  = (n-1)*A.nb;
                   int tempkn = n == A.nt-1 ? A.n-n*A.nb : A.nb;
                   tempm = A.m;

                    #pragma omp task depend(in:ipivk[0:k2]) \
                                     depend(inout:akk[0:tempm*tempkn])
                    {
                        if (sequence->status == PlasmaSuccess) {
#pragma omp taskwait
printf( " swap(A(%d,%d), %dx%d, %d..%d\n",tempi,tempj,tempm,tempkn,k1,k2 );
printf( " before swap(%d,%d):\n",A.mt-1,n-1 );
for (int i=0; i<A.mb; i++) {
  for (int j=0; j<A.mb; j++) printf( "%.2e ",A(A.mt-1,n-1)[i+j*A.mb] );
  printf( "\n" );
}
                            plasma_desc_t view =
                                plasma_desc_view(A, tempi, tempj, tempm, tempkn);
                            core_zlaswp(PlasmaRowwise, view, k1, k2, ipiv, 1);
#pragma omp taskwait
printf( " after swap:\n" );
for (int i=0; i<A.mb; i++) {
  for (int j=0; j<A.mb; j++) printf( "%.2e ",A(A.mt-1,n-1)[i+j*A.mb] );
  printf( "\n" );
}
                        }
                    }
                }
                #pragma omp taskwait

                /* -- symmetrically apply pivoting to trailing A -- */
printf( " before sym-pivot:\n" );
for (int i=0; i<nvak; i++) {
  for (int j=0; j<nvak; j++) printf( "%.2e ",A(k+1,k+1)[i+j*ldak] );
  printf( "\n" );
}
                core_omp_zlaswp_sym(PlasmaLower, k, ib,
                                    A, W,
                                    iperm, iperm2work, perm2work,
                                    sequence, request);
#pragma omp taskwait
printf( " after sym-pivot:\n" );
for (int i=0; i<nvak; i++) {
  for (int j=0; j<nvak; j++) printf( "%.2e ",A(k+1,k+1)[i+j*ldak] );
  printf( "\n" );
}

                /* ================================== */
                /* ==  end of PLASMA recursive LU  == */
                /* ================================== */

                /* computing T(k+1, k) */
                tempn = k == A.nt-1 ? A.n-k*A.nb : A.nb;
                tempm = (k+1) == A.nt-1 ? A.n-(k+1)*A.nb : A.nb; // dimension
                /* copy upper-triangular part of L(k+1,k+1) to T(k+1,k) */
                /* and then zero it out                                 */
                core_omp_zlacpy(
                        PlasmaUpper,
                        tempm, tempn,
                        L(k+1, k+1),  ldakp1,
                        T(k+1, k  ),  A.mb,
                        sequence, request);
                int ldakp1_n = plasma_tile_nmain(A, k+2);
                core_omp_zlaset(
                        PlasmaUpper,
                        ldakp1, ldakp1_n, 0, 0,
                        tempm, tempn,
                        zzero, zone,
                        L(k+1, k+1));
#pragma omp taskwait
printf( " T(%d,%d)\n",k+1,k );
for (int i=0; i<A.mb; i++) {
  for (int j=0; j<A.mb; j++) printf( "%.2e ",T(k+1,k)[i+j*A.mb] );
  printf( "\n" );
}
printf( "\n" );
                if (k > 0) {
                    core_omp_ztrsm(
                        PlasmaRight, PlasmaLower,
                        PlasmaConjTrans, PlasmaUnit,
                        tempm, tempn,
                        zone, L(k,   k), ldak,
                              T(k+1, k), A.mb,
                        sequence, request);
                }
#pragma omp taskwait
for (int i=0; i<A.mb; i++) {
  for (int j=0; j<A.mb; j++) printf( "%.2e ",T(k+1,k)[i+j*A.mb] );
  printf( "\n" );
}
            }
        }
    }
    //==============
    // PlasmaUpper
    //==============
    else {
        for (int k = 0; k < A.nt; k++) {
            int nvak = plasma_tile_nview(A, k);
            int ldak = plasma_tile_mmain(A, k);
            core_omp_zpotrf(
                PlasmaUpper, nvak,
                A(k, k), ldak,
                A.nb*k,
                sequence, request);

            for (int m = k+1; m < A.nt; m++) {
                int nvam = plasma_tile_nview(A, m);
                core_omp_ztrsm(
                    PlasmaLeft, PlasmaUpper,
                    PlasmaConjTrans, PlasmaNonUnit,
                    A.nb, nvam,
                    1.0, A(k, k), ldak,
                         A(k, m), ldak,
                    sequence, request);
            }
            for (int m = k+1; m < A.nt; m++) {
                int nvam = plasma_tile_nview(A, m);
                int ldam = plasma_tile_mmain(A, m);
                core_omp_zherk(
                    PlasmaUpper, PlasmaConjTrans,
                    nvam, A.mb,
                    -1.0, A(k, m), ldak,
                     1.0, A(m, m), ldam,
                    sequence, request);

                for (int n = k+1; n < m; n++) {
                    int ldan = plasma_tile_mmain(A, n);
                    core_omp_zgemm(
                        PlasmaConjTrans, PlasmaNoTrans,
                        A.mb, nvam, A.mb,
                        -1.0, A(k, n), ldak,
                              A(k, m), ldak,
                         1.0, A(n, m), ldan,
                        sequence, request);
                }
            }
        }
    }
}
