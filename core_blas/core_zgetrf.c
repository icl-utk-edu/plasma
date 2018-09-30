/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> c d s
 *
 **/

#include <plasma_core_blas.h>
#include "core_lapack.h"
#include "plasma_barrier.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_types.h"

#include <omp.h>
#include <assert.h>
#include <math.h>

#define A(m, n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)

/******************************************************************************/
__attribute__((weak))
void plasma_core_zgetrf(plasma_desc_t A, int *ipiv, int ib, int rank, int size,
                 volatile int *max_idx, volatile plasma_complex64_t *max_val,
                 volatile int *info, plasma_barrier_t *barrier)
{
    double sfmin = LAPACKE_dlamch_work('S');
    for (int k = 0; k < imin(A.m, A.n); k += ib) {
        int kb = imin(imin(A.m, A.n)-k, ib);

        plasma_complex64_t *a0 = A(0, 0);
        int lda0 = plasma_tile_mmain(A, 0);
        int mva0 = plasma_tile_mview(A, 0);
        int nva0 = plasma_tile_nview(A, 0);

        //======================
        // panel factorization
        //======================
        for (int j = k; j < k+kb; j++) {
            // pivot search
            max_idx[rank] = 0;
            max_val[rank] = a0[j+j*lda0];

            for (int l = rank; l < A.mt; l += size) {
                plasma_complex64_t *al = A(l, 0);
                int ldal = plasma_tile_mmain(A, l);
                int mval = plasma_tile_mview(A, l);

                if (l == 0) {
                    for (int i = 1; i < mva0-j; i++)
                        if (plasma_core_dcabs1(a0[j+i+j*lda0]) >
                            plasma_core_dcabs1(max_val[rank])) {

                            max_val[rank] = a0[j+i+j*lda0];
                            max_idx[rank] = i;
                        }
                }
                else {
                    for (int i = 0; i < mval; i++)
                        if (plasma_core_dcabs1(al[i+j*ldal]) >
                            plasma_core_dcabs1(max_val[rank])) {

                            max_val[rank] = al[i+j*ldal];
                            max_idx[rank] = A.mb*l+i-j;
                        }
                }
            }

            plasma_barrier_wait(barrier, size);
            if (rank == 0)
            {
                // max reduction
                for (int i = 1; i < size; i++) {
                    if (plasma_core_dcabs1(max_val[i]) >
                        plasma_core_dcabs1(max_val[0])) {
                        max_val[0] = max_val[i];
                        max_idx[0] = max_idx[i];
                    }
                }

                // pivot adjustment
                int jp = j+max_idx[0];
                ipiv[j] = jp-k+1;

                // singularity check
                if (*info == 0 && max_val[0] == 0.0) {
                    *info = j+1;
                }
                else {
                    // pivot swap
                    if (jp != j) {
                        plasma_complex64_t *ap = A(jp/A.mb, 0);
                        int ldap = plasma_tile_mmain(A, jp/A.mb);

                        cblas_zswap(kb,
                                    &a0[j+k*lda0], lda0,
                                    &ap[jp%A.mb+k*ldap], ldap);
                    }
                }
            }
            plasma_barrier_wait(barrier, size);

            // column scaling and trailing update (all ranks)
            for (int l = rank; l < A.mt; l += size) {
                plasma_complex64_t *al = A(l, 0);
                int ldal = plasma_tile_mmain(A, l);
                int mval = plasma_tile_mview(A, l);

                if (*info == 0) {
                    // column scaling
                    if (cabs(a0[j+j*lda0]) >= sfmin) {
                        if (l == 0) {
                            for (int i = 1; i < mva0-j; i++)
                                a0[j+i+j*lda0] /= a0[j+j*lda0];
                        }
                        else {
                            for (int i = 0; i < mval; i++)
                                al[i+j*ldal] /= a0[j+j*lda0];
                        }
                    }
                    else {
                        plasma_complex64_t scal = 1.0/a0[j+j*lda0];
                        if (l == 0)
                            cblas_zscal(mva0-j-1, CBLAS_SADDR(scal),
                                        &a0[j+1+j*lda0], 1);
                        else
                            cblas_zscal(mval, CBLAS_SADDR(scal),
                                        &al[j*ldal], 1);
                    }
                }

                // trailing update
                plasma_complex64_t zmone = -1.0;
                if (l == 0) {
                    cblas_zgeru(CblasColMajor,
                                mva0-j-1, k+kb-j-1,
                                CBLAS_SADDR(zmone), &a0[j+1+j*lda0], 1,
                                                    &a0[j+(j+1)*lda0], lda0,
                                                    &a0[j+1+(j+1)*lda0], lda0);
                }
                else {
                    cblas_zgeru(CblasColMajor,
                                mval, k+kb-j-1,
                                CBLAS_SADDR(zmone), &al[+j*ldal], 1,
                                                    &a0[j+(j+1)*lda0], lda0,
                                                    &al[+(j+1)*ldal], ldal);
                }
            }
            plasma_barrier_wait(barrier, size);
        }

        //===================================
        // right pivoting and trsm (rank 0)
        //===================================
        plasma_barrier_wait(barrier, size);
        if (rank == 0) {
            // pivot adjustment
            for (int i = k+1; i <= imin(A.m, k+kb); i++)
                ipiv[i-1] += k;

            // right pivoting
            for (int i = k; i < k+kb; i++) {
                plasma_complex64_t *ap = A((ipiv[i]-1)/A.mb, 0);
                int ldap = plasma_tile_mmain(A, (ipiv[i]-1)/A.mb);

                cblas_zswap(nva0-k-kb,
                            &a0[i+(k+kb)*lda0], lda0,
                            &ap[(ipiv[i]-1)%A.mb+(k+kb)*ldap], ldap);
            }
            // trsm
            plasma_complex64_t zone = 1.0;
            cblas_ztrsm(CblasColMajor,
                        CblasLeft, CblasLower,
                        CblasNoTrans, CblasUnit,
                        kb,
                        nva0-k-kb,
                        CBLAS_SADDR(zone), &a0[k+k*lda0], lda0,
                                           &a0[k+(k+kb)*lda0], lda0);
        }
        plasma_barrier_wait(barrier, size);

        //===================
        // gemm (all ranks)
        //===================
        plasma_complex64_t zone = 1.0;
        plasma_complex64_t zmone = -1.0;
        for (int i = rank; i < A.mt; i += size) {
            plasma_complex64_t *ai = A(i, 0);
            int mvai = plasma_tile_mview(A, i);
            int ldai = plasma_tile_mmain(A, i);

            if (i == 0) {
                cblas_zgemm(CblasColMajor,
                            CblasNoTrans, CblasNoTrans,
                            mva0-k-kb,
                            nva0-k-kb,
                            kb,
                            CBLAS_SADDR(zmone), &a0[k+kb+k*lda0], lda0,
                                                &a0[k+(k+kb)*lda0], lda0,
                            CBLAS_SADDR(zone),  &a0[(k+kb)+(k+kb)*lda0], lda0);
            }
            else {
                cblas_zgemm(CblasColMajor,
                            CblasNoTrans, CblasNoTrans,
                            mvai,
                            nva0-k-kb,
                            kb,
                            CBLAS_SADDR(zmone), &ai[k*ldai], ldai,
                                                &a0[k+(k+kb)*lda0], lda0,
                            CBLAS_SADDR(zone),  &ai[(k+kb)*ldai], ldai);
            }
        }
        plasma_barrier_wait(barrier, size);
    }

    //============================
    // left pivoting (all ranks)
    //============================
    for (int k = ib; k < imin(A.m, A.n); k += ib) {
        if (k%ib == rank) {
            for (int i = k; i < imin(A.m, A.n); i++) {
                plasma_complex64_t *ai = A(i/A.mb, 0);
                plasma_complex64_t *ap = A((ipiv[i]-1)/A.mb, 0);
                int ldai = plasma_tile_mmain(A, (i/A.mb));
                int ldap = plasma_tile_mmain(A, (ipiv[i]-1)/A.mb);

                cblas_zswap(ib,
                            &ai[i%A.mb+(k-ib)*ldai], ldai,
                            &ap[(ipiv[i]-1)%A.mb+(k-ib)*ldap], ldap);
            }
        }
    }
}
