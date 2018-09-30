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
#include <math.h>

#include <plasma_core_blas.h>
#include "plasma_types.h"
#include "plasma_internal.h"
#include "core_lapack.h"

#define COMPLEX
#define A(m,n) ((plasma_complex64_t*)plasma_tile_addr(A, m, n))

/***************************************************************************//**
 *
 * @ingroup core_heswp
 *
 *  Applies symmetric pivoting.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaGeneral: entire A,
 *          - PlasmaUpper:   upper triangle,
 *          - PlasmaLower:   lower triangle.
 *
 * @param[in] A
 *          The matrix to be pivoted.
 *
 * @param[in] k1
 *          The first element of IPIV for which a row interchange will
 *          be done.
 *
 * @param[in] k2
 *          The last element of IPIV for which a row interchange will
 *          be done.
 *
 * @param[in] ipiv
 *          The vector of pivot indices.  Only the elements in positions
 *          K1 through K2 of IPIV are accessed.
 *          IPIV(K) = L implies rows K and L are to be interchanged.
 *
 * @param[in] incx
 *          The increment between successive values of IPIV.  If IPIV
 *          is negative, the pivots are applied in reverse order.
 *
 ******************************************************************************/
__attribute__((weak))
void plasma_core_zheswp(int rank, int num_threads,
                 int uplo, plasma_desc_t A, int k1, int k2, const int *ipiv,
                 int incx, plasma_barrier_t *barrier)
{
    if (uplo == PlasmaLower) {
        if (incx > 0) {
            for (int i = k1-1; i <= k2-1; i += incx) {
                if (ipiv[i]-1 != i) {
                    int p1 = i;
                    int p2 = ipiv[i]-1;

                    int i1 = p1%A.mb;
                    int i2 = p2%A.mb;
                    int m1 = p1/A.mb;
                    int m2 = p2/A.mb;
                    int lda1 = plasma_tile_mmain(A, m1);
                    int lda2 = plasma_tile_mmain(A, m2);


                    int i1p1 = (p1+1)%A.mb;
                    int i2p1 = (p2+1)%A.mb;
                    int m1p1 = (p1+1)/A.mb;
                    int m2p1 = (p2+1)/A.mb;
                    int lda1p1 = plasma_tile_mmain(A, m1p1);
                    int lda2p1 = plasma_tile_mmain(A, m2p1);

                    // swap rows of previous column (assuming (k1,k2) stay within a tile)
                    if (i > k1-1 && rank == 0) {
                        cblas_zswap(i-(k1-1),
                                    A(m1, m1) + i1, lda1,
                                    A(m2, m1) + i2, lda2);
                    }

                    // swap columns p1 and p2
                    int mvam = plasma_tile_mview(A, m2p1);
                    if (mvam > i2+1 && rank == 1%num_threads) {
                        // between first tiles A(p2,p1) and A(p2,p2)
                        cblas_zswap(mvam-(i2+1),
                                    A(m2p1, m1) + i2p1 + i1*lda2p1, 1,
                                    A(m2p1, m2) + i2p1 + i2*lda2p1, 1);
                    }
                    int ell = ceil(((double)A.mt - (m1+1))/((double)num_threads));
                    int k_start = m1+1 + rank*ell;
                    int k_end   = imin(k_start+ell, A.mt);
                    for (int k = imax(m2+1, k_start); k < k_end; k++) {
                        int mvak = plasma_tile_mview(A, k);
                        int ldak = plasma_tile_mmain(A, k);
                        cblas_zswap(mvak,
                                    A(k, m1) + i1*ldak, 1,
                                    A(k, m2) + i2*ldak, 1);
                    }

                    // sym swap
                    mvam = plasma_tile_mview(A, m1);
                    if (imin(mvam,p2-(k1-1)) > i1+1 && rank == 2%num_threads) {
                        #ifdef COMPLEX
                        LAPACKE_zlacgv_work(imin(mvam,p2-(k1-1))-(i1+1), A(m1p1, m1) + i1p1 + i1*lda1p1, 1);
                        LAPACKE_zlacgv_work(imin(mvam,p2-(k1-1))-(i1+1), A(m2, m1p1) + i2 + i1p1*lda2, lda2);
                        #endif
                        cblas_zswap(imin(mvam,p2-(k1-1))-(i1+1),
                                    A(m1p1, m1) + i1p1 + i1*lda1p1, 1,
                                    A(m2, m1p1) + i2 + i1p1*lda2, lda2);
                    }
                    for (int k = k_start; k <= imin(k_end-1, m2); k++) {
                        int mvak = plasma_tile_mview(A, k);
                        int ldak = plasma_tile_mmain(A, k);
                        #ifdef COMPLEX
                        LAPACKE_zlacgv_work(imin(mvak, (p2-1)-k*A.mb+1), A(k, m1) +  i1*ldak, 1);
                        LAPACKE_zlacgv_work(imin(mvak, (p2-1)-k*A.mb+1), A(m2, k) +  i2, lda2);
                        #endif
                        cblas_zswap(imin(mvak, (p2-1)-k*A.mb+1),
                                    A(k, m1) +  i1*ldak, 1,
                                    A(m2, k) +  i2, lda2);
                    }

                    if (rank == 3%num_threads) {
                        #ifdef COMPLEX
                        LAPACKE_zlacgv_work(1, A(m2, m1) +  i2 + i1*lda2, 1);
                        #endif

                        // swap diagonal
                        cblas_zswap(1,
                                    A(m1, m1) + i1 + i1*lda1, lda1,
                                    A(m2, m2) + i2 + i2*lda2, lda2);
                    }
                }
                plasma_barrier_wait(barrier, num_threads);
            }
        }
    }
}
