/**
 *
 * @file core_zlacpy_band.c
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> c d s
 *
 **/

#include "core_blas.h"
#include "plasma_types.h"
#include "plasma_internal.h"

#ifdef PLASMA_WITH_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

/*******************************************************************************
 *
 * @ingroup CORE_PLASMA_Complex64_t
 *
 *  CORE_zlacpy copies all or part of a two-dimensional matrix A to another
 *  matrix B
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          Specifies the part of the matrix A to be copied to B.
 *            = PlasmaUpperLower: All the matrix A
 *            = PlasmaUpper: Upper triangular part
 *            = PlasmaLower: Lower triangular part
 *
 * @param[in] M
 *          The number of rows of the matrices A and B. M >= 0.
 *
 * @param[in] N
 *          The number of columns of the matrices A and B. N >= 0.
 *
 * @param[in] A
 *          The M-by-N matrix to copy.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,M).
 *
 * @param[out] B
 *          The M-by-N copy of the matrix A.
 *          On exit, B = A ONLY in the locations specified by uplo.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1,M).
 *
 ******************************************************************************/
void CORE_zlacpy_lapack2tile_band(int K, int J, int M, int N, int NB, int KL, int KU,
                                  const PLASMA_Complex64_t *A, int lda,
                                        PLASMA_Complex64_t *B, int ldb)
{
    int i, j;
    PLASMA_Complex64_t zzero = 0.0;

    int j_start = 0; /* pivot back and could fill in */
    int j_end = (J <= K ? N : imin(N, (K-J)*NB+M+KU+KL+1));
    for (j=0; j<j_start; j++) {
        for (i=0; i<M; i++) {
            B[i + j*ldb] = zzero;
        }
    }
    for (j=j_start; j<j_end; j++) {
        int i_start = (J <= K ? 0 : imax(0, (J-K)*NB+j-KU-KL));
        int i_end = (J >= K ? M : imin(M, (J-K)*NB+j+KL+NB+1)); /* +NB because we use zgetrf on panel and pivot back within the panel.
                                                             *  so the last tile in panel could fill.  */
        for (i=0; i<i_start; i++) {
            B[i + j*ldb] = zzero;
        }
        for (i=i_start; i<i_end; i++) {
            B[i + j*ldb] = A[i + j*lda];
        }
        for (i=i_end; i<M; i++) {
            B[i + j*ldb] = zzero; 
        }
    }
    for (j=j_end; j<N; j++) {
        for (i=0; i<M; i++) {
            B[i + j*ldb] = zzero;
        }
    }
}

void CORE_OMP_zlacpy_lapack2tile_band(int k, int j, int m, int n, int nb, int kl, int ku,
                                      const PLASMA_Complex64_t *A, int lda,
                                            PLASMA_Complex64_t *B, int ldb) 
{
    #pragma omp task depend(in:A[0:m*n]) depend(out:B[0:m*n])
    CORE_zlacpy_lapack2tile_band(k, j, m, n, nb, kl, ku, A, lda, B, ldb);
}


/******************************************************************************
 *
 ******************************************************************************/
void CORE_zlacpy_tile2lapack_band(int K, int J, int M, int N, int NB, int KL, int KU,
                                  const PLASMA_Complex64_t *B, int ldb,
                                        PLASMA_Complex64_t *A, int lda)
{
    int i, j;
    PLASMA_Complex64_t zzero = 0.0;

    int j_start = 0; /* pivot back and could fill in */
    int j_end = (J <= K ? N : imin(N, (K-J)*NB+M+KU+KL+1));
    for (j=j_start; j<j_end; j++) {
        int i_start = (J <= K ? 0 : imax(0, (J-K)*NB+j-KU-KL));
        int i_end = (J >= K ? M : imin(M, (J-K)*NB+j+KL+NB+1)); /* +NB because we use zgetrf on panel and pivot back within the panel.
                                                              *  so the last tile in panel could fill.  */
        for (i=i_start; i<i_end; i++) {
            A[i + j*lda] = B[i + j*ldb];
        }
    }
}

void CORE_OMP_zlacpy_tile2lapack_band(int k, int j, int m, int n, int nb, int kl, int ku,
                                      const PLASMA_Complex64_t *B, int ldb,
                                            PLASMA_Complex64_t *A, int lda) 
{
    #pragma omp task depend(in:B[0:m*n]) depend(out:A[0:m*n])
    CORE_zlacpy_tile2lapack_band(k, j, m, n, nb, kl, ku, B, ldb, A, lda);
}
