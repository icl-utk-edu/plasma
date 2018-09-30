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
#include "plasma_types.h"
#include "plasma_internal.h"
#include "core_lapack.h"

/*******************************************************************************
 *
 * @ingroup core_plasma_complex64_t
 *
 *  plasma_core_zlacpy copies a sub-block A of a band matrix stored in LAPACK's band format
 *  to a corresponding sub-block B of a band matrix in PLASMA's band format
 *
 *******************************************************************************
 *
 * @param[in] it
 *          The row block index of the tile.
 *
 * @param[in] jt
 *          The column block index of the tile.
 *
 * @param[in] m
 *          The number of rows of the matrices A and B. M >= 0.
 *
 * @param[in] n
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
__attribute__((weak))
void plasma_core_zlacpy_lapack2tile_band(plasma_enum_t uplo,
                                  int it, int jt,
                                  int m, int n, int nb, int kl, int ku,
                                  const plasma_complex64_t *A, int lda,
                                        plasma_complex64_t *B, int ldb)
{
    int i, j;
    int j_start, j_end;
    if (uplo == PlasmaGeneral) {
        j_start = 0; // pivot back and could fill in
        j_end = (jt <= it ? n : imin(n, (it-jt)*nb+m+ku+kl+1));
    }
    else if (uplo == PlasmaUpper) {
        j_start = 0;
        j_end = imin(n, (it-jt)*nb+m+ku+1);
    }
    else {
        j_start = imax(0, (it-jt)*nb-kl);
        j_end = n;
    }

    for (j = 0; j < j_start; j++) {
        for (i = 0; i < m; i++) {
            B[i + j*ldb] = 0.0;
        }
    }
    for (j = j_start; j < j_end; j++) {
        int i_start, i_end;
        if (uplo == PlasmaGeneral) {
            i_start = (jt <= it ? 0 : imax(0, (jt-it)*nb+j-ku-kl));
            i_end = (jt >= it ? m : imin(m, (jt-it)*nb+j+kl+nb+1));
            // +nb because we use zgetrf on panel and pivot back within the panel.
            //  so the last tile in panel could fill.
        }
        else if (uplo == PlasmaUpper) {
            i_start = imax(0, (jt-it)*nb+j-ku);
            i_end = imin(m, (jt-it)*nb+j+1);
        }
        else {
            i_start = imax(0, (jt-it)*nb+j);
            i_end = imin(m, (jt-it)*nb+j+kl+1);
        }

        for (i = 0; i < i_start; i++) {
            B[i + j*ldb] = 0.0;
        }
        for (i = i_start; i < i_end; i++) {
            B[i + j*ldb] = A[i + j*lda];
        }
        for (i = i_end; i < m; i++) {
            B[i + j*ldb] = 0.0;
        }
    }
    for (j = j_end; j < n; j++) {
        for (i = 0; i < m; i++) {
            B[i + j*ldb] = 0.0;
        }
    }
}

/******************************************************************************/
void plasma_core_omp_zlacpy_lapack2tile_band(plasma_enum_t uplo,
                                      int it, int jt,
                                      int m, int n, int nb, int kl, int ku,
                                      const plasma_complex64_t *A, int lda,
                                            plasma_complex64_t *B, int ldb)
{
    #pragma omp task depend(in:A[0:lda*n]) \
                     depend(out:B[0:ldb*n])
    plasma_core_zlacpy_lapack2tile_band(uplo,
                                 it, jt, m, n, nb, kl, ku,
                                 A, lda,
                                 B, ldb);
}

/*******************************************************************************
 *
 * @ingroup core_plasma_complex64_t
 *
 *  plasma_core_zlacpy copies all or part of a two-dimensional matrix A to another
 *  matrix B
 *
 *******************************************************************************
 *
 * @param[in] it
 *          The row block index of the tile.
 *
 * @param[in] jt
 *          The column block index of the tile.
 *
 * @param[in] m
 *          The number of rows of the matrices A and B. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrices A and B. n >= 0.
 *
 * @param[in] A
 *          The m-by-n matrix to copy.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, m).
 *
 * @param[out] B
 *          The m-by-n copy of the matrix A.
 *          On exit, B = A ONLY in the locations specified by uplo.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1, m).
 *
 ******************************************************************************/
__attribute__((weak))
void plasma_core_zlacpy_tile2lapack_band(plasma_enum_t uplo,
                                  int it, int jt,
                                  int m, int n, int nb, int kl, int ku,
                                  const plasma_complex64_t *B, int ldb,
                                        plasma_complex64_t *A, int lda)
{
    int i, j;
    int j_start, j_end;

    if (uplo == PlasmaGeneral) {
        j_start = 0; // pivot back and could fill in
        j_end = (jt <= it ? n : imin(n, (it-jt)*nb+m+ku+kl+1));
    }
    else if (uplo == PlasmaUpper) {
        j_start = 0;
        j_end = imin(n, (it-jt)*nb+m+ku+1);
    }
    else {
        j_start = imax(0, (it-jt)*nb-kl);
        j_end = n;
    }

    for (j = j_start; j < j_end; j++) {
        int i_start, i_end;

        if (uplo == PlasmaGeneral) {
            i_start = (jt <= it ? 0 : imax(0, (jt-it)*nb+j-ku-kl));
            i_end = (jt >= it ? m : imin(m, (jt-it)*nb+j+kl+nb+1));
            // +nb because we use zgetrf on panel and pivot back within the panel.
            //  so the last tile in panel could fill.
        }
        else if (uplo == PlasmaUpper) {
            i_start = imax(0, (jt-it)*nb+j-ku);
            i_end = imin(m, (jt-it)*nb+j+1);
        }
        else {
            i_start = imax(0, (jt-it)*nb+j);
            i_end = imin(m, (jt-it)*nb+j+kl+1);
        }

        for (i = i_start; i < i_end; i++) {
            A[i + j*lda] = B[i + j*ldb];
        }
    }
}

/******************************************************************************/
void plasma_core_omp_zlacpy_tile2lapack_band(plasma_enum_t uplo,
                                      int it, int jt,
                                      int m, int n, int nb, int kl, int ku,
                                      const plasma_complex64_t *B, int ldb,
                                            plasma_complex64_t *A, int lda)
{
    #pragma omp task depend(in:B[0:ldb*n]) \
                     depend(out:A[0:lda*n])
    plasma_core_zlacpy_tile2lapack_band(uplo,
                                 it, jt, m, n, nb, kl, ku,
                                 B, ldb,
                                 A, lda);
}
