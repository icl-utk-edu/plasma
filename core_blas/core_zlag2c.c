/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions mixed zc -> ds
 *
 **/

#include <plasma_core_blas.h>
#include "core_lapack.h"
#include "plasma_types.h"

/***************************************************************************//**
 *
 * @ingroup core_lag2
 *
 *  Converts m-by-n matrix A from double complex to single complex precision.
 *
 *******************************************************************************
 *
 * @param[in] m
 *          The number of rows of the matrix A.
 *          m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix A.
 *          n >= 0.
 *
 * @param[in] A
 *          The lda-by-n matrix in double complex precision to convert.
 *
 * @param[in] lda
 *          The leading dimension of the matrix A.
 *          lda >= max(1,m).
 *
 * @param[out] As
 *          On exit, the converted ldas-by-n matrix in single complex precision.
 *
 * @param[in] ldas
 *          The leading dimension of the matrix As.
 *          ldas >= max(1,m).
 *
 ******************************************************************************/
__attribute__((weak))
void plasma_core_zlag2c(int m, int n,
                 plasma_complex64_t *A,  int lda,
                 plasma_complex32_t *As, int ldas)
{
    LAPACKE_zlag2c_work(LAPACK_COL_MAJOR, m, n, A, lda, As, ldas);
}

/******************************************************************************/
void plasma_core_omp_zlag2c(int m, int n,
                     plasma_complex64_t *A,  int lda,
                     plasma_complex32_t *As, int ldas,
                     plasma_sequence_t *sequence, plasma_request_t *request)
{
    #pragma omp task depend(in:A[0:lda*n]) \
                     depend(out:As[0:ldas*n])
    {
        if (sequence->status == PlasmaSuccess)
            plasma_core_zlag2c(m, n, A, lda, As, ldas);
    }
}
