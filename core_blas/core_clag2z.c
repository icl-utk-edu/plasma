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
 *  Converts m-by-n matrix A from single complex to double complex precision.
 *
 *******************************************************************************
 *
 * @param[in] m
 *          The number of rows of the matrix As.
 *          m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix As.
 *          n >= 0.
 *
 * @param[in] As
 *          The ldas-by-n matrix in single complex precision to convert.
 *
 * @param[in] ldas
 *          The leading dimension of the matrix As.
 *          ldas >= max(1,m).
 *
 * @param[out] A
 *          On exit, the converted lda-by-n matrix in double complex precision.
 *
 * @param[in] lda
 *          The leading dimension of the matrix A.
 *          lda >= max(1,m).
 *
 ******************************************************************************/
__attribute__((weak))
void plasma_core_clag2z(int m, int n,
                 plasma_complex32_t *As, int ldas,
                 plasma_complex64_t *A,  int lda)
{
    LAPACKE_clag2z_work(LAPACK_COL_MAJOR, m, n, As, ldas, A, lda);
}

/******************************************************************************/
void plasma_core_omp_clag2z(int m, int n,
                     plasma_complex32_t *As, int ldas,
                     plasma_complex64_t *A,  int lda,
                     plasma_sequence_t *sequence, plasma_request_t *request)
{
    #pragma omp task depend(in:As[0:ldas*n]) \
                     depend(out:A[0:lda*n])
    {
        if (sequence->status == PlasmaSuccess)
            plasma_core_clag2z(m, n, As, ldas, A, lda);
    }
}
