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
#include "core_lapack.h"

#include <math.h>

/***************************************************************************//**
 *
 * @ingroup core_lange
 *
 *  Calculates max, one, infinity or Frobenius norm of a given matrix.
 *
 *******************************************************************************
 *
 * @param[in] norm
 *          - PlasmaMaxNorm: Max norm
 *          - PlasmaOneNorm: One norm
 *          - PlasmaInfNorm: Infinity norm
 *          - PlasmaFrobeniusNorm: Frobenius norm
 *
 * @param[in] m
 *          The number of rows of the matrix A. m >= 0. When m = 0,
 *          the returned value is set to zero.
 *
 * @param[in] n
 *          The number of columns of the matrix A. n >= 0. When n = 0,
 *          the returned value is set to zero.
 *
 * @param[in] A
 *          The m-by-n matrix A.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,m).
 *
 * @param[in] work
 *          The auxiliary work array.
 *
 * @param[out] value
 *          The specified norm of the given matrix A
 *
 ******************************************************************************/
__attribute__((weak))
void plasma_core_zlange(plasma_enum_t norm, int m, int n,
                 const plasma_complex64_t *A, int lda,
                 double *work, double *value)
{
    *value = LAPACKE_zlange_work(LAPACK_COL_MAJOR,
                                 lapack_const(norm),
                                 m, n, A, lda, work);
}

/******************************************************************************/
void plasma_core_omp_zlange(plasma_enum_t norm, int m, int n,
                     const plasma_complex64_t *A, int lda,
                     double *work, double *value,
                     plasma_sequence_t *sequence, plasma_request_t *request)
{
    #pragma omp task depend(in:A[0:lda*n]) \
                     depend(out:value[0:1])
    {
        if (sequence->status == PlasmaSuccess)
            plasma_core_zlange(norm, m, n, A, lda, work, value);
    }
}

/******************************************************************************/
void plasma_core_omp_zlange_aux(plasma_enum_t norm, int m, int n,
                         const plasma_complex64_t *A, int lda,
                         double *value,
                         plasma_sequence_t *sequence, plasma_request_t *request)
{
    switch (norm) {
    case PlasmaOneNorm:
        #pragma omp task depend(in:A[0:lda*n]) \
                         depend(out:value[0:n])
        {
            if (sequence->status == PlasmaSuccess) {
                for (int j = 0; j < n; j++) {
                    value[j] = cabs(A[lda*j]);
                    for (int i = 1; i < m; i++) {
                        value[j] += cabs(A[lda*j+i]);
                    }
                }
            }
        }
        break;
    case PlasmaInfNorm:
        #pragma omp task depend(in:A[0:lda*n]) \
                         depend(out:value[0:m])
        {
            if (sequence->status == PlasmaSuccess) {
                for (int i = 0; i < m; i++)
                    value[i] = 0.0;

                for (int j = 0; j < n; j++) {
                    for (int i = 0; i < m; i++) {
                        value[i] += cabs(A[lda*j+i]);
                    }
                }
            }
        }
        break;
    }
}
