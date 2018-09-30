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

/******************************************************************************/
__attribute__((weak))
void plasma_core_zlansy(plasma_enum_t norm, plasma_enum_t uplo,
                 int n,
                 const plasma_complex64_t *A, int lda,
                 double *work, double *value)
{
    *value = LAPACKE_zlansy_work(LAPACK_COL_MAJOR,
                                 lapack_const(norm),
                                 lapack_const(uplo),
                                 n, A, lda, work);
}

/******************************************************************************/
void plasma_core_omp_zlansy(plasma_enum_t norm, plasma_enum_t uplo,
                     int n,
                     const plasma_complex64_t *A, int lda,
                     double *work, double *value,
                     plasma_sequence_t *sequence, plasma_request_t *request)
{
    #pragma omp task depend(in:A[0:lda*n]) \
                     depend(out:value[0:1])
    {
        if (sequence->status == PlasmaSuccess)
            plasma_core_zlansy(norm, uplo, n, A, lda, work, value);
    }
}

/******************************************************************************/
void plasma_core_omp_zlansy_aux(plasma_enum_t norm, plasma_enum_t uplo,
                         int n,
                         const plasma_complex64_t *A, int lda,
                         double *value,
                         plasma_sequence_t *sequence, plasma_request_t *request)
{
    switch (norm) {
    case PlasmaOneNorm:
    case PlasmaInfNorm:
        #pragma omp task depend(in:A[0:lda*n]) \
                         depend(out:value[0:n])
        {
            if (sequence->status == PlasmaSuccess) {
                if (uplo == PlasmaUpper) {
                    for (int i = 0; i < n; i++)
                        value[i] = 0.0;

                    for (int j = 0; j < n; j++) {
                        for (int i = 0; i < j; i++) {
                            value[i] += cabs(A[lda*j+i]);
                            value[j] += cabs(A[lda*j+i]);
                        }
                        value[j] += cabs(A[lda*j+j]);
                    }
                }
                else { // PlasmaLower
                    for (int i = 0; i < n; i++)
                        value[i] = 0.0;

                    for (int j = 0; j < n; j++) {
                        value[j] += cabs(A[lda*j+j]);
                        for (int i = j+1; i < n; i++) {
                            value[i] += cabs(A[lda*j+i]);
                            value[j] += cabs(A[lda*j+i]);
                        }
                    }
                }
            }
        }
        break;
    }
}
