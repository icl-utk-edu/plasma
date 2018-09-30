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
#include "core_lapack.h"

/******************************************************************************/
__attribute__((weak))
void plasma_core_zgessq(int m, int n,
                 const plasma_complex64_t *A, int lda,
                 double *scale, double *sumsq)
{
    int ione = 1;
    for (int j = 0; j < n; j++) {
        // TODO: Inline this operation.
        LAPACK_zlassq(&m, &A[j*lda], &ione, scale, sumsq);
    }
}

/******************************************************************************/
void plasma_core_omp_zgessq(int m, int n,
                     const plasma_complex64_t *A, int lda,
                     double *scale, double *sumsq,
                     plasma_sequence_t *sequence, plasma_request_t *request)
{
    #pragma omp task depend(in:A[0:lda*n]) \
                     depend(out:scale[0:n]) \
                     depend(out:sumsq[0:n])
    {
        if (sequence->status == PlasmaSuccess) {
            *scale = 0.0;
            *sumsq = 1.0;
            plasma_core_zgessq(m, n, A, lda, scale, sumsq);
        }
    }
}

/******************************************************************************/
void plasma_core_omp_zgessq_aux(int n,
                         const double *scale, const double *sumsq,
                         double *value,
                         plasma_sequence_t *sequence, plasma_request_t *request)
{
    #pragma omp task depend(in:scale[0:n]) \
                     depend(in:sumsq[0:n]) \
                     depend(out:value[0:1])
    {
        if (sequence->status == PlasmaSuccess) {
            double scl = 0.0;
            double sum = 1.0;
            for (int i = 0; i < n; i++) {
                if (scl < scale[i]) {
                    sum = sumsq[i] + sum*((scl/scale[i])*(scl/scale[i]));
                    scl = scale[i];
                }
                else {
                    sum = sum + sumsq[i]*(scale[i]/scl)*(scale[i]/scl);
                }
            }
            *value = scl*sqrt(sum);
        }
    }
}
