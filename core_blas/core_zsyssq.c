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
void plasma_core_zsyssq(plasma_enum_t uplo,
                 int n,
                 const plasma_complex64_t *A, int lda,
                 double *scale, double *sumsq)
{
    int ione = 1;
    if (uplo == PlasmaUpper) {
        for (int j = 1; j < n; j++)
            // TODO: Inline this operation.
            LAPACK_zlassq(&j, &A[lda*j], &ione, scale, sumsq);
    }
    else { // PlasmaLower
        for (int j = 0; j < n-1; j++) {
            int len = n-j-1;
            // TODO: Inline this operation.
            LAPACK_zlassq(&len, &A[lda*j+j+1], &ione, scale, sumsq);
        }
    }
    *sumsq *= 2.0;
    for (int i = 0; i < n; i++) {
        // diagonal is complex, don't ignore complex part
        double absa = cabs(A[lda*i+i]);
        if (absa != 0.0) { // != propagates nan
            if (*scale < absa) {
                *sumsq = 1.0 + *sumsq*((*scale/absa)*(*scale/absa));
                *scale = absa;
            }
            else {
                *sumsq = *sumsq + ((absa/(*scale))*(absa/(*scale)));
            }
        }
    }
}

/******************************************************************************/
void plasma_core_omp_zsyssq(plasma_enum_t uplo,
                     int n,
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
            plasma_core_zsyssq(uplo, n, A, lda, scale, sumsq);
        }
    }
}

/******************************************************************************/
void plasma_core_omp_zsyssq_aux(int m, int n,
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
            for (int j = 0; j < n; j++) {
                for (int i = j+1; i < n; i++) {
                    int idx = m*j+i;
                    if (scl < scale[idx]) {
                        sum = sumsq[idx] +
                            sum*((scl/scale[idx])*(scl/scale[idx]));
                        scl = scale[idx];
                    }
                    else {
                        sum = sum +
                            sumsq[idx]*((scale[idx]/scl)*(scale[idx]/scl));
                    }
                }
            }
            sum = 2.0*sum;
            for (int j = 0; j < n; j++) {
                int idx = m*j+j;
                if (scl < scale[idx]) {
                    sum = sumsq[idx] + sum*((scl/scale[idx])*(scl/scale[idx]));
                    scl = scale[idx];
                }
                else {
                    sum = sum + sumsq[idx]*((scale[idx]/scl)*(scale[idx]/scl));
                }
            }
            *value = scl*sqrt(sum);
        }
    }
}
