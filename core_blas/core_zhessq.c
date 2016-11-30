/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> c
 *
 **/

#include "core_blas.h"
#include "plasma_types.h"
#include "core_lapack.h"

/******************************************************************************/
void core_zhessq(int uplo,
                 int n,
                 const plasma_complex64_t *A, int lda,
                 double *scale, double *sumsq)
{
    int ione = 1;
    if (uplo == PlasmaUpper) {
        for (int j = 1; j < n; j++)
            zlassq(&j, &A[lda*j], &ione, scale, sumsq);
    }
    else { // PlasmaLower
        for (int j = 0; j < n-1; j++) {
            int len = n-j-1;
            zlassq(&len, &A[lda*j+j+1], &ione, scale, sumsq);
        }
    }
    *sumsq *= 2.0;
    for (int i = 0; i < n; i++) {
        if ((double)A[lda*i+i] != 0.0) {
            double absa = fabs((double)A[lda*i+i]);
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
void core_omp_zhessq(int uplo,
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
            core_zhessq(uplo, n, A, lda, scale, sumsq);
        }
    }
}
