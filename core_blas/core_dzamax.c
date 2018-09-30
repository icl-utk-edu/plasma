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
void plasma_core_omp_dzamax(int colrow, int m, int n,
                     const plasma_complex64_t *A, int lda,
                     double *values,
                     plasma_sequence_t *sequence, plasma_request_t *request)
{
    switch (colrow) {
    case PlasmaColumnwise:
        #pragma omp task depend(in:A[0:lda*n]) \
                         depend(out:values[0:n])
        {
            if (sequence->status == PlasmaSuccess) {
                for (int j = 0; j < n; j++) {
                    values[j] = plasma_core_dcabs1(A[lda*j]);
                    for (int i = 1; i < m; i++) {
                        double tmp = plasma_core_dcabs1(A[lda*j+i]);
                        if (tmp > values[j])
                            values[j] = tmp;
                    }
                }
            }
        }
        break;
    case PlasmaRowwise:
        #pragma omp task depend(in:A[0:lda*n]) \
                         depend(out:values[0:m])
        {
            if (sequence->status == PlasmaSuccess) {
                for (int i = 0; i < m; i++)
                    values[i] = plasma_core_dcabs1(A[i]);

                for (int j = 1; j < n; j++) {
                    for (int i = 0; i < m; i++) {
                        double tmp = plasma_core_dcabs1(A[lda*j+i]);
                        if (tmp > values[i])
                            values[i] = tmp;
                    }
                }
            }
        }
        break;
    }
}
