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

#include <math.h>

/******************************************************************************/
__attribute__((weak))
void plasma_core_zlantr(plasma_enum_t norm, plasma_enum_t uplo, plasma_enum_t diag,
                 int m, int n,
                 const plasma_complex64_t *A, int lda,
                 double *work, double *value)
{
    // Due to a bug in LAPACKE < 3.6.1, this function always returns zero.
    // *value = LAPACKE_zlantr_work(LAPACK_COL_MAJOR,
    //                              lapack_const(norm), lapack_const(uplo),
    //                              lapack_const(diag),
    //                              m, n, A, lda, work);

    // Calling LAPACK directly instead.
    char nrm = lapack_const(norm);
    char upl = lapack_const(uplo);
    char dia = lapack_const(diag);
    *value = LAPACK_zlantr(&nrm, &upl, &dia, &m, &n, A, &lda, work);
}

/******************************************************************************/
void plasma_core_omp_zlantr(plasma_enum_t norm, plasma_enum_t uplo, plasma_enum_t diag,
                     int m, int n,
                     const plasma_complex64_t *A, int lda,
                     double *work, double *value,
                     plasma_sequence_t *sequence, plasma_request_t *request)
{
    #pragma omp task depend(in:A[0:lda*n]) \
                     depend(out:value[0:1])
    {
        if (sequence->status == PlasmaSuccess)
            plasma_core_zlantr(norm, uplo, diag, m, n, A, lda, work, value);
    }
}

/******************************************************************************/
void plasma_core_omp_zlantr_aux(plasma_enum_t norm, plasma_enum_t uplo,
                         plasma_enum_t diag,
                         int m, int n,
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
                if (uplo == PlasmaUpper) {
                    if (diag == PlasmaNonUnit) {
                        for (int j = 0; j < n; j++) {
                            value[j] = cabs(A[lda*j]);
                            for (int i = 1; i < imin(j+1, m); i++) {
                                value[j] += cabs(A[lda*j+i]);
                            }
                        }
                    }
                    else { // PlasmaUnit
                        int j;
                        for (j = 0; j < imin(n, m); j++) {
                            value[j] = 1.0;
                            for (int i = 0; i < j; i++) {
                                value[j] += cabs(A[lda*j+i]);
                            }
                        }
                        for (; j < n; j++) {
                            value[j] = cabs(A[lda*j]);
                            for (int i = 1; i < m; i++) {
                                value[j] += cabs(A[lda*j+i]);
                            }
                        }
                    }
                }
                else { // PlasmaLower
                    if (diag == PlasmaNonUnit) {
                        int j;
                        for (j = 0; j < imin(n, m); j++) {
                            value[j] = cabs(A[lda*j+j]);
                            for (int i = j+1; i < m; i++) {
                                value[j] += cabs(A[lda*j+i]);
                            }
                        }
                        for (; j < n; j++)
                            value[j] = 0.0;
                    }
                    else { // PlasmaUnit
                        int j;
                        for (j = 0; j < imin(n, m); j++) {
                            value[j] = 1.0;
                            for (int i = j+1; i < m; i++) {
                                value[j] += cabs(A[lda*j+i]);
                            }
                        }
                        for (; j < n; j++)
                            value[j] = 0.0;
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
                if (uplo == PlasmaUpper) {
                    if (diag == PlasmaNonUnit) {
                        for (int i = 0; i < m; i++)
                            value[i] = 0.0;

                        for (int j = 0; j < n; j++) {
                            for (int i = 0; i < imin(j+1, m); i++) {
                                value[i] += cabs(A[lda*j+i]);
                            }
                        }
                    }
                    else { // PlasmaUnit
                        int i;
                        for (i = 0; i < imin(m, n); i++)
                            value[i] = 1.0;

                        for (; i < m; i++)
                            value[i] = 0.0;

                        int j;
                        for (j = 0; j < imin(n, m); j++) {
                            for (i = 0; i < j; i++) {
                                value[i] += cabs(A[lda*j+i]);
                            }
                        }
                        for (; j < n; j++) {
                            for (i = 0; i < m; i++) {
                                value[i] += cabs(A[lda*j+i]);
                            }
                        }
                    }
                }
                else { // PlasmaLower
                    if (diag == PlasmaNonUnit) {
                        for (int i = 0; i < m; i++)
                            value[i] = 0.0;

                        for (int j = 0; j < imin(n, m); j++) {
                            for (int i = j; i < m; i++) {
                                value[i] += cabs(A[lda*j+i]);
                            }
                        }
                    }
                    else { // PlasmaUnit
                        int i;
                        for (i = 0; i < imin(m, n); i++)
                            value[i] = 1.0;

                        for (; i < m; i++)
                            value[i] = 0.0;

                        for (int j = 0; j < imin(n, m); j++) {
                            for (i = j+1; i < m; i++) {
                                value[i] += cabs(A[lda*j+i]);
                            }
                        }
                    }
                }
            }
        }
        break;
    }
}
