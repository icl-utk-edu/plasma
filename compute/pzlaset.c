/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> s d c
 *
 **/

#include "plasma_async.h"
#include "plasma_descriptor.h"
#include "plasma_types.h"
#include "plasma_internal.h"
#include "core_blas_z.h"

#define A(m, n) ((PLASMA_Complex64_t*) plasma_getaddr(A, m, n))
/***************************************************************************//**
 *  Parallel initialization a 2-D array A to BETA on the diagonal and
 *  ALPHA on the offdiagonals.
 **/
void plasma_pzlaset(PLASMA_enum uplo,
                    PLASMA_Complex64_t alpha, PLASMA_Complex64_t beta,
                    PLASMA_desc A,
                    PLASMA_sequence *sequence, PLASMA_request *request)
{
    int i, j;
    int ldai, ldaj;
    int tempim, tempjm, tempjn;
    int minmn = imin(A.mt, A.nt);

    // Check sequence status.
    if (sequence->status != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);
        return;
    }

    if (uplo == PlasmaLower) {
        for (j = 0; j < minmn; j++) {
            tempjm = j == A.mt-1 ? A.m-j*A.mb : A.mb;
            tempjn = j == A.nt-1 ? A.n-j*A.nb : A.nb;
            ldaj = BLKLDD(A, j);
            CORE_OMP_zlaset(PlasmaLower, tempjm, tempjn, alpha, beta,
                            A(j, j), ldaj);

            for (i = j+1; i < A.mt; i++) {
                tempim = i == A.mt-1 ? A.m-i*A.mb : A.mb;
                ldai = BLKLDD(A, i);
                CORE_OMP_zlaset(PlasmaFull,
                                tempim, tempjn,
                                alpha, alpha, A(i, j), ldai);
            }
        }
    }
    else if (uplo == PlasmaUpper) {
        for (j = 1; j < A.nt; j++) {
            tempjn = j == A.nt-1 ? A.n-j*A.nb : A.nb;
            for (i = 0; i < imin(j, A.mt); i++) {
                tempim = i == A.mt-1 ? A.m-i*A.mb : A.mb;
                ldai = BLKLDD(A, i);
                CORE_OMP_zlaset(PlasmaFull,
                                tempim, tempjn,
                                alpha, alpha, A(i, j), ldai);
            }
        }
        for (j = 0; j < minmn; j++) {
            tempjm = j == A.mt-1 ? A.m-j*A.mb : A.mb;
            tempjn = j == A.nt-1 ? A.n-j*A.nb : A.nb;
            ldaj = BLKLDD(A, j);
            CORE_OMP_zlaset(PlasmaUpper,
                            tempjm, tempjn,
                            alpha, beta, A(j, j), ldaj);
        }
    }
    else { // PlasmaFull, i.e. diagonal matrix
        for (i = 0; i < A.mt; i++) {
            tempim = i == A.mt-1 ? A.m-i*A.mb : A.mb;
            ldai = BLKLDD(A, i);
            for (j = 0; j < A.nt; j++) {
                tempjn = j == A.nt-1 ? A.n-j*A.nb : A.nb;
                CORE_OMP_zlaset(PlasmaFull,
                                tempim, tempjn,
                                alpha, alpha, A(i, j), ldai);
            }
        }
        for (j = 0; j < minmn; j++) {
            tempjm = j == A.mt-1 ? A.m-j*A.mb : A.mb;
            tempjn = j == A.nt-1 ? A.n-j*A.nb : A.nb;
            ldaj = BLKLDD(A, j);
            CORE_OMP_zlaset(PlasmaFull,
                            tempjm, tempjn,
                            alpha, beta, A(j, j), ldaj);
        }
    }
}
