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

#define A(m,n) ((PLASMA_Complex64_t*) plasma_getaddr(A, m, n))
#define B(m,n) ((PLASMA_Complex64_t*) plasma_getaddr(B, m, n))

/***************************************************************************//**
 * Parallel tile matrix-matrix addition.
 * @see plasma_omp_ztradd
 ******************************************************************************/
void plasma_pztradd(PLASMA_enum uplo, PLASMA_enum transA,
                    PLASMA_Complex64_t alpha,  PLASMA_desc A,
                    PLASMA_Complex64_t beta,   PLASMA_desc B,
                    plasma_sequence_t *sequence, PLASMA_request *request)
{
    int tempmm, tempnn, tempmn, tempnm;
    int m, n;
    int ldam, ldan, ldbm, ldbn;

    // Check sequence status
    if (sequence->status != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);
        return;
    }

    switch (uplo) {
    case PlasmaLower:
        if (transA == PlasmaNoTrans) {
            for (n = 0; n < imin(B.mt,B.nt); n++) {
                tempnm = n == B.mt-1 ? B.m-n*B.mb : B.mb;
                tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                ldan = BLKLDD(A, n);
                ldbn = BLKLDD(B, n);

                core_omp_ztradd(
                    uplo, transA, tempnm, tempnn,
                    alpha, A(n, n), ldan,
                    beta,  B(n, n), ldbn);

                for (m = n+1; m < B.mt; m++) {
                    tempmm = m == B.mt-1 ? B.m-B.mb*m : B.nb;
                    ldam = BLKLDD(A, m);
                    ldbm = BLKLDD(B, m);

                    core_omp_zgeadd(
                        transA, tempmm, tempnn,
                        alpha, A(m, n), ldam,
                        beta,  B(m, n), ldbm);
                }
            }
        }
        else {
            for (n = 0; n < imin(B.mt,B.nt); n++) {
                tempnm = n == B.mt-1 ? B.m-n*B.mb : B.mb;
                tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                ldan = BLKLDD(A, n);
                ldbn = BLKLDD(B, n);

                core_omp_ztradd(
                    uplo, transA, tempnm, tempnn,
                    alpha, A(n, n), ldan,
                    beta,  B(n, n), ldbn);

                for (m = n+1; m < B.mt; m++) {
                    tempmm = m == B.mt-1 ? B.m-B.mb*m : B.nb;
                    ldbm = BLKLDD(B, m);

                    core_omp_zgeadd(
                        transA, tempmm, tempnn,
                        alpha, A(n, m), ldan,
                        beta,  B(m, n), ldbm);
                }
            }
        }
        break;
    case PlasmaUpper:
        if (transA == PlasmaNoTrans) {
            for (m = 0; m < imin(B.mt,B.nt); m++) {
                tempmm = m == B.mt-1 ? B.m-B.mb*m : B.nb;
                tempmn = m == B.nt-1 ? B.n-m*B.nb : B.nb;
                ldam = BLKLDD(A, m);
                ldbm = BLKLDD(B, m);

                core_omp_ztradd(
                    uplo, transA, tempmm, tempmn,
                    alpha, A(m, m), ldam,
                    beta,  B(m, m), ldbm);

                for (n = m+1; n < B.nt; n++) {
                    tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;

                    core_omp_zgeadd(
                        transA, tempmm, tempnn,
                        alpha, A(m, n), ldam,
                        beta,  B(m, n), ldbm);
                }
            }
        }
        else {
            for (m = 0; m < imin(B.mt,B.nt); m++) {
                tempmm = m == B.mt-1 ? B.m-B.mb*m : B.nb;
                tempmn = m == B.nt-1 ? B.n-m*B.nb : B.nb;
                ldam = BLKLDD(A, m);
                ldbm = BLKLDD(B, m);

                core_omp_ztradd(
                    uplo, transA, tempmm, tempmn,
                    alpha, A(m, m), ldam,
                    beta,  B(m, m), ldbm);

                for (n = m+1; n < B.nt; n++) {
                    tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                    ldan = BLKLDD(A, n);

                    core_omp_zgeadd(
                        transA, tempmm, tempnn,
                        alpha, A(n, m), ldan,
                        beta,  B(m, n), ldbm);
                }
            }
        }
        break;
    case PlasmaFull:
    default:
        if (transA == PlasmaNoTrans) {
            for (m = 0; m < B.mt; m++) {
                tempmm = m == B.mt-1 ? B.m-B.mb*m : B.nb;
                ldam = BLKLDD(A, m);
                ldbm = BLKLDD(B, m);

                for (n = 0; n < B.nt; n++) {
                    tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;

                    core_omp_zgeadd(
                        transA, tempmm, tempnn,
                        alpha, A(m, n), ldam,
                        beta,  B(m, n), ldbm);
                }
            }
        }
        else {
            for (m = 0; m < B.mt; m++) {
                tempmm = m == B.mt-1 ? B.m-B.mb*m : B.nb;
                ldam = BLKLDD(A, m);
                ldbm = BLKLDD(B, m);

                for (n = 0; n < B.nt; n++) {
                    tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                    ldan = BLKLDD(A, n);

                    core_omp_zgeadd(
                        transA, tempmm, tempnn,
                        alpha, A(n, m), ldan,
                        beta,  B(m, n), ldbm);
                }
            }
        }
    }
}
