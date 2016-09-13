/**
 *
 * @File pzsymm.c
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
#define B(m, n) ((PLASMA_Complex64_t*) plasma_getaddr(B, m, n))
#define C(m, n) ((PLASMA_Complex64_t*) plasma_getaddr(C, m, n))

/***************************************************************************//**
 *  Parallel tile symmetric matrix-matrix multiplication.
 *  @see plasma_omp_zsymm
 ******************************************************************************/
void plasma_pzsymm(PLASMA_enum side, PLASMA_enum uplo,
                   PLASMA_Complex64_t alpha, plasma_desc_t A,
                                             plasma_desc_t B,
                   PLASMA_Complex64_t beta,  plasma_desc_t C,
                   plasma_sequence_t *sequence, plasma_request_t *request)
{
    int k, m, n;
    int ldak, ldam, ldan, ldbk, ldbm, ldcm;
    int tempmm, tempnn, tempkn, tempkm;

    PLASMA_Complex64_t zbeta;
    PLASMA_Complex64_t zone = 1.0;

    // Check sequence status.
    if (sequence->status != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);
        return;
    }

    for (m = 0; m < C.mt; m++) {
        tempmm = m == C.mt-1 ? C.m-m*C.mb : C.mb;
        ldcm = BLKLDD(C, m);
        for (n = 0; n < C.nt; n++) {
            tempnn = n == C.nt-1 ? C.n-n*C.nb : C.nb;
            if (side == PlasmaLeft) {
                ldam = BLKLDD(A, m);
                //=======================================
                // SIDE: PlasmaLeft / UPLO: PlasmaLower
                //=======================================
                if (uplo == PlasmaLower) {
                    for (k = 0; k < C.mt; k++) {
                        tempkm = k == C.mt-1 ? C.m-k*C.mb : C.mb;
                        ldak = BLKLDD(A, k);
                        ldbk = BLKLDD(B, k);
                        zbeta = k == 0 ? beta : zone;
                        if (k < m) {
                            core_omp_zgemm(
                                PlasmaNoTrans, PlasmaNoTrans,
                                tempmm, tempnn, tempkm,
                                alpha, A(m, k), ldam,
                                       B(k, n), ldbk,
                                zbeta, C(m, n), ldcm);
                        }
                        else {
                            if (k == m) {
                                core_omp_zsymm(
                                    side, uplo,
                                    tempmm, tempnn,
                                    alpha, A(k, k), ldak,
                                           B(k, n), ldbk,
                                    zbeta, C(m, n), ldcm);
                            }
                            else {
                                core_omp_zgemm(
                                    PlasmaTrans, PlasmaNoTrans,
                                    tempmm, tempnn, tempkm,
                                    alpha, A(k, m), ldak,
                                           B(k, n), ldbk,
                                    zbeta, C(m, n), ldcm);
                            }
                        }
                    }
                }
                //=======================================
                // SIDE: PlasmaLeft / UPLO: PlasmaUpper
                //=======================================
                else {
                    for (k = 0; k < C.mt; k++) {
                        tempkm = k == C.mt-1 ? C.m-k*C.mb : C.mb;
                        ldak = BLKLDD(A, k);
                        ldbk = BLKLDD(B, k);
                        zbeta = k == 0 ? beta : zone;
                        if (k < m) {
                            core_omp_zgemm(
                                PlasmaTrans, PlasmaNoTrans,
                                tempmm, tempnn, tempkm,
                                alpha, A(k, m), ldak,
                                       B(k, n), ldbk,
                                zbeta, C(m, n), ldcm);
                        }
                        else {
                            if (k == m) {
                                core_omp_zsymm(
                                    side, uplo,
                                    tempmm, tempnn,
                                    alpha, A(k, k), ldak,
                                           B(k, n), ldbk,
                                    zbeta, C(m, n), ldcm);
                            }
                            else {
                                core_omp_zgemm(
                                    PlasmaNoTrans, PlasmaNoTrans,
                                    tempmm, tempnn, tempkm,
                                    alpha, A(m, k), ldam,
                                           B(k, n), ldbk,
                                    zbeta, C(m, n), ldcm);
                            }
                        }
                    }
                }
            }
            else {
                ldan = BLKLDD(A, n);
                ldbm = BLKLDD(B, m);
                //=======================================
                // SIDE: PlasmaRight / UPLO: PlasmaLower
                //=======================================
                if (uplo == PlasmaLower) {
                    for (k = 0; k < C.nt; k++) {
                        tempkn = k == C.nt-1 ? C.n-k*C.nb : C.nb;
                        ldak = BLKLDD(A, k);
                        zbeta = k == 0 ? beta : zone;
                        if (k < n) {
                            core_omp_zgemm(
                                PlasmaNoTrans, PlasmaTrans,
                                tempmm, tempnn, tempkn,
                                alpha, B(m, k), ldbm,
                                       A(n, k), ldan,
                                zbeta, C(m, n), ldcm);
                        }
                        else {
                            if (n == k) {
                                core_omp_zsymm(
                                    side, uplo,
                                    tempmm, tempnn,
                                    alpha, A(k, k), ldak,
                                           B(m, k), ldbm,
                                    zbeta, C(m, n), ldcm);
                            }
                            else {
                                core_omp_zgemm(
                                    PlasmaNoTrans, PlasmaNoTrans,
                                    tempmm, tempnn, tempkn,
                                    alpha, B(m, k), ldbm,
                                           A(k, n), ldak,
                                    zbeta, C(m, n), ldcm);
                            }
                        }
                    }
                }
                //=======================================
                // SIDE: PlasmaRight / UPLO: PlasmaUpper
                //=======================================
                else {
                    for (k = 0; k < C.nt; k++) {
                        tempkn = k == C.nt-1 ? C.n-k*C.nb : C.nb;
                        ldak = BLKLDD(A, k);
                        zbeta = k == 0 ? beta : zone;
                        if (k < n) {
                            core_omp_zgemm(
                                PlasmaNoTrans, PlasmaNoTrans,
                                tempmm, tempnn, tempkn,
                                alpha, B(m, k), ldbm,
                                       A(k, n), ldak,
                                zbeta, C(m, n), ldcm);
                        }
                        else {
                            if (n == k) {
                                core_omp_zsymm(
                                    side, uplo,
                                    tempmm, tempnn,
                                    alpha, A(k, k), ldak,
                                           B(m, k), ldbm,
                                    zbeta, C(m, n), ldcm);
                            }
                            else {
                                core_omp_zgemm(
                                    PlasmaNoTrans, PlasmaTrans,
                                    tempmm, tempnn, tempkn,
                                    alpha, B(m, k), ldbm,
                                           A(n, k), ldan,
                                    zbeta, C(m, n), ldcm);
                            }
                        }
                    }
                }
            }
        }
    }
}
