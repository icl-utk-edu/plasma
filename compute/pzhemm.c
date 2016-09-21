/**
 *
 * @File pzhemm.c
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> c
 *
 **/

#include "plasma_async.h"
#include "plasma_descriptor.h"
#include "plasma_types.h"
#include "plasma_internal.h"
#include "core_blas_z.h"

#define A(m, n) ((plasma_complex64_t*) plasma_tile_addr(A, m, n))
#define B(m, n) ((plasma_complex64_t*) plasma_tile_addr(B, m, n))
#define C(m, n) ((plasma_complex64_t*) plasma_tile_addr(C, m, n))

/***************************************************************************//**
 *  Parallel tile Hermitian matrix-matrix multiplication.
 *  @see plasma_omp_zhemm
 ******************************************************************************/
void plasma_pzhemm(plasma_enum_t side, plasma_enum_t uplo,
                   plasma_complex64_t alpha, plasma_desc_t A,
                                             plasma_desc_t B,
                   plasma_complex64_t beta,  plasma_desc_t C,
                   plasma_sequence_t *sequence, plasma_request_t *request)
{
    int k, m, n;
    int ldak, ldam, ldan, ldbk, ldbm, ldcm;
    int tempmm, tempnn, tempkn, tempkm;

    plasma_complex64_t zbeta;
    plasma_complex64_t zone = 1.0;

    // Check sequence status.
    if (sequence->status != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorSequence);
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
                                core_omp_zhemm(
                                    side, uplo,
                                    tempmm, tempnn,
                                    alpha, A(k, k), ldak,
                                           B(k, n), ldbk,
                                    zbeta, C(m, n), ldcm);
                            }
                            else {
                                core_omp_zgemm(
                                    PlasmaConjTrans, PlasmaNoTrans,
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
                                PlasmaConjTrans, PlasmaNoTrans,
                                tempmm, tempnn, tempkm,
                                alpha, A(k, m), ldak,
                                       B(k, n), ldbk,
                                zbeta, C(m, n), ldcm);
                        }
                        else {
                            if (k == m) {
                                core_omp_zhemm(
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
                                PlasmaNoTrans, PlasmaConjTrans,
                                tempmm, tempnn, tempkn,
                                alpha, B(m, k), ldbm,
                                       A(n, k), ldan,
                                zbeta, C(m, n), ldcm);
                        }
                        else {
                            if (n == k) {
                                core_omp_zhemm(
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
                                core_omp_zhemm(
                                    side, uplo,
                                    tempmm, tempnn,
                                    alpha, A(k, k), ldak,
                                           B(m, k), ldbm,
                                    zbeta, C(m, n), ldcm);
                            }
                            else {
                                core_omp_zgemm(
                                    PlasmaNoTrans, PlasmaConjTrans,
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
