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

#include "plasma_async.h"
#include "plasma_descriptor.h"
#include "plasma_types.h"
#include "plasma_internal.h"
#include "core_blas_z.h"

#define A(m, n) ((PLASMA_Complex64_t*) plasma_getaddr(A, m, n))
#define B(m, n) ((PLASMA_Complex64_t*) plasma_getaddr(B, m, n))
#define C(m, n) ((PLASMA_Complex64_t*) plasma_getaddr(C, m, n))
/***************************************************************************//**
 * Parallel tile Hermitian rank 2k update.
 * @see plasma_omp_zher2k
 ******************************************************************************/
void plasma_pzher2k(PLASMA_enum uplo, PLASMA_enum trans,
                    PLASMA_Complex64_t alpha, PLASMA_desc A,
                    PLASMA_desc B, double beta,  PLASMA_desc C,
                    plasma_sequence_t *sequence, PLASMA_request *request)
{
    int m, n, k;
    int ldak, ldam, ldan, ldcm, ldcn;
    int ldbk, ldbm, ldbn;
    int tempnn, tempmm, tempkn, tempkm;

    PLASMA_Complex64_t zone   = (PLASMA_Complex64_t)1.0;
    PLASMA_Complex64_t zbeta;
    double dbeta;

    // Check sequence status.
    if (sequence->status != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);
        return;
    }

    for (n = 0; n < C.nt; n++) {
        tempnn = n == C.nt-1 ? C.n-n*C.nb : C.nb;
        ldan = BLKLDD(A, n);
        ldbn = BLKLDD(B, n);
        ldcn = BLKLDD(C, n);
        //=======================================
        // PlasmaNoTrans
        //=======================================
        if (trans == PlasmaNoTrans) {
            for (k = 0; k < A.nt; k++) {
                tempkn = k == A.nt-1 ? A.n-k*A.nb : A.nb;
                dbeta = k == 0 ? beta : 1.0;
                core_omp_zher2k(
                    uplo, trans,
                    tempnn, tempkn,
                    alpha, A(n, k), ldan,
                           B(n, k), ldbn,
                    dbeta, C(n, n), ldcn);
            }
            //=======================================
            // PlasmaNoTrans / PlasmaLower
            //=======================================
            if (uplo == PlasmaLower) {
                for (m = n+1; m < C.mt; m++) {
                    tempmm = m == C.mt-1 ? C.m-m*C.mb : C.mb;
                    ldam = BLKLDD(A, m);
                    ldbm = BLKLDD(B, m);
                    ldcm = BLKLDD(C, m);
                    for (k = 0; k < A.nt; k++) {
                        tempkn = k == A.nt-1 ? A.n-k*A.nb : A.nb;
                        zbeta = k == 0 ? (PLASMA_Complex64_t)beta : zone;
                        core_omp_zgemm(
                            trans, PlasmaConjTrans,
                            tempmm, tempnn, tempkn,
                            alpha, A(m, k), ldam,
                                   B(n, k), ldbn,
                            zbeta, C(m, n), ldcm);

                        core_omp_zgemm(
                            trans, PlasmaConjTrans,
                            tempmm, tempnn, tempkn,
                            conj(alpha), B(m, k), ldam,
                                         A(n, k), ldan,
                                  zone,  C(m, n), ldcm);
                    }
                }
            }
            //=======================================
            // PlasmaNoTrans / PlasmaUpper
            //=======================================
            else {
                for (m = n+1; m < C.mt; m++) {
                    tempmm = m == C.mt-1 ? C.m-m*C.mb : C.mb;
                    ldam = BLKLDD(A, m);
                    ldbm = BLKLDD(B, m);
                    for (k = 0; k < A.nt; k++) {
                        tempkn = k == A.nt-1 ? A.n-k*A.nb : A.nb;
                        zbeta = k == 0 ? (PLASMA_Complex64_t)beta : zone;
                        core_omp_zgemm(
                            trans, PlasmaConjTrans,
                            tempnn, tempmm, tempkn,
                            alpha, A(n, k), ldan,
                                   B(m, k), ldbm,
                            zbeta, C(n, m), ldcn);

                        core_omp_zgemm(
                            trans, PlasmaConjTrans,
                            tempnn, tempmm, tempkn,
                            conj(alpha), B(n, k), ldan,
                                         A(m, k), ldam,
                                   zone, C(n, m), ldcn);
                    }
                }
            }
        }
        //=======================================
        // Plasma[Conj]Trans
        //=======================================
        else {
            for (k = 0; k < A.mt; k++) {
                tempkm = k == A.mt-1 ? A.m-k*A.mb : A.mb;
                ldak = BLKLDD(A, k);
                ldbk = BLKLDD(B, k);
                dbeta = k == 0 ? beta : 1.0;
                core_omp_zher2k(
                    uplo, trans,
                    tempnn, tempkm,
                    alpha, A(k, n), ldak,
                           B(k, n), ldbk,
                    dbeta, C(n, n), ldcn);
            }
            //=======================================
            // Plasma[Conj]Trans / PlasmaLower
            //=======================================
            if (uplo == PlasmaLower) {
                for (m = n+1; m < C.mt; m++) {
                    tempmm = m == C.mt-1 ? C.m-m*C.mb : C.mb;
                    ldcm = BLKLDD(C, m);
                    for (k = 0; k < A.mt; k++) {
                        tempkm = k == A.mt-1 ? A.m-k*A.mb : A.mb;
                        ldak = BLKLDD(A, k);
                        ldbk = BLKLDD(B, k);
                        zbeta = k == 0 ? (PLASMA_Complex64_t)beta : zone;
                        core_omp_zgemm(
                            trans, PlasmaNoTrans,
                            tempmm, tempnn, tempkm,
                            alpha, A(k, m), ldak,
                                   B(k, n), ldbk,
                            zbeta, C(m, n), ldcm);

                        core_omp_zgemm(
                            trans, PlasmaNoTrans,
                            tempmm, tempnn, tempkm,
                            conj(alpha), B(k, m),
                            ldbk, A(k, n), ldak,
                            zone, C(m, n), ldcm);
                    }
                }
            }
            //=======================================
            // Plasma[Conj]Trans / PlasmaUpper
            //=======================================
            else {
                for (m = n+1; m < C.mt; m++) {
                    tempmm = m == C.mt-1 ? C.m-m*C.mb : C.mb;
                    for (k = 0; k < A.mt; k++) {
                        tempkm = k == A.mt-1 ? A.m-k*A.mb : A.mb;
                        ldak = BLKLDD(A, k);
                        ldbk = BLKLDD(B, k);
                        zbeta = k == 0 ? (PLASMA_Complex64_t)beta : zone;

                        core_omp_zgemm(
                            trans, PlasmaNoTrans,
                            tempnn, tempmm, tempkm,
                            alpha, A(k, n), ldak,
                                   B(k, m), ldbk,
                            zbeta, C(n, m), ldcn);

                        core_omp_zgemm(
                            trans, PlasmaNoTrans,
                            tempnn, tempmm, tempkm,
                            conj(alpha), B(k, n), ldbk,
                                         A(k, m), ldak,
                                  zone,  C(n, m), ldcn);
                    }
                }
            }
        }
    }
}
