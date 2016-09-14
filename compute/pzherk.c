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

#define A(m, n) ((plasma_complex64_t*) plasma_getaddr(A, m, n))
#define C(m, n) ((plasma_complex64_t*) plasma_getaddr(C, m, n))
/***************************************************************************//**
 * Parallel tile Hermitian rank k update.
 * @see plasma_omp_zherk
 ******************************************************************************/
void plasma_pzherk(plasma_enum_t uplo, plasma_enum_t trans,
                   double alpha, plasma_desc_t A,
                   double beta,  plasma_desc_t C,
                   plasma_sequence_t *sequence, plasma_request_t *request)
{
    int m, n, k;
    int ldak, ldam, ldan, ldcm, ldcn;
    int tempnn, tempmm, tempkn, tempkm;

    plasma_complex64_t zbeta;
    plasma_complex64_t zone = 1.0;
    double dbeta;

    // Check sequence status.
    if (sequence->status != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorSequence);
        return;
    }

    for (n = 0; n < C.nt; n++) {
        tempnn = n == C.nt-1 ? C.n-n*C.nb : C.nb;
        ldan = BLKLDD(A, n);
        ldcn = BLKLDD(C, n);
        //=======================================
        // PlasmaNoTrans
        //=======================================
        if (trans == PlasmaNoTrans) {
            for (k = 0; k < A.nt; k++) {
                tempkn = k == A.nt-1 ? A.n-k*A.nb : A.nb;
                dbeta = k == 0 ? beta : 1.0;
                core_omp_zherk(
                    uplo, trans,
                    tempnn, tempkn,
                    alpha, A(n, k), ldan,
                    dbeta, C(n, n), ldcn);
            }
            //=======================================
            // PlasmaNoTrans / PlasmaLower
            //=======================================
            if (uplo == PlasmaLower) {
                for (m = n+1; m < C.mt; m++) {
                    tempmm = m == C.mt-1 ? C.m-m*C.mb : C.mb;
                    ldam = BLKLDD(A, m);
                    ldcm = BLKLDD(C, m);
                    for (k = 0; k < A.nt; k++) {
                        tempkn = k == A.nt-1 ? A.n-k*A.nb : A.nb;
                        zbeta = k == 0 ? beta : zone;
                        core_omp_zgemm(
                            trans, PlasmaConjTrans,
                            tempmm, tempnn, tempkn,
                            alpha, A(m, k), ldam,
                                   A(n, k), ldan,
                            zbeta, C(m, n), ldcm);
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
                    for (k = 0; k < A.nt; k++) {
                        tempkn = k == A.nt-1 ? A.n-k*A.nb : A.nb;
                        zbeta = k == 0 ? beta : zone;
                        core_omp_zgemm(
                            trans, PlasmaConjTrans,
                            tempnn, tempmm, tempkn,
                            alpha, A(n, k), ldan,
                                   A(m, k), ldam,
                            zbeta, C(n, m), ldcn);
                    }
                }
            }
        }
        //=======================================
        // PlasmaConjTrans
        //=======================================
        else {
            for (k = 0; k < A.mt; k++) {
                tempkm = k == A.mt-1 ? A.m-k*A.mb : A.mb;
                ldak = BLKLDD(A, k);
                dbeta = k == 0 ? beta : 1.0;
                core_omp_zherk(
                    uplo, trans,
                    tempnn, tempkm,
                    alpha, A(k, n), ldak,
                    dbeta, C(n, n), ldcn);
            }
            //=======================================
            // PlasmaConjTrans / PlasmaLower
            //=======================================
            if (uplo == PlasmaLower) {
                for (m = n+1; m < C.mt; m++) {
                    tempmm = m == C.mt-1 ? C.m-m*C.mb : C.mb;
                    ldcm = BLKLDD(C, m);
                    for (k = 0; k < A.mt; k++) {
                        tempkm = k == A.mt-1 ? A.m-k*A.mb : A.mb;
                        ldak = BLKLDD(A, k);
                        zbeta = k == 0 ? beta : zone;
                        core_omp_zgemm(
                            trans, PlasmaNoTrans,
                            tempmm, tempnn, tempkm,
                            alpha, A(k, m), ldak,
                                   A(k, n), ldak,
                            zbeta, C(m, n), ldcm);
                    }
                }
            }
            //=======================================
            // PlasmaConjTrans / PlasmaUpper
            //=======================================
            else {
                for (m = n+1; m < C.mt; m++) {
                    tempmm = m == C.mt-1 ? C.m-m*C.mb : C.mb;
                    for (k = 0; k < A.mt; k++) {
                        tempkm = k == A.mt-1 ? A.m-k*A.mb : A.mb;
                        ldak = BLKLDD(A, k);
                        zbeta = k == 0 ? beta : zone;
                        core_omp_zgemm(
                            trans, PlasmaNoTrans,
                            tempnn, tempmm, tempkm,
                            alpha, A(k, n), ldak,
                                   A(k, m), ldak,
                            zbeta, C(n, m), ldcn);
                    }
                }
            }
        }
    }
}
