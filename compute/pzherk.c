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
#define C(m, n) ((PLASMA_Complex64_t*) plasma_getaddr(C, m, n))
/***************************************************************************//**
 * Parallel tile Hermitian rank k update.
 * @see PLASMA_zherk_Tile_Async
 ******************************************************************************/
void plasma_pzherk(PLASMA_enum uplo, PLASMA_enum trans,
                   double alpha, PLASMA_desc A,
                   double beta,  PLASMA_desc C,
                   PLASMA_sequence *sequence, PLASMA_request *request)
{
    int m, n, k;
    int ldak, ldam, ldan, ldcm, ldcn;
    int tempnn, tempmm, tempkn, tempkm;

    PLASMA_Complex64_t zbeta;
    PLASMA_Complex64_t zone = 1.0;

    // Check sequence status.
    if (sequence->status != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);
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
                zbeta = k == 0 ? beta : zone;
                CORE_OMP_zherk(
                    uplo, trans,
                    tempnn, tempkn,
                    alpha, A(n, k), ldan,
                    zbeta, C(n, n), ldcn);
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
                        CORE_OMP_zgemm(
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
                        CORE_OMP_zgemm(
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
                zbeta = k == 0 ? beta : zone;
                CORE_OMP_zherk(
                    uplo, trans,
                    tempnn, tempkm,
                    alpha, A(k, n), ldak,
                    zbeta, C(n, n), ldcn);
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
                        CORE_OMP_zgemm(
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
                        CORE_OMP_zgemm(
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
