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
#define B(m, n) ((PLASMA_Complex64_t*) plasma_getaddr(B, m, n))
#define C(m, n) ((PLASMA_Complex64_t*) plasma_getaddr(C, m, n))
/***************************************************************************//**
 * Parallel tile matrix-matrix multiplication.
 * @see plasma_omp_zgemm
 ******************************************************************************/
void plasma_pzgemm(PLASMA_enum transA, PLASMA_enum transB,
                   PLASMA_Complex64_t alpha, plasma_desc_t A,
                                             plasma_desc_t B,
                   PLASMA_Complex64_t beta,  plasma_desc_t C,
                   plasma_sequence_t *sequence, plasma_request_t *request)
{
    int m, n, k;
    int ldam, ldak, ldbn, ldbk, ldcm;
    int tempmm, tempnn, tempkn, tempkm;

    PLASMA_Complex64_t zbeta;
    PLASMA_Complex64_t zone = 1.0;

    int innerK = (transA == PlasmaNoTrans ? A.n : A.m);

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
            if (alpha == 0.0 || innerK == 0) {
                //=======================================
                // alpha*A*B does not contribute; scale C
                //=======================================
                ldam = imax( 1, BLKLDD(A, 0) );
                ldbk = imax( 1, BLKLDD(B, 0) );
                core_omp_zgemm(
                    transA, transB,
                    tempmm, tempnn, 0,
                    alpha, A(0, 0), ldam,
                           B(0, 0), ldbk,
                    beta,  C(m, n), ldcm);
            }
            else if (transA == PlasmaNoTrans) {
                ldam = BLKLDD(A, m);
                //=======================================
                // A: PlasmaNoTrans / B: PlasmaNoTrans
                //=======================================
                if (transB == PlasmaNoTrans) {
                    for (k = 0; k < A.nt; k++) {
                        tempkn = k == A.nt-1 ? A.n-k*A.nb : A.nb;
                        ldbk = BLKLDD(B, k);
                        zbeta = k == 0 ? beta : zone;
                        core_omp_zgemm(
                            transA, transB,
                            tempmm, tempnn, tempkn,
                            alpha, A(m, k), ldam,
                                   B(k, n), ldbk,
                            zbeta, C(m, n), ldcm);
                    }
                }
                //==========================================
                // A: PlasmaNoTrans / B: Plasma[Conj]Trans
                //==========================================
                else {
                    ldbn = BLKLDD(B, n);
                    for (k = 0; k < A.nt; k++) {
                        tempkn = k == A.nt-1 ? A.n-k*A.nb : A.nb;
                        zbeta = k == 0 ? beta : zone;
                        core_omp_zgemm(
                            transA, transB,
                            tempmm, tempnn, tempkn,
                            alpha, A(m, k), ldam,
                                   B(n, k), ldbn,
                            zbeta, C(m, n), ldcm);
                    }
                }
            }
            else {
                //==========================================
                // A: Plasma[Conj]Trans / B: PlasmaNoTrans
                //==========================================
                if (transB == PlasmaNoTrans) {
                    for (k = 0; k < A.mt; k++) {
                        tempkm = k == A.mt-1 ? A.m-k*A.mb : A.mb;
                        ldak = BLKLDD(A, k);
                        ldbk = BLKLDD(B, k);
                        zbeta = k == 0 ? beta : zone;
                        core_omp_zgemm(
                            transA, transB,
                            tempmm, tempnn, tempkm,
                            alpha, A(k, m), ldak,
                                   B(k, n), ldbk,
                            zbeta, C(m, n), ldcm);
                    }
                }
                //==============================================
                // A: Plasma[Conj]Trans / B: Plasma[Conj]Trans
                //==============================================
                else {
                    ldbn = BLKLDD(B, n);
                    for (k = 0; k < A.mt; k++) {
                        tempkm = k == A.mt-1 ? A.m-k*A.mb : A.mb;
                        ldak = BLKLDD(A, k);
                        zbeta = k == 0 ? beta : zone;
                        core_omp_zgemm(
                            transA, transB,
                            tempmm, tempnn, tempkm,
                            alpha, A(k, m), ldak,
                                   B(n, k), ldbn,
                            zbeta, C(m, n), ldcm);
                    }
                }
            }
        }
    }
}
