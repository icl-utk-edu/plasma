/**
 *
 * @file pzgemm.c
 *
 *  PLASMA auxiliary routines.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver.
 *
 * @version 3.0.0
 * @author Jakub Kurzak
 * @date 2016-01-01
 * @precisions normal z -> s d c
 *
 **/
// #include "common.h"

#include "plasma.h"

// #define A(m, n) BLKADDR(A, PLASMA_Complex64_t, m, n)
// #define B(m, n) BLKADDR(B, PLASMA_Complex64_t, m, n)
// #define C(m, n) BLKADDR(C, PLASMA_Complex64_t, m, n)
/***************************************************************************//**
 *  Parallel tile matrix-matrix multiplication.
 **/
void plasma_pzgemm(PLASMA_enum transA, PLASMA_enum transB,
                   PLASMA_Complex64_t alpha, PLASMA_desc A,
                                             PLASMA_desc B,
                   PLASMA_Complex64_t beta,  PLASMA_desc C,
                   PLASMA_sequence *sequence, PLASMA_request *request)
{
#if 0 // ==========
    int m, n, k;
    int ldam, ldak, ldbn, ldbk, ldcm;
    int tempmm, tempnn, tempkn, tempkm;

    PLASMA_Complex64_t zbeta;
    PLASMA_Complex64_t zone = (PLASMA_Complex64_t)1.0;

    if (sequence->status != PLASMA_SUCCESS)
        return;

    plasma_context_t plasma = plasma_context_self();

    for (m = 0; m < C.mt; m++) {
        tempmm = m == C.mt-1 ? C.m-m*C.mb : C.mb;
        ldcm = BLKLDD(C, m);
        for (n = 0; n < C.nt; n++) {
            tempnn = n == C.nt-1 ? C.n-n*C.nb : C.nb;
            /*
             *  A: PlasmaNoTrans / B: PlasmaNoTrans
             */
            if (transA == PlasmaNoTrans) {
                ldam = BLKLDD(A, m);
                if (transB == PlasmaNoTrans) {
                    for (k = 0; k < A.nt; k++) {
                        tempkn = k == A.nt-1 ? A.n-k*A.nb : A.nb;
                        ldbk = BLKLDD(B, k);
                        zbeta = k == 0 ? beta : zone;
                        CORE_OMP_zgemm(
                            transA, transB,
                            tempmm, tempnn, tempkn, A.mb,
                            alpha, A(m, k), ldam,  /* lda * Z */
                                   B(k, n), ldbk,  /* ldb * Y */
                            zbeta, C(m, n), ldcm); /* ldc * Y */
                    }
                }
                /*
                 *  A: PlasmaNoTrans / B: Plasma[Conj]Trans
                 */
                else {
                    ldbn = BLKLDD(B, n);
                    for (k = 0; k < A.nt; k++) {
                        tempkn = k == A.nt-1 ? A.n-k*A.nb : A.nb;
                        zbeta = k == 0 ? beta : zone;
                        CORE_OMP_zgemm(
                            transA, transB,
                            tempmm, tempnn, tempkn, A.mb,
                            alpha, A(m, k), ldam,  /* lda * Z */
                                   B(n, k), ldbn,  /* ldb * Z */
                            zbeta, C(m, n), ldcm); /* ldc * Y */
                    }
                }
            }
            /*
             *  A: Plasma[Conj]Trans / B: PlasmaNoTrans
             */
            else {
                if (transB == PlasmaNoTrans) {
                    for (k = 0; k < A.mt; k++) {
                        tempkm = k == A.mt-1 ? A.m-k*A.mb : A.mb;
                        ldak = BLKLDD(A, k);
                        ldbk = BLKLDD(B, k);
                        zbeta = k == 0 ? beta : zone;
                        CORE_OMP_zgemm(
                            transA, transB,
                            tempmm, tempnn, tempkm, A.mb,
                            alpha, A(k, m), ldak,  /* lda * X */
                                   B(k, n), ldbk,  /* ldb * Y */
                            zbeta, C(m, n), ldcm); /* ldc * Y */
                    }
                }
                /*
                 *  A: Plasma[Conj]Trans / B: Plasma[Conj]Trans
                 */
                else {
                    ldbn = BLKLDD(B, n);
                    for (k = 0; k < A.mt; k++) {
                        tempkm = k == A.mt-1 ? A.m-k*A.mb : A.mb;
                        ldak = BLKLDD(A, k);
                        zbeta = k == 0 ? beta : zone;
                        CORE_OMP_zgemm(
                            transA, transB,
                            tempmm, tempnn, tempkm, A.mb,
                            alpha, A(k, m), ldak,  /* lda * X */
                                   B(n, k), ldbn,  /* ldb * Z */
                            zbeta, C(m, n), ldcm); /* ldc * Y */
                    }
                }
            }
        }
    }
#endif  // ========== #if 0
}
