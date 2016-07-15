/**
 *
 * @file pztrmm.c
 *
 *  PLASMA auxiliary routines
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 3.0.0
 * @author  Mathieu Faverge
 * @author  Maksims Abalenkovs
 * @date    2016-06-22
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

/***************************************************************************//**
 *  Parallel tile triangular matrix-matrix multiplication.
 *  @see PLASMA_zgemm_Tile_Async
 ******************************************************************************/
void plasma_pztrmm(PLASMA_enum side, PLASMA_enum uplo,
                   PLASMA_enum trans, PLASMA_enum diag,
                   PLASMA_Complex64_t alpha, PLASMA_desc A, PLASMA_desc B,
                   PLASMA_sequence *sequence, PLASMA_request *request)
{
    int k, m, n;
    int ldak, ldam, ldan, ldbk, ldbm;
    int tempkm, tempkn, tempmm, tempnn;

    PLASMA_Complex64_t zone = (PLASMA_Complex64_t)1.0;

    if (sequence->status != PLASMA_SUCCESS)
        return;

    //==========================================
    // PlasmaLeft / PlasmaUpper / PlasmaNoTrans
    //==========================================
    if (side == PlasmaLeft) {
        if (uplo == PlasmaUpper) {
            if (trans == PlasmaNoTrans) {
                for (m = 0; m < B.mt; m++) {
                    tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                    ldbm = BLKLDD(B, m);
                    ldam = BLKLDD(A, m);
                    for (n = 0; n < B.nt; n++) {
                        tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                        CORE_OMP_ztrmm(
                            side, uplo, trans, diag,
                            tempmm, tempnn,
                            alpha, A(m, m), ldam,  /* lda * tempkm */
                                   B(m, n), ldbm); /* ldb * tempnn */

                        for (k = m+1; k < A.mt; k++) {
                            tempkn = k == A.nt-1 ? A.n-k*A.nb : A.nb;
                            ldbk = BLKLDD(B, k);
                            CORE_OMP_zgemm(
                                trans, PlasmaNoTrans,
                                tempmm, tempnn, tempkn,
                                alpha, A(m, k), ldam,
                                       B(k, n), ldbk,
                                zone,  B(m, n), ldbm);
                        }
                    }
                }
            }
            //==============================================
            // PlasmaLeft / PlasmaUpper / Plasma[Conj]Trans
            //==============================================
            else {
                for (m = B.mt-1; m > -1; m--) {
                    tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                    ldbm = BLKLDD(B, m);
                    ldam = BLKLDD(A, m);
                    for (n = 0; n < B.nt; n++) {
                        tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                        CORE_OMP_ztrmm(
                            side, uplo, trans, diag,
                            tempmm, tempnn,
                            alpha, A(m, m), ldam,  /* lda * tempkm */
                                   B(m, n), ldbm); /* ldb * tempnn */

                        for (k = 0; k < m; k++) {
                            ldbk = BLKLDD(B, k);
                            ldak = BLKLDD(A, k);
                            CORE_OMP_zgemm(
                                trans, PlasmaNoTrans,
                                tempmm, tempnn, B.mb,
                                alpha, A(k, m), ldak,
                                       B(k, n), ldbk,
                                zone,  B(m, n), ldbm);
                        }
                    }
                }
            }
        }
        //==========================================
        // PlasmaLeft / PlasmaLower / PlasmaNoTrans
        //==========================================
        else {
            if (trans == PlasmaNoTrans) {
                for (m = B.mt-1; m > -1; m--) {
                    tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                    ldbm = BLKLDD(B, m);
                    ldam = BLKLDD(A, m);
                    for (n = 0; n < B.nt; n++) {
                        tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                        CORE_OMP_ztrmm(
                            side, uplo, trans, diag,
                            tempmm, tempnn,
                            alpha, A(m, m), ldam,  /* lda * tempkm */
                                   B(m, n), ldbm); /* ldb * tempnn */

                        for (k = 0; k < m; k++) {
                            ldbk = BLKLDD(B, k);
                            CORE_OMP_zgemm(
                                trans, PlasmaNoTrans,
                                tempmm, tempnn, B.mb,
                                alpha, A(m, k), ldam,
                                       B(k, n), ldbk,
                                zone,  B(m, n), ldbm);
                        }
                    }
                }
            }
            //==============================================
            // PlasmaLeft / PlasmaLower / Plasma[Conj]Trans
            //==============================================
            else {
                for (m = 0; m < B.mt; m++) {
                    tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                    ldbm = BLKLDD(B, m);
                    ldam = BLKLDD(A, m);
                    for (n = 0; n < B.nt; n++) {
                        tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                        CORE_OMP_ztrmm(
                            side, uplo, trans, diag,
                            tempmm, tempnn,
                            alpha, A(m, m), ldam,  /* lda * tempkm */
                                   B(m, n), ldbm); /* ldb * tempnn */

                        for (k = m+1; k < A.mt; k++) {
                            tempkm = k == A.mt-1 ? A.m-k*A.mb : A.mb;
                            ldak = BLKLDD(A, k);
                            ldbk = BLKLDD(B, k);
                            CORE_OMP_zgemm(
                                trans, PlasmaNoTrans,
                                tempmm, tempnn, tempkm,
                                alpha, A(k, m), ldak,
                                       B(k, n), ldbk,
                                zone,  B(m, n), ldbm);
                        }
                    }
                }
            }
        }
    }
    //===========================================
    // PlasmaRight / PlasmaUpper / PlasmaNoTrans
    //===========================================
    else {
        if (uplo == PlasmaUpper) {
            if (trans == PlasmaNoTrans) {
                for (n = B.nt-1; n > -1; n--) {
                    tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                    ldan = BLKLDD(A, n);
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = BLKLDD(B, m);
                        CORE_OMP_ztrmm(
                            side, uplo, trans, diag,
                            tempmm, tempnn,
                            alpha, A(n, n), ldan,  /* lda * tempkm */
                                   B(m, n), ldbm); /* ldb * tempnn */

                        for (k = 0; k < n; k++) {
                            ldak = BLKLDD(A, k);
                            CORE_OMP_zgemm(
                                PlasmaNoTrans, trans,
                                tempmm, tempnn, B.mb,
                                alpha, B(m, k), ldbm,
                                       A(k, n), ldak,
                                zone,  B(m, n), ldbm);
                        }
                    }
                }
            }
            //===============================================
            // PlasmaRight / PlasmaUpper / Plasma[Conj]Trans
            //===============================================
            else {
                for (n = 0; n < B.nt; n++) {
                    tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                    ldan = BLKLDD(A, n);
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = BLKLDD(B, m);
                        CORE_OMP_ztrmm(
                            side, uplo, trans, diag,
                            tempmm, tempnn,
                            alpha, A(n, n), ldan,  /* lda * tempkm */
                                   B(m, n), ldbm); /* ldb * tempnn */

                        for (k = n+1; k < A.mt; k++) {
                            tempkn = k == A.nt-1 ? A.n-k*A.nb : A.nb;
                            CORE_OMP_zgemm(
                                PlasmaNoTrans, trans,
                                tempmm, tempnn, tempkn,
                                alpha, B(m, k), ldbm,
                                       A(n, k), ldan,
                                zone,  B(m, n), ldbm);
                        }
                    }
                }
            }
        }
        //===========================================
        // PlasmaRight / PlasmaLower / PlasmaNoTrans
        //===========================================
        else {
            if (trans == PlasmaNoTrans) {
                for (n = 0; n < B.nt; n++) {
                    tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                    ldan = BLKLDD(A, n);
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = BLKLDD(B, m);
                        CORE_OMP_ztrmm(
                            side, uplo, trans, diag,
                            tempmm, tempnn,
                            alpha, A(n, n), ldan,  /* lda * tempkm */
                                   B(m, n), ldbm); /* ldb * tempnn */

                        for (k = n+1; k < A.mt; k++) {
                            tempkn = k == A.nt-1 ? A.n-k*A.nb : A.nb;
                            ldak = BLKLDD(A, k);
                            CORE_OMP_zgemm(
                                PlasmaNoTrans, trans,
                                tempmm, tempnn, tempkn,
                                alpha, B(m, k), ldbm,
                                       A(k, n), ldak,
                                zone,  B(m, n), ldbm);
                        }
                    }
                }
            }
            //===============================================
            // PlasmaRight / PlasmaLower / Plasma[Conj]Trans
            //===============================================
            else {
                for (n = B.nt-1; n > -1; n--) {
                    tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                    ldan = BLKLDD(A, n);
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = BLKLDD(B, m);
                        CORE_OMP_ztrmm(
                            side, uplo, trans, diag,
                            tempmm, tempnn,
                            alpha, A(n, n), ldan,  /* lda * tempkm */
                                   B(m, n), ldbm); /* ldb * tempnn */

                        for (k = 0; k < n; k++) {
                            CORE_OMP_zgemm(
                                PlasmaNoTrans, trans,
                                tempmm, tempnn, B.mb,
                                alpha, B(m, k), ldbm,
                                       A(n, k), ldan,
                                zone,  B(m, n), ldbm);
                        }
                    }
                }
            }
        }
    }
}
