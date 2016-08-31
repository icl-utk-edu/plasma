/**
 *
 * @file pztbsm.c
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

#define KUT ((A.ku+A.kl+A.nb-1)/A.nb) // number of tiles in upper band (not including diagonal)
/* band matrix storage (not skewed) */
#define A(m,n) ((PLASMA_Complex64_t*)plasma_getaddr(A, KUT+(m)-(n), (n)))
#define BLKLDD_BAND(A, m, n) BLKLDD((A), KUT+(m)-(n))
#define B(m,n) ((PLASMA_Complex64_t*)plasma_getaddr(B, (m), (n)))
#define IPIV(k) (&(IPIV[B.mb*(k)]))

/***************************************************************************//**
 *  Parallel tile triangular solve - dynamic scheduling
 **/
void plasma_pztbsm(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
                   PLASMA_Complex64_t alpha, PLASMA_desc A, PLASMA_desc B, const int *IPIV,
                   PLASMA_sequence *sequence, PLASMA_request *request)
{
    int k, m, n;
    int tempi;
    int tempkm, tempmm, tempnn;

    PLASMA_Complex64_t zone       = (PLASMA_Complex64_t) 1.0;
    PLASMA_Complex64_t mzone      = (PLASMA_Complex64_t)-1.0;
    PLASMA_Complex64_t lalpha;

    if (sequence->status != PLASMA_SUCCESS)
        return;

    /*
     *  PlasmaLeft / PlasmaUpper / PlasmaNoTrans
     */
    if (side == PlasmaLeft) {
        if (uplo == PlasmaUpper) {
            if (trans == PlasmaNoTrans) {
                for (k = 0; k < B.mt; k++) {
                    tempkm = k == 0 ? B.m-(B.mt-1)*B.mb : B.mb;
                    lalpha = k == 0 ? alpha : zone;
                    for (n = 0; n < B.nt; n++) {
                        tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                        CORE_OMP_ztrsm(
                            side, uplo, trans, diag,
                            tempkm, tempnn, 
                            lalpha, A(B.mt-1-k, B.mt-1-k), BLKLDD_BAND(A, B.mt-1-k, B.mt-1-k),
                                    B(B.mt-1-k,        n), BLKLDD(B, B.mt-1-k));
                    }
                    for (m = imax(0, (B.mt-1-k)-A.kut+1); m < B.mt-1-k; m++) {
                        for (n = 0; n < B.nt; n++) {
                            tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                            CORE_OMP_zgemm(
                                PlasmaNoTrans, PlasmaNoTrans,
                                B.mb, tempnn, tempkm, 
                                mzone,  A(m, B.mt-1-k), BLKLDD_BAND(A, m, B.mt-1-k),
                                        B(B.mt-1-k, n), BLKLDD(B, B.mt-1-k),
                                lalpha, B(m, n       ), BLKLDD(B, m));
                        }
                    }
                }
            }
            /*
             *  PlasmaLeft / PlasmaUpper / Plasma[Conj]Trans
             */
            else {
                for (k = 0; k < B.mt; k++) {
                    tempkm = k == B.mt-1 ? B.m-k*B.mb : B.mb;
                    lalpha = k == 0 ? alpha : zone;
                    for (n = 0; n < B.nt; n++) {
                        tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                        CORE_OMP_ztrsm(
                            side, uplo, trans, diag,
                            tempkm, tempnn, 
                            lalpha, A(k, k), BLKLDD_BAND(A, k, k),
                                    B(k, n), BLKLDD(B, k));
                    }
                    for (m = k+1; m < imin(A.mt, k+A.kut); m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        for (n = 0; n < B.nt; n++) {
                            tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                            CORE_OMP_zgemm(
                                trans, PlasmaNoTrans,
                                tempmm, tempnn, B.mb, 
                                mzone,  A(k, m), BLKLDD_BAND(A, k, m),
                                        B(k, n), BLKLDD(B, k),
                                lalpha, B(m, n), BLKLDD(B, m));
                        }
                    }
                }
            }
        }
        /*
         *  PlasmaLeft / PlasmaLower / PlasmaNoTrans
         */
        else {
            if (trans == PlasmaNoTrans) {
                for (k = 0; k < B.mt; k++) {
                    tempkm = k == B.mt-1 ? B.m-k*B.mb : B.mb;
                    lalpha = k == 0 ? alpha : zone;

                    tempi = k*B.mb;
                    for (n = 0; n < B.nt; n++) {
                        tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                        CORE_OMP_zlaswp_ontile(
                            B, k, n, B.m-tempi, tempnn,
                            1, tempkm, IPIV(k), 1);

                        CORE_OMP_ztrsm(
                            side, uplo, trans, diag,
                            tempkm, tempnn, 
                            lalpha, A(k, k), BLKLDD_BAND(A, B.mt-1-k, B.mt-1-k),
                                    B(k, n), BLKLDD(B, k));
                    }
                    for (m = k+1; m < imin(k+A.klt, A.mt); m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        for (n = 0; n < B.nt; n++) {
                            tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                            CORE_OMP_zgemm(
                                PlasmaNoTrans, PlasmaNoTrans,
                                tempmm, tempnn, B.mb, 
                                mzone,  A(m, k), BLKLDD_BAND(A, m, k),
                                        B(k, n), BLKLDD(B, k),
                                lalpha, B(m, n), BLKLDD(B, m));
                        }
                    }
                }
            }
            /*
             *  PlasmaLeft / PlasmaLower / Plasma[Conj]Trans
             */
            else {
                for (k = 0; k < B.mt; k++) {
                    tempkm = k == 0 ? B.m-(B.mt-1)*B.mb : B.mb;
                    lalpha = k == 0 ? alpha : zone;
                    for (m = (B.mt-1-k)+1; m < imin((B.mt-1-k)+A.klt, A.mt); m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        for (n = 0; n < B.nt; n++) {
                            tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                            CORE_OMP_zgemm(
                                trans, PlasmaNoTrans,
                                tempkm, tempnn, tempmm, 
                                mzone,  A(m, B.mt-1-k), BLKLDD_BAND(A, m, B.mt-1-k),
                                        B(m, n       ), BLKLDD(B, m),
                                lalpha, B(B.mt-1-k, n), BLKLDD(B, B.mt-1-k));
                        }
                    }
                    for (n = 0; n < B.nt; n++) {
                        tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                        tempi  = (B.mt-1-k)*B.mb;
                        CORE_OMP_ztrsm(
                            side, uplo, trans, diag,
                            tempkm, tempnn, 
                            lalpha, A(B.mt-1-k, B.mt-1-k), BLKLDD_BAND(A, B.mt-1-k, B.mt-1-k),
                                    B(B.mt-1-k,        n), BLKLDD(B, B.mt-1-k));
                        CORE_OMP_zlaswp_ontile(
                            B, B.mt-1-k, n, B.m-tempi, tempnn,
                            1, tempkm, IPIV(B.mt-1-k), -1);
                    }
                }
            }
        }
    }
    /*
     *  PlasmaRight / PlasmaUpper / PlasmaNoTrans
     */
    else {
#if 0
        if (uplo == PlasmaUpper) {
            if (trans == PlasmaNoTrans) {
                for (k = 0; k < B.nt; k++) {
                    tempkn = k == B.nt-1 ? B.n-k*B.nb : B.nb;
                    ldak = BLKLDD(A, k);
                    lalpha = k == 0 ? alpha : zone;
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = BLKLDD(B, m);
                        QUARK_CORE_ztrsm(
                            plasma->quark, &task_flags,
                            side, uplo, trans, diag,
                            tempmm, tempkn, A.mb,
                            lalpha, A(k, k), ldak,  /* lda * tempkn */
                                    B(m, k), ldbm); /* ldb * tempkn */
                    }
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = BLKLDD(B, m);
                        for (n = k+1; n < B.nt; n++) {
                            tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                            QUARK_CORE_zgemm(
                                plasma->quark, &task_flags,
                                PlasmaNoTrans, PlasmaNoTrans,
                                tempmm, tempnn, B.mb, A.mb,
                                mzone,  B(m, k), ldbm,  /* ldb * B.mb   */
                                        A(k, n), ldak,  /* lda * tempnn */
                                lalpha, B(m, n), ldbm); /* ldb * tempnn */
                        }
                    }
                }
            }
            /*
             *  PlasmaRight / PlasmaUpper / Plasma[Conj]Trans
             */
            else {
                for (k = 0; k < B.nt; k++) {
                    tempkn = k == 0 ? B.n-(B.nt-1)*B.nb : B.nb;
                    ldak = BLKLDD(A, B.nt-1-k);
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = BLKLDD(B, m);
                        QUARK_CORE_ztrsm(
                            plasma->quark, &task_flags,
                            side, uplo, trans, diag,
                            tempmm, tempkn, A.mb,
                            alpha, A(B.nt-1-k, B.nt-1-k), ldak,  /* lda * tempkn */
                                   B(       m, B.nt-1-k), ldbm); /* ldb * tempkn */

                        for (n = k+1; n < B.nt; n++) {
                            ldan = BLKLDD(A, B.nt-1-n);
                            QUARK_CORE_zgemm(
                                plasma->quark, &task_flags,
                                PlasmaNoTrans, trans,
                                tempmm, B.nb, tempkn, A.mb,
                                minvalpha, B(m,        B.nt-1-k), ldbm,  /* ldb  * tempkn */
                                           A(B.nt-1-n, B.nt-1-k), ldan, /* A.mb * tempkn (Never last row) */
                                zone,      B(m,        B.nt-1-n), ldbm); /* ldb  * B.nb   */
                        }
                    }
                }
            }
        }
        /*
         *  PlasmaRight / PlasmaLower / PlasmaNoTrans
         */
        else {
            if (trans == PlasmaNoTrans) {
                for (k = 0; k < B.nt; k++) {
                    tempkn = k == 0 ? B.n-(B.nt-1)*B.nb : B.nb;
                    ldak = BLKLDD(A, B.nt-1-k);
                    lalpha = k == 0 ? alpha : zone;
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = BLKLDD(B, m);
                        QUARK_CORE_ztrsm(
                            plasma->quark, &task_flags,
                            side, uplo, trans, diag,
                            tempmm, tempkn, A.mb,
                            lalpha, A(B.nt-1-k, B.nt-1-k), ldak,  /* lda * tempkn */
                                    B(       m, B.nt-1-k), ldbm); /* ldb * tempkn */

                        for (n = k+1; n < B.nt; n++) {
                            QUARK_CORE_zgemm(
                                plasma->quark, &task_flags,
                                PlasmaNoTrans, PlasmaNoTrans,
                                tempmm, B.nb, tempkn, A.mb,
                                mzone,  B(m,        B.nt-1-k), ldbm,  /* ldb * tempkn */
                                        A(B.nt-1-k, B.nt-1-n), ldak,  /* lda * B.nb   */
                                lalpha, B(m,        B.nt-1-n), ldbm); /* ldb * B.nb   */
                        }
                    }
                }
            }
            /*
             *  PlasmaRight / PlasmaLower / Plasma[Conj]Trans
             */
            else {
                for (k = 0; k < B.nt; k++) {
                    tempkn = k == B.nt-1 ? B.n-k*B.nb : B.nb;
                    ldak = BLKLDD(A, k);
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = BLKLDD(B, m);
                        QUARK_CORE_ztrsm(
                            plasma->quark, &task_flags,
                            side, uplo, trans, diag,
                            tempmm, tempkn, A.mb,
                            alpha, A(k, k), ldak,  /* lda * tempkn */
                                   B(m, k), ldbm); /* ldb * tempkn */

                        for (n = k+1; n < B.nt; n++) {
                            tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                            ldan = BLKLDD(A, n);
                            QUARK_CORE_zgemm(
                                plasma->quark, &task_flags,
                                PlasmaNoTrans, trans,
                                tempmm, tempnn, B.mb, A.mb,
                                minvalpha, B(m, k), ldbm,  /* ldb  * tempkn */
                                           A(n, k), ldan, /* ldan * tempkn */
                                zone,      B(m, n), ldbm); /* ldb  * tempnn */
                        }
                    }
                }
            }
        }
#endif
    }
}
