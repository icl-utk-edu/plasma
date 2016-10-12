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
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_types.h"
#include "plasma_workspace.h"
#include "core_blas.h"

#define A(m,n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)
#define B(m,n) (plasma_complex64_t*)plasma_tile_addr(B, m, n)
#define IPIV(k) &(IPIV[B.mb*(k)])

/***************************************************************************//**
 *  Parallel tile triangular solve - dynamic scheduling
 **/
void plasma_pztbsm(plasma_enum_t side, plasma_enum_t uplo,
                   plasma_enum_t trans, plasma_enum_t diag,
                   plasma_complex64_t alpha, plasma_desc_t A,
                                             plasma_desc_t B,
                   const int *IPIV,
                   plasma_sequence_t *sequence, plasma_request_t *request)
{
    int k, m, n;
    int tempkm, tempmm, tempnn;

    plasma_complex64_t lalpha;

    if (sequence->status != PlasmaSuccess)
        return;

    /*
     *  PlasmaLeft / PlasmaUpper / PlasmaNoTrans
     */
    if (side == PlasmaLeft) {
        if (uplo == PlasmaUpper) {
            if (trans == PlasmaNoTrans) {
                for (k = 0; k < B.mt; k++) {
                    tempkm = k == 0 ? B.m-(B.mt-1)*B.mb : B.mb;
                    lalpha = k == 0 ? alpha : 1.0;
                    for (n = 0; n < B.nt; n++) {
                        tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            tempkm, tempnn,
                            lalpha, A(B.mt-1-k, B.mt-1-k), BLKLDD_BAND(uplo, A, B.mt-1-k, B.mt-1-k),
                                    B(B.mt-1-k,        n), plasma_tile_mmain(B, B.mt-1-k),
                            sequence, request);
                    }
                    for (m = imax(0, (B.mt-1-k)-A.kut+1); m < B.mt-1-k; m++) {
                        for (n = 0; n < B.nt; n++) {
                            tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                            core_omp_zgemm(
                                PlasmaNoTrans, PlasmaNoTrans,
                                B.mb, tempnn, tempkm,
                                -1.0,  A(m, B.mt-1-k), BLKLDD_BAND(uplo, A, m, B.mt-1-k),
                                        B(B.mt-1-k, n), plasma_tile_mmain(B, B.mt-1-k),
                                lalpha, B(m, n       ), plasma_tile_mmain(B, m),
                                sequence, request);
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
                    lalpha = k == 0 ? alpha : 1.0;
                    for (n = 0; n < B.nt; n++) {
                        tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            tempkm, tempnn,
                            lalpha, A(k, k), BLKLDD_BAND(uplo, A, k, k),
                                    B(k, n), plasma_tile_mmain(B, k),
                            sequence, request);
                    }
                    for (m = k+1; m < imin(A.mt, k+A.kut); m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        for (n = 0; n < B.nt; n++) {
                            tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                            core_omp_zgemm(
                                trans, PlasmaNoTrans,
                                tempmm, tempnn, B.mb,
                                -1.0,  A(k, m), BLKLDD_BAND(uplo, A, k, m),
                                        B(k, n), plasma_tile_mmain(B, k),
                                lalpha, B(m, n), plasma_tile_mmain(B, m),
                                sequence, request);
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
                    lalpha = k == 0 ? alpha : 1.0;

                    for (n = 0; n < B.nt; n++) {
                        tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                        if (IPIV != NULL) {
                            #ifdef ZLASWP_ONTILE
                            // commented out because it takes descriptor
                            tempi = k*B.mb;
                            core_omp_zlaswp_ontile(
                                B, k, n, B.m-tempi, tempnn,
                                1, tempkm, IPIV(k), 1);
                            #endif
                        }
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            tempkm, tempnn,
                            lalpha, A(k, k), BLKLDD_BAND(uplo, A, B.mt-1-k, B.mt-1-k),
                                    B(k, n), plasma_tile_mmain(B, k),
                            sequence, request);
                    }
                    for (m = k+1; m < imin(k+A.klt, A.mt); m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        for (n = 0; n < B.nt; n++) {
                            tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                            core_omp_zgemm(
                                PlasmaNoTrans, PlasmaNoTrans,
                                tempmm, tempnn, B.mb,
                                -1.0,  A(m, k), BLKLDD_BAND(uplo, A, m, k),
                                        B(k, n), plasma_tile_mmain(B, k),
                                lalpha, B(m, n), plasma_tile_mmain(B, m),
                                sequence, request);
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
                    lalpha = k == 0 ? alpha : 1.0;
                    for (m = (B.mt-1-k)+1; m < imin((B.mt-1-k)+A.klt, A.mt); m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        for (n = 0; n < B.nt; n++) {
                            tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                            core_omp_zgemm(
                                trans, PlasmaNoTrans,
                                tempkm, tempnn, tempmm,
                                -1.0,  A(m, B.mt-1-k), BLKLDD_BAND(uplo, A, m, B.mt-1-k),
                                        B(m, n       ), plasma_tile_mmain(B, m),
                                lalpha, B(B.mt-1-k, n), plasma_tile_mmain(B, B.mt-1-k),
                                sequence, request);
                        }
                    }
                    for (n = 0; n < B.nt; n++) {
                        tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                        core_omp_ztrsm(
                            side, uplo, trans, diag,
                            tempkm, tempnn,
                            lalpha, A(B.mt-1-k, B.mt-1-k), BLKLDD_BAND(uplo, A, B.mt-1-k, B.mt-1-k),
                                    B(B.mt-1-k,        n), plasma_tile_mmain(B, B.mt-1-k),
                            sequence, request);
                        if (IPIV != NULL) {
                            #ifdef ZLASWP_ONTILE
                            // commented out because it takes descriptor
                            tempi  = (B.mt-1-k)*B.mb;
                            core_omp_zlaswp_ontile(
                                B, B.mt-1-k, n, B.m-tempi, tempnn,
                                1, tempkm, IPIV(B.mt-1-k), -1);
                            #endif
                        }
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
                    ldak = plasma_tile_mmain(A, k);
                    lalpha = k == 0 ? alpha : 1.0;
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = plasma_tile_mmain(B, m);
                        QUARK_core_ztrsm(
                            plasma->quark, &task_flags,
                            side, uplo, trans, diag,
                            tempmm, tempkn, A.mb,
                            lalpha, A(k, k), ldak
                                    B(m, k), ldbm);
                    }
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = plasma_tile_mmain(B, m);
                        for (n = k+1; n < B.nt; n++) {
                            tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                            QUARK_core_zgemm(
                                plasma->quark, &task_flags,
                                PlasmaNoTrans, PlasmaNoTrans,
                                tempmm, tempnn, B.mb, A.mb,
                                -1.0,  B(m, k), ldbm,
                                        A(k, n), ldak,
                                lalpha, B(m, n), ldbm);
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
                    ldak = plasma_tile_mmain(A, B.nt-1-k);
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = plasma_tile_mmain(B, m);
                        QUARK_core_ztrsm(
                            plasma->quark, &task_flags,
                            side, uplo, trans, diag,
                            tempmm, tempkn, A.mb,
                            alpha, A(B.nt-1-k, B.nt-1-k), ldak,
                                   B(       m, B.nt-1-k), ldbm);

                        for (n = k+1; n < B.nt; n++) {
                            ldan = plasma_tile_mmain(A, B.nt-1-n);
                            QUARK_core_zgemm(
                                plasma->quark, &task_flags,
                                PlasmaNoTrans, trans,
                                tempmm, B.nb, tempkn, A.mb,
                                minvalpha, B(m,        B.nt-1-k), ldbm,
                                           A(B.nt-1-n, B.nt-1-k), ldan,
                                1.0,      B(m,        B.nt-1-n), ldbm);
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
                    ldak = plasma_tile_mmain(A, B.nt-1-k);
                    lalpha = k == 0 ? alpha : 1.0;
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = plasma_tile_mmain(B, m);
                        QUARK_core_ztrsm(
                            plasma->quark, &task_flags,
                            side, uplo, trans, diag,
                            tempmm, tempkn, A.mb,
                            lalpha, A(B.nt-1-k, B.nt-1-k), ldak,
                                    B(       m, B.nt-1-k), ldbm);

                        for (n = k+1; n < B.nt; n++) {
                            QUARK_core_zgemm(
                                plasma->quark, &task_flags,
                                PlasmaNoTrans, PlasmaNoTrans,
                                tempmm, B.nb, tempkn, A.mb,
                                -1.0,  B(m,        B.nt-1-k), ldbm,
                                        A(B.nt-1-k, B.nt-1-n), ldak,
                                lalpha, B(m,        B.nt-1-n), ldbm);
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
                    ldak = plasma_tile_mmain(A, k);
                    for (m = 0; m < B.mt; m++) {
                        tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
                        ldbm = plasma_tile_mmain(B, m);
                        QUARK_core_ztrsm(
                            plasma->quark, &task_flags,
                            side, uplo, trans, diag,
                            tempmm, tempkn, A.mb,
                            alpha, A(k, k), ldak,
                                   B(m, k), ldbm);

                        for (n = k+1; n < B.nt; n++) {
                            tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                            ldan = plasma_tile_mmain(A, n);
                            QUARK_core_zgemm(
                                plasma->quark, &task_flags,
                                PlasmaNoTrans, trans,
                                tempmm, tempnn, B.mb, A.mb,
                                minvalpha, B(m, k), ldbm,
                                           A(n, k), ldan,
                                1.0,      B(m, n), ldbm);
                        }
                    }
                }
            }
        }
#endif
    }
}
