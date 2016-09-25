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
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_types.h"
#include "plasma_workspace.h"
#include "core_blas.h"

#define A(m, n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)
#define B(m, n) (plasma_complex64_t*)plasma_tile_addr(B, m, n)
#define C(m, n) (plasma_complex64_t*)plasma_tile_addr(C, m, n)

/***************************************************************************//**
 * Parallel tile Hermitian rank 2k update.
 * @see plasma_omp_zher2k
 ******************************************************************************/
void plasma_pzher2k(plasma_enum_t uplo, plasma_enum_t trans,
                    plasma_complex64_t alpha, plasma_desc_t A,
                    plasma_desc_t B, double beta,  plasma_desc_t C,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    int m, n, k;
    int ldak, ldam, ldan, ldcm, ldcn, ldbk, ldbm, ldbn;
    int tempnn, tempmm, tempkn, tempkm;

    plasma_complex64_t zone = 1.0;
    plasma_complex64_t zbeta;
    double dbeta;

    // Check sequence status.
    if (sequence->status != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorSequence);
        return;
    }

    for (n = 0; n < C.nt; n++) {
        tempnn = plasma_tile_ndim(C, n);
        ldan   = plasma_tile_mdim(A, n);
        ldbn   = plasma_tile_mdim(B, n);
        ldcn   = plasma_tile_mdim(C, n);
        //================
        // PlasmaNoTrans
        //================
        if (trans == PlasmaNoTrans) {
            for (k = 0; k < A.nt; k++) {
                tempkn = plasma_tile_ndim(A, k);
                dbeta = k == 0 ? beta : 1.0;
                core_omp_zher2k(
                    uplo, trans,
                    tempnn, tempkn,
                    alpha, A(n, k), ldan,
                           B(n, k), ldbn,
                    dbeta, C(n, n), ldcn);
            }
            //==============================
            // PlasmaNoTrans / PlasmaLower
            //==============================
            if (uplo == PlasmaLower) {
                for (m = n+1; m < C.mt; m++) {
                    tempmm = plasma_tile_mdim(C, m);
                    ldam   = plasma_tile_mdim(A, m);
                    ldbm   = plasma_tile_mdim(B, m);
                    ldcm   = plasma_tile_mdim(C, m);
                    for (k = 0; k < A.nt; k++) {
                        tempkn = plasma_tile_ndim(A, k);
                        zbeta = k == 0 ? beta : zone;
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
            //==============================
            // PlasmaNoTrans / PlasmaUpper
            //==============================
            else {
                for (m = n+1; m < C.mt; m++) {
                    tempmm = plasma_tile_mdim(C, m);
                    ldam   = plasma_tile_mdim(A, m);
                    ldbm   = plasma_tile_mdim(B, m);
                    for (k = 0; k < A.nt; k++) {
                        tempkn = plasma_tile_ndim(A, k);
                        zbeta = k == 0 ? beta : zone;
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
        //=====================
        // Plasma[_Conj]Trans
        //=====================
        else {
            for (k = 0; k < A.mt; k++) {
                tempkm = plasma_tile_mdim(A, k);
                ldak   = plasma_tile_mdim(A, k);
                ldbk   = plasma_tile_mdim(B, k);
                dbeta = k == 0 ? beta : 1.0;
                core_omp_zher2k(
                    uplo, trans,
                    tempnn, tempkm,
                    alpha, A(k, n), ldak,
                           B(k, n), ldbk,
                    dbeta, C(n, n), ldcn);
            }
            //===================================
            // Plasma[_Conj]Trans / PlasmaLower
            //===================================
            if (uplo == PlasmaLower) {
                for (m = n+1; m < C.mt; m++) {
                    tempmm = plasma_tile_mdim(C, m);
                    ldcm   = plasma_tile_mdim(C, m);
                    for (k = 0; k < A.mt; k++) {
                        tempkm = plasma_tile_mdim(A, k);
                        ldak   = plasma_tile_mdim(A, k);
                        ldbk   = plasma_tile_mdim(B, k);
                        zbeta = k == 0 ? beta : zone;
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
            //===================================
            // Plasma[_Conj]Trans / PlasmaUpper
            //===================================
            else {
                for (m = n+1; m < C.mt; m++) {
                    tempmm = plasma_tile_mdim(C, m);
                    for (k = 0; k < A.mt; k++) {
                        tempkm = plasma_tile_mdim(A, k);
                        ldak   = plasma_tile_mdim(A, k);
                        ldbk   = plasma_tile_mdim(B, k);
                        zbeta = k == 0 ? beta : zone;
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
