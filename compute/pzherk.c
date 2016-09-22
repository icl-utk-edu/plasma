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

#define A(m, n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)
#define C(m, n) (plasma_complex64_t*)plasma_tile_addr(C, m, n)

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
        tempnn = plasma_tile_ndim(C, n);
        ldan = plasma_tile_mdim(A, n);
        ldcn = plasma_tile_mdim(C, n);
        //=======================================
        // PlasmaNoTrans
        //=======================================
        if (trans == PlasmaNoTrans) {
            for (k = 0; k < A.nt; k++) {
                tempkn = plasma_tile_ndim(A, k);
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
                    tempmm = plasma_tile_mdim(C, m);
                    ldam = plasma_tile_mdim(A, m);
                    ldcm = plasma_tile_mdim(C, m);
                    for (k = 0; k < A.nt; k++) {
                        tempkn = plasma_tile_ndim(A, k);
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
                    tempmm = plasma_tile_mdim(C, m);
                    ldam = plasma_tile_mdim(A, m);
                    for (k = 0; k < A.nt; k++) {
                        tempkn = plasma_tile_ndim(A, k);
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
                tempkm = plasma_tile_mdim(A, k);
                ldak = plasma_tile_mdim(A, k);
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
                    tempmm = plasma_tile_mdim(C, m);
                    ldcm = plasma_tile_mdim(C, m);
                    for (k = 0; k < A.mt; k++) {
                        tempkm = plasma_tile_mdim(A, k);
                        ldak = plasma_tile_mdim(A, k);
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
                    tempmm = plasma_tile_mdim(C, m);
                    for (k = 0; k < A.mt; k++) {
                        tempkm = plasma_tile_mdim(A, k);
                        ldak = plasma_tile_mdim(A, k);
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
