/**
 *
 * @file pzpotrf.c
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
/***************************************************************************//**
 *  Parallel tile Cholesky factorization.
 * @see PLASMA_zpotrf_Tile_Async
 ******************************************************************************/
void plasma_pzpotrf(PLASMA_enum uplo, PLASMA_desc A,
                    PLASMA_sequence *sequence, PLASMA_request *request)
{
    int k, m, n;
    int ldak, ldam, ldan;
    int tempkm, tempmm;

    PLASMA_Complex64_t zone  = (PLASMA_Complex64_t) 1.0;
    PLASMA_Complex64_t mzone = (PLASMA_Complex64_t)-1.0;

    // Check sequence status.
    if (sequence->status != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);
        return;
    }

    //=======================================
    // PlasmaLower
    //=======================================
    if (uplo == PlasmaLower) {
        for (k = 0; k < A.mt; k++) {
            tempkm = k == A.mt-1 ? A.m-k*A.mb : A.mb;
            ldak = BLKLDD(A, k);
            CORE_OMP_zpotrf(
                PlasmaLower, tempkm,
                A(k, k), ldak);
            for (m = k+1; m < A.mt; m++) {
                tempmm = m == A.mt-1 ? A.m-m*A.mb : A.mb;
                ldam = BLKLDD(A, m);
                CORE_OMP_ztrsm(
                    PlasmaRight, PlasmaLower,
                    PlasmaConjTrans, PlasmaNonUnit,
                    tempmm, A.mb,
                    zone, A(k, k), ldak,
                          A(m, k), ldam);
            }
            for (m = k+1; m < A.mt; m++) {
                tempmm = m == A.mt-1 ? A.m-m*A.mb : A.mb;
                ldam = BLKLDD(A, m);
                CORE_OMP_zherk(
                    PlasmaLower, PlasmaNoTrans,
                    tempmm, A.mb,
                    -1.0, A(m, k), ldam,
                     1.0, A(m, m), ldam);
                for (n = k+1; n < m; n++) {
                    ldan = BLKLDD(A, n);
                    CORE_OMP_zgemm(
                        PlasmaNoTrans, PlasmaConjTrans,
                        tempmm, A.mb, A.mb,
                        mzone, A(m, k), ldam,
                               A(n, k), ldan,
                        zone,  A(m, n), ldam);
                }
            }
        }
    }
    //=======================================
    // PlasmaUpper
    //=======================================
    else {
        for (k = 0; k < A.nt; k++) {
            tempkm = k == A.nt-1 ? A.n-k*A.nb : A.nb;
            ldak = BLKLDD(A, k);
            CORE_OMP_zpotrf(
                PlasmaUpper, tempkm,
                A(k, k), ldak);
            for (m = k+1; m < A.nt; m++) {
                tempmm = m == A.nt-1 ? A.n-m*A.nb : A.nb;
                CORE_OMP_ztrsm(
                    PlasmaLeft, PlasmaUpper,
                    PlasmaConjTrans, PlasmaNonUnit,
                    A.nb, tempmm,
                    zone, A(k, k), ldak,
                          A(k, m), ldak);
            }
            for (m = k+1; m < A.nt; m++) {
                tempmm = m == A.nt-1 ? A.n-m*A.nb : A.nb;
                ldam = BLKLDD(A, m);
                CORE_OMP_zherk(
                    PlasmaUpper, PlasmaConjTrans,
                    tempmm, A.mb,
                    -1.0, A(k, m), ldak,
                     1.0, A(m, m), ldam);
                for (n = k+1; n < m; n++) {
                    ldan = BLKLDD(A, n);
                    CORE_OMP_zgemm(
                        PlasmaConjTrans, PlasmaNoTrans,
                        A.mb, tempmm, A.mb,
                        mzone, A(k, n), ldak,
                               A(k, m), ldak,
                        zone,  A(n, m), ldan);
                }
            }
        }
    }
}
