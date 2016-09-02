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

#define A(m,n) ((PLASMA_Complex64_t*)plasma_getaddr_band(uplo, A, m, n))

/***************************************************************************//**
 *  Parallel tile Cholesky factorization of a band matrix.
 * @see PLASMA_zgbtrf_Tile_Async
 ******************************************************************************/
void plasma_pzpbtrf(PLASMA_enum uplo, PLASMA_desc A,
                    PLASMA_sequence *sequence, PLASMA_request *request)
{
    int k, m, n;
    int tempkm, tempmm, tempmn;

    PLASMA_Complex64_t zone  = (PLASMA_Complex64_t)1.0;
    PLASMA_Complex64_t mzone = (PLASMA_Complex64_t)-1.0;

    if (sequence->status != PLASMA_SUCCESS)
        return;


    //=======================================
    // PlasmaLower
    //=======================================
    if (uplo == PlasmaLower) {
        for (k = 0; k < A.mt; k++)
        {
            tempkm = k == A.mt-1 ? A.m-k*A.mb : A.mb;
            CORE_OMP_zpotrf(
                PlasmaLower, tempkm,
                A(k, k), BLKLDD_BAND(uplo, A, k, k),
                sequence, request, A.nb*k);
            for (m = k+1; m < imin(A.nt, k+A.klt); m++)
            {
                tempmm = m == A.mt-1 ? A.m-m*A.mb : A.mb;
                CORE_OMP_ztrsm(
                    PlasmaRight, PlasmaLower,
                    PlasmaConjTrans, PlasmaNonUnit,
                    tempmm, tempkm,
                    zone, A(k, k), BLKLDD_BAND(uplo, A, k, k),
                          A(m, k), BLKLDD_BAND(uplo, A, m, k));
                CORE_OMP_zherk(
                    PlasmaLower, PlasmaNoTrans,
                    tempmm, A.mb,
                    -1.0, A(m, k), BLKLDD_BAND(uplo, A, m, k),
                     1.0, A(m, m), BLKLDD_BAND(uplo, A, m, m));
                for (n = imax(k+1, m-A.klt); n < m; n++)
                {
                    tempmn = n == A.nt-1 ? A.n-n*A.nb : A.nb;
                    CORE_OMP_zgemm(
                        PlasmaNoTrans, PlasmaConjTrans,
                        tempmm, tempmn, tempkm,
                        mzone, A(m, k), BLKLDD_BAND(uplo, A, m, k),
                               A(n, k), BLKLDD_BAND(uplo, A, n, k),
                        zone,  A(m, n), BLKLDD_BAND(uplo, A, m, n));
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
            CORE_OMP_zpotrf(
                PlasmaUpper, tempkm,
                A(k, k), BLKLDD_BAND(uplo, A, k, k),
                sequence, request, A.nb*k);
            for (m = k+1; m < imin(A.nt, k+A.kut); m++) {
                tempmm = m == A.nt-1 ? A.n-m*A.nb : A.nb;
                CORE_OMP_ztrsm(
                    PlasmaLeft, PlasmaUpper,
                    PlasmaConjTrans, PlasmaNonUnit,
                    A.nb, tempmm,
                    zone, A(k, k), BLKLDD_BAND(uplo, A, k, k),
                          A(k, m), BLKLDD_BAND(uplo, A, k, m));
                CORE_OMP_zherk(
                    PlasmaUpper, PlasmaConjTrans,
                    tempmm, A.mb,
                    -1.0, A(k, m), BLKLDD_BAND(uplo, A, k, m),
                     1.0, A(m, m), BLKLDD_BAND(uplo, A, m, m));
                for (n = imax(k+1, m-A.kut); n < m; n++) {
                    CORE_OMP_zgemm(
                        PlasmaConjTrans, PlasmaNoTrans,
                        A.mb, tempmm, A.mb,
                        mzone, A(k, n), BLKLDD_BAND(uplo, A, k, n),
                               A(k, m), BLKLDD_BAND(uplo, A, k, m),
                        zone,  A(n, m), BLKLDD_BAND(uplo, A, n, m));
                }
            }
        }
    }
}

