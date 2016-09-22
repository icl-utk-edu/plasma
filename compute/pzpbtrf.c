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

#define A(m,n) ((plasma_complex64_t*)plasma_tile_addr(A, m, n))

/***************************************************************************//**
 *  Parallel tile Cholesky factorization of a band matrix.
 * @see plasma_omp_zgbtrf
 ******************************************************************************/
void plasma_pzpbtrf(plasma_enum_t uplo, plasma_desc_t A,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    int k, m, n;
    int tempkm, tempmm, tempmn;

    plasma_complex64_t zone  = (plasma_complex64_t)1.0;
    plasma_complex64_t mzone = (plasma_complex64_t)-1.0;

    // Check sequence status.
    if (sequence->status != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorSequence);
        return;
    }

    //=======================================
    // PlasmaLower
    //=======================================
    if (uplo == PlasmaLower) {
        for (k = 0; k < A.mt; k++)
        {
            tempkm = k == A.mt-1 ? A.m-k*A.mb : A.mb;
            core_omp_zpotrf(
                PlasmaLower, tempkm,
                A(k, k), BLKLDD_BAND(uplo, A, k, k),
                sequence, request, A.nb*k);
            for (m = k+1; m < imin(A.nt, k+A.klt); m++)
            {
                tempmm = m == A.mt-1 ? A.m-m*A.mb : A.mb;
                core_omp_ztrsm(
                    PlasmaRight, PlasmaLower,
                    PlasmaConjTrans, PlasmaNonUnit,
                    tempmm, tempkm,
                    zone, A(k, k), BLKLDD_BAND(uplo, A, k, k),
                          A(m, k), BLKLDD_BAND(uplo, A, m, k));
                core_omp_zherk(
                    PlasmaLower, PlasmaNoTrans,
                    tempmm, A.mb,
                    -1.0, A(m, k), BLKLDD_BAND(uplo, A, m, k),
                     1.0, A(m, m), BLKLDD_BAND(uplo, A, m, m));
                for (n = imax(k+1, m-A.klt); n < m; n++)
                {
                    tempmn = n == A.nt-1 ? A.n-n*A.nb : A.nb;
                    core_omp_zgemm(
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
            core_omp_zpotrf(
                PlasmaUpper, tempkm,
                A(k, k), BLKLDD_BAND(uplo, A, k, k),
                sequence, request, A.nb*k);
            for (m = k+1; m < imin(A.nt, k+A.kut); m++) {
                tempmm = m == A.nt-1 ? A.n-m*A.nb : A.nb;
                core_omp_ztrsm(
                    PlasmaLeft, PlasmaUpper,
                    PlasmaConjTrans, PlasmaNonUnit,
                    A.nb, tempmm,
                    zone, A(k, k), BLKLDD_BAND(uplo, A, k, k),
                          A(k, m), BLKLDD_BAND(uplo, A, k, m));
                core_omp_zherk(
                    PlasmaUpper, PlasmaConjTrans,
                    tempmm, A.mb,
                    -1.0, A(k, m), BLKLDD_BAND(uplo, A, k, m),
                     1.0, A(m, m), BLKLDD_BAND(uplo, A, m, m));
                for (n = imax(k+1, m-A.kut); n < m; n++) {
                    core_omp_zgemm(
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
