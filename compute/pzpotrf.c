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

#define A(m, n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)

/***************************************************************************//**
 *  Parallel tile Cholesky factorization.
 * @see plasma_omp_zpotrf
 ******************************************************************************/
void plasma_pzpotrf(plasma_enum_t uplo, plasma_desc_t A,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    int k, m, n;
    int ldak, ldam, ldan;
    int tempkm, tempmm;

    plasma_complex64_t zone  = (plasma_complex64_t) 1.0;
    plasma_complex64_t mzone = (plasma_complex64_t)-1.0;

    // Check sequence status.
    if (sequence->status != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorSequence);
        return;
    }

    //==============
    // PlasmaLower
    //==============
    if (uplo == PlasmaLower) {
        for (k = 0; k < A.mt; k++) {
            tempkm = plasma_tile_mdim(A, k);
            ldak = plasma_tile_mdim(A, k);
            core_omp_zpotrf(
                PlasmaLower, tempkm,
                A(k, k), ldak,
                sequence, request, A.nb*k);

            for (m = k+1; m < A.mt; m++) {
                tempmm = plasma_tile_mdim(A, m);
                ldam = plasma_tile_mdim(A, m);
                core_omp_ztrsm(
                    PlasmaRight, PlasmaLower,
                    PlasmaConjTrans, PlasmaNonUnit,
                    tempmm, A.mb,
                    zone, A(k, k), ldak,
                          A(m, k), ldam);
            }
            for (m = k+1; m < A.mt; m++) {
                tempmm = plasma_tile_mdim(A, m);
                ldam = plasma_tile_mdim(A, m);
                core_omp_zherk(
                    PlasmaLower, PlasmaNoTrans,
                    tempmm, A.mb,
                    -1.0, A(m, k), ldam,
                     1.0, A(m, m), ldam);

                for (n = k+1; n < m; n++) {
                    ldan = plasma_tile_mdim(A, n);
                    core_omp_zgemm(
                        PlasmaNoTrans, PlasmaConjTrans,
                        tempmm, A.mb, A.mb,
                        mzone, A(m, k), ldam,
                               A(n, k), ldan,
                        zone,  A(m, n), ldam);
                }
            }
        }
    }
    //==============
    // PlasmaUpper
    //==============
    else {
        for (k = 0; k < A.nt; k++) {
            tempkm = plasma_tile_ndim(A, k);
            ldak = plasma_tile_mdim(A, k);
            core_omp_zpotrf(
                PlasmaUpper, tempkm,
                A(k, k), ldak,
                sequence, request, A.nb*k);

            for (m = k+1; m < A.nt; m++) {
                tempmm = plasma_tile_ndim(A, m);
                core_omp_ztrsm(
                    PlasmaLeft, PlasmaUpper,
                    PlasmaConjTrans, PlasmaNonUnit,
                    A.nb, tempmm,
                    zone, A(k, k), ldak,
                          A(k, m), ldak);
            }
            for (m = k+1; m < A.nt; m++) {
                tempmm = plasma_tile_ndim(A, m);
                ldam = plasma_tile_mdim(A, m);
                core_omp_zherk(
                    PlasmaUpper, PlasmaConjTrans,
                    tempmm, A.mb,
                    -1.0, A(k, m), ldak,
                     1.0, A(m, m), ldam);

                for (n = k+1; n < m; n++) {
                    ldan = plasma_tile_mdim(A, n);
                    core_omp_zgemm(
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
