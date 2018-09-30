/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> c d s
 *
 **/

#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_types.h"
#include "plasma_workspace.h"
#include <plasma_core_blas.h>

#define A(m, n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)

/***************************************************************************//**
 * Parallel U*U^H or L^H*L operation.
 * @see plasma_omp_zlauum
 ******************************************************************************/
void plasma_pzlauum(plasma_enum_t uplo, plasma_desc_t A,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    //==============
    // PlasmaLower
    //==============
    if (uplo == PlasmaLower) {
        for (int k = 0; k < A.mt; k++) {
            int mvak = plasma_tile_mview(A, k);
            int nvak = plasma_tile_nview(A, k);
            int ldak = plasma_tile_mmain(A, k);
            for (int n = 0; n < k; n++) {
                int mvan = plasma_tile_mview(A, n);
                int nvan = plasma_tile_nview(A, n);
                int ldan = plasma_tile_mmain(A, n);
                plasma_core_omp_zherk(
                    uplo, PlasmaConjTrans,
                    imin(mvan, nvan), imin(mvak, nvan),
                    1.0, A(k, n), ldak,
                    1.0, A(n, n), ldan,
                    sequence, request);

                for (int m = n+1; m < k; m++) {
                    int mvam = plasma_tile_mview(A, m);
                    int ldam = plasma_tile_mmain(A, m);
                    plasma_core_omp_zgemm(
                        PlasmaConjTrans, PlasmaNoTrans,
                        mvam, nvan, mvak,
                        1.0, A(k, m), ldak,
                             A(k, n), ldak,
                        1.0, A(m, n), ldam,
                        sequence, request);
                }
            }
            for (int n = 0; n < k; n++) {
                int nvan = plasma_tile_nview(A, n);
                plasma_core_omp_ztrmm(
                    PlasmaLeft, uplo, PlasmaConjTrans, PlasmaNonUnit,
                    mvak, nvan,
                    1.0, A(k, k), ldak,
                         A(k, n), ldak,
                    sequence, request);
            }
            plasma_core_omp_zlauum(
                uplo, imin(mvak, nvak),
                A(k, k), ldak,
                sequence, request);
        }
    }
    //==============
    // PlasmaLower
    //==============
    else {
        for (int k = 0; k < A.mt; k++) {
            int mvak = plasma_tile_mview(A, k);
            int nvak = plasma_tile_nview(A, k);
            int ldak = plasma_tile_mmain(A, k);

            for (int m = 0; m < k; m++) {
                int mvam = plasma_tile_mview(A, m);
                int nvam = plasma_tile_nview(A, m);
                int ldam = plasma_tile_mmain(A, m);
                plasma_core_omp_zherk(
                    uplo, PlasmaNoTrans,
                    imin(mvam, nvam), imin(mvam, nvak),
                    1.0, A(m, k), ldam,
                    1.0, A(m, m), ldam,
                    sequence, request);

                for (int n = m+1; n < k; n++) {
                    int nvan = plasma_tile_nview(A, n);
                    int ldan = plasma_tile_mmain(A, n);
                    plasma_core_omp_zgemm(
                        PlasmaNoTrans, PlasmaConjTrans,
                        mvam, nvan, nvak,
                        1.0, A(m, k), ldam,
                             A(n, k), ldan,
                        1.0, A(m, n), ldam,
                        sequence, request);
                }
            }
            for (int m = 0; m < k; m++) {
                int mvam = plasma_tile_mview(A, m);
                int ldam = plasma_tile_mmain(A, m);
                plasma_core_omp_ztrmm(
                    PlasmaRight, uplo, PlasmaConjTrans, PlasmaNonUnit,
                    mvam, nvak,
                    1.0, A(k, k), ldak,
                         A(m, k), ldam,
                    sequence, request);
            }
            plasma_core_omp_zlauum(
                uplo, imin(mvak, nvak),
                A(k, k), ldak,
                sequence, request);
        }
    }
}
