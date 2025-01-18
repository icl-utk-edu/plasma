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
#include "plasma_core_blas.h"

#define A(m, n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)
#define AB(m_, n_) &(AB[(m_) + ldab*((n_)*nb) ])

/***************************************************************************//**
 * Parallel copy of a band matrix from full n x n tile storage to band
 * storage (n x ldab). As this function is internal and the space is the
 * same for either Lower or Upper, it ALWAYS converts to lower band and
 * then the bulge chasing will always work with a lower band matrix.
 **/
void plasma_pzhecpy_tile2lapack_band(
    plasma_enum_t uplo,
    plasma_desc_t A,
    plasma_complex64_t *AB, int ldab,
    plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    int nb = A.mb;

    //=============================================
    // NOTE :
    // this function transform the Lower/Upper Tile
    // band matrix to LOWER Band storage matrix.
    // For Lower it copies it directly.
    // For Upper it is conjugate-transposed during the copy.
    //=============================================
    int ldx = ldab-1;
    int minmn = imin(A.mt, A.nt);
    if (uplo == PlasmaLower) {
        // copy Lower to Lower
        for (int j = 0; j < minmn; ++j) {
            int mvaj = plasma_tile_mview(A, j);
            int nvaj = plasma_tile_nview(A, j);
            int ldaj = plasma_tile_mmain(A, j);

            plasma_core_omp_zlacpy(
                PlasmaLower, PlasmaNoTrans,
                mvaj, nvaj,
                A(j, j), ldaj, AB(0, j), ldx,
                sequence, request);

            if (j < minmn - 1) {
                mvaj = plasma_tile_mview(A, j+1);
                ldaj = plasma_tile_mmain(A, j+1);

                plasma_core_omp_zlacpy(
                    PlasmaUpper, PlasmaNoTrans,
                    mvaj, nvaj,
                    A(j+1, j), ldaj, AB(nb, j), ldx,
                    sequence, request);
            }
        }
    }
    else if (uplo == PlasmaUpper) {
        // conj-transpose Upper when copying it to Lower
        for (int j = 0; j < minmn; ++j) {
            int mvaj = plasma_tile_mview(A, j);
            int nvaj = plasma_tile_nview(A, j);
            int ldaj = plasma_tile_mmain(A, j);

            plasma_core_omp_zlacpy(
                PlasmaUpper, PlasmaConjTrans,
                mvaj, nvaj,
                A(j, j), ldaj, AB(0, j), ldx,
                sequence, request);

            if (j < minmn - 1) {
                nvaj = plasma_tile_nview(A, j+1);

                plasma_core_omp_zlacpy(
                    PlasmaLower, PlasmaConjTrans,
                    mvaj, nvaj,
                    A(j, j+1), ldaj, AB(nb, j), ldx,
                    sequence, request);
            }
        }
    }
}
#undef AB
#undef A
