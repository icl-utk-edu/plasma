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

#define A(i_, j_) (plasma_complex64_t*) plasma_tile_addr(A, i_, j_)
#define Aband(i_, j_) &(Aband[ (i_)*nb + lda_band*((j_)*nb) ])

/***************************************************************************//**
 * Parallel copy of a Hermitian band matrix, with bandwidth of nb (1 tile),
 * from full n-by-n tile storage to compact band storage (lda_band-by-n).
 * As this function is internal and the space is the
 * same for either Lower or Upper, it ALWAYS converts to lower band and
 * then the bulge chasing will always work with a lower band matrix.
 **/
void plasma_pzhecpy_tile2lapack_band(
    plasma_enum_t uplo,
    plasma_desc_t A,
    plasma_complex64_t *Aband, int lda_band,
    plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    int nb = A.mb;
    int ldx = lda_band - 1;
    int minmn = imin( A.mt, A.nt );

    //=============================================
    // NOTE :
    // this function transform the Lower/Upper Tile
    // band matrix to LOWER Band storage matrix.
    // For Lower it copies it directly.
    // For Upper it is conjugate-transposed during the copy.
    //=============================================
    if (uplo == PlasmaLower) {
        // copy Lower to Lower
        for (int j = 0; j < minmn; ++j) {
            int mvaj = plasma_tile_mview(A, j);
            int nvaj = plasma_tile_nview(A, j);
            int ldaj = plasma_tile_mmain(A, j);

            plasma_core_omp_zlacpy(
                PlasmaLower, PlasmaNoTrans,
                mvaj, nvaj,
                A(j, j), ldaj, Aband(0, j), ldx,
                sequence, request);

            if (j < minmn - 1) {
                mvaj = plasma_tile_mview(A, j+1);
                ldaj = plasma_tile_mmain(A, j+1);
                plasma_core_omp_zlacpy(
                    PlasmaUpper, PlasmaNoTrans,
                    mvaj, nvaj,
                    A(j+1, j), ldaj, Aband(1, j), ldx,
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
                A(j, j), ldaj, Aband(0, j), ldx,
                sequence, request);

            if (j < minmn - 1) {
                nvaj = plasma_tile_nview(A, j+1);
                plasma_core_omp_zlacpy(
                    PlasmaLower, PlasmaConjTrans,
                    mvaj, nvaj,
                    A(j, j+1), ldaj, Aband(1, j), ldx,
                    sequence, request);
            }
        }
    }
}

#undef Aband
#undef A
