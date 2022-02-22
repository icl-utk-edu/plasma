/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> s d  c
 *
 **/

#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_types.h"
#include "plasma_workspace.h"
#include "plasma_core_blas.h"

#define A(m, n) ((plasma_complex64_t*) plasma_tile_addr(A, m, n))
#define pA_band(m_, n_)  &(pA_band[ (m_) + lda_band*((n_)*nb )])
/**********************************************************************//**
 * Parallel copy of a band matrix from full nxn tile storage to compact band
 * storage (lda_bandxn).
 * NOTE : this function transform the
 * Lower/Upper Tile band matrix to LOWER Band storage matrix. For
 * Lower it copy it directly. For Upper it conjtransposed during the
 * copy.
 * */
void plasma_pzgecpy_tile2lapack_band(plasma_enum_t uplo, plasma_desc_t A,
                   plasma_complex64_t *pA_band, int lda_band,
                   plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    /*=============================================
     * NOTE :
     * this function transforms the Lower/Upper Tile
     * band matrix to LOWER Band storage matrix.
     * For Lower it copy it directly.
     * For Upper it conjtransposed during the copy.
     *=============================================*/
    
    int nb = A.mb;
    int ldx = lda_band - 1;
    
    // copy Lower to Lower
    if ( uplo == PlasmaLower ) {
        for (int j = 0; j < imin(A.mt, A.nt); j++) {
            int mvaj = plasma_tile_mview(A, j);
            int nvaj = plasma_tile_nview(A, j);
            int ldaj = plasma_tile_mmain(A, j);
            
            plasma_core_omp_zlacpy(PlasmaLower, PlasmaNoTrans,
                            mvaj, nvaj,
                            A(j, j), ldaj, pA_band(0, j), ldx,
                            sequence, request);
            
            if (j< imin(A.mt, A.nt)-1 ) {
                mvaj = plasma_tile_mview(A, j+1);
                ldaj = plasma_tile_mmain(A, j+1);
                plasma_core_omp_zlacpy(PlasmaUpper, PlasmaNoTrans,
                                mvaj, nvaj,
                                A(j+1, j), ldaj, pA_band(nb, j), ldx,
                                sequence, request);
            }
        }
    }
    //Mawussi: This comment is misleading : I
    // think it is Upper to Upper
    // conjtranspose Upper when copying it to Lower
    else if ( uplo == PlasmaUpper ) {
        for (int j = 0; j < imin(A.mt, A.nt); j++) {
            int mvaj = plasma_tile_mview(A, j);
            int nvaj = plasma_tile_nview(A, j);
            int ldaj = plasma_tile_mmain(A, j);
            
            plasma_core_omp_zlacpy(PlasmaUpper, PlasmaNoTrans,
                            mvaj, nvaj,
                            A(j, j), ldaj, pA_band(nb, j), ldx,
                            sequence, request);
            
            if (j<imin(A.mt, A.nt)-1) {
                nvaj = plasma_tile_nview(A, j+1);
                plasma_core_omp_zlacpy(PlasmaLower, PlasmaNoTrans,
                                mvaj, nvaj,
                                A(j, j+1), ldaj, pA_band(0, j+1), ldx,
                                sequence, request);
            }
        }
    }
}

#undef pA_band
#undef A
