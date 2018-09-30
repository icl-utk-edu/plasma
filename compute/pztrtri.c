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
 * Parallel tile triangular inversion.
 * @see plasma_omp_ztrtri
 ******************************************************************************/
void plasma_pztrtri(plasma_enum_t uplo, plasma_enum_t diag,
                    plasma_desc_t A,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    //==============
    // PlasmaLower
    //==============
    if (uplo == PlasmaLower) {
        for (int k = 0; k < A.nt; k++) {
            int mvak = plasma_tile_mview(A, k);
            int nvak = plasma_tile_nview(A, k);
            int ldak = plasma_tile_mmain(A, k);
            for (int m =  k+1; m < A.mt; m++) {
                int mvam = plasma_tile_mview(A, m);
                int ldam = plasma_tile_mmain(A, m);
                plasma_core_omp_ztrsm(
                    PlasmaRight, uplo, PlasmaNoTrans, diag,
                    mvam, nvak,
                    -1.0, A(k, k), ldak,
                          A(m, k), ldam,
                    sequence, request);
            }
            for (int m = k+1; m < A.mt; m++) {
                int mvam = plasma_tile_mview(A, m);
                int ldam = plasma_tile_mmain(A, m);
                for (int n = 0; n < k; n++) {
                    int nvan = plasma_tile_nview(A, n);
                    plasma_core_omp_zgemm(
                        PlasmaNoTrans, PlasmaNoTrans,
                        mvam, nvan, imin(nvak, mvak),
                        1.0, A(m, k), ldam,
                             A(k, n), ldak,
                        1.0, A(m, n), ldam,
                        sequence, request);
                }
            }
            for (int n = 0; n < k; n++) {
                int nvan = plasma_tile_nview(A, n);
                plasma_core_omp_ztrsm(
                    PlasmaLeft, uplo, PlasmaNoTrans, diag,
                    mvak, nvan,
                    1.0, A(k, k), ldak,
                         A(k, n), ldak,
                    sequence, request);
            }
            plasma_core_omp_ztrtri(
                uplo, diag,
                nvak,
                A(k, k), ldak,
                A.nb*k,
                sequence, request);
        }
    }
    //==============
    // PlasmaUpper
    //==============
    else {
        for (int k = 0; k < A.mt; k++) {
            int mvak = plasma_tile_mview(A, k);
            int nvak = plasma_tile_nview(A, k);
            int ldak = plasma_tile_mmain(A, k);
            for (int n = k+1; n < A.nt; n++) {
                int nvan = plasma_tile_nview(A, n);
                plasma_core_omp_ztrsm(
                    PlasmaLeft, uplo, PlasmaNoTrans, diag,
                    mvak, nvan,
                    -1.0, A(k, k), ldak,
                          A(k, n), ldak,
                    sequence, request);
            }
            for (int m = 0; m < k; m++) {
                int mvam = plasma_tile_mview(A, m);
                int ldam = plasma_tile_mmain(A, m);
                for (int n = k+1; n < A.nt; n++) {
                    int nvan = plasma_tile_nview(A, n);
                    plasma_core_omp_zgemm(
                        PlasmaNoTrans, PlasmaNoTrans,
                        mvam, nvan, imin(nvak, mvak),
                        1.0, A(m, k), ldam,
                             A(k, n), ldak,
                        1.0, A(m, n), ldam,
                        sequence, request);
                }
                plasma_core_omp_ztrsm(
                    PlasmaRight, uplo, PlasmaNoTrans, diag,
                    mvam, nvak,
                    1.0, A(k, k), ldak,
                         A(m, k), ldam,
                    sequence, request);
            }
            plasma_core_omp_ztrtri(
                uplo, diag,
                mvak,
                A(k, k), ldak,
                A.nb*k,
                sequence, request);
        }
    }
}
