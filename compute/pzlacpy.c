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
#include "plasma_internal.h"
#include "plasma_types.h"
#include "core_blas.h"

#define A(m, n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)
#define B(m, n) (plasma_complex64_t*)plasma_tile_addr(B, m, n)

/***************************************************************************//**
 * Parallel tile matrix copy.
 * @see plasma_omp_zlacpy
 ******************************************************************************/
void plasma_pzlacpy(plasma_enum_t uplo, plasma_desc_t A, plasma_desc_t B,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Check sequence status.
    if (sequence->status != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorSequence);
        return;
    }

    switch (uplo) {
    //=============
    // PlasmaUpper
    //=============
    case PlasmaUpper:
        for (int m = 0; m < A.mt; m++) {
            int am  = plasma_tile_mview(A, m);
            int lda = imax(1, plasma_tile_mmain(A, m));
            int ldb = imax(1, plasma_tile_mmain(B, m));
            if (m < A.nt) {
                int an = plasma_tile_nview(A, m);
                core_omp_zlacpy(
                    PlasmaUpper,
                    am, an,
                    A(m, m), lda,
                    B(m, m), ldb,
                    sequence, request);
            }
            for (int n = m+1; n < A.nt; n++) {
                int an = plasma_tile_nview(A, n);
                core_omp_zlacpy(
                    PlasmaGeneral,
                    am, an,
                    A(m, n), lda,
                    B(m, n), ldb,
                    sequence, request);
            }
        }
        break;
    //=============
    // PlasmaLower
    //=============
    case PlasmaLower:
        for (int m = 0; m < A.mt; m++) {
            int am  = plasma_tile_mview(A, m);
            int lda = plasma_tile_mmain(A, m);
            int ldb = plasma_tile_mmain(B, m);
            if (m < A.nt) {
                int an = plasma_tile_nview(A, m);
                core_omp_zlacpy(
                    PlasmaLower,
                    am, an,
                    A(m, m), lda,
                    B(m, m), ldb,
                    sequence, request);
            }
            for (int n = 0; n < imin(m, A.nt); n++) {
                int an = plasma_tile_nview(A, n);
                core_omp_zlacpy(
                    PlasmaGeneral,
                    am, an,
                    A(m, n), lda,
                    B(m, n), ldb,
                    sequence, request);
            }
        }
        break;
    //===============
    // PlasmaGeneral
    //===============
    case PlasmaGeneral:
    default:
        for (int m = 0; m < A.mt; m++) {
            int am  = plasma_tile_mview(A, m);
            int lda = plasma_tile_mmain(A, m);
            int ldb = plasma_tile_mmain(B, m);
            for (int n = 0; n < A.nt; n++) {
                int an = plasma_tile_nview(A, n);
                core_omp_zlacpy(
                    PlasmaGeneral,
                    am, an,
                    A(m, n), lda,
                    B(m, n), ldb,
                    sequence, request);
            }
        }
    }
}
