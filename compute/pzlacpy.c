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
    // Check sequence status
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
            int X = m == A.mt-1 ? A.m-m*A.mb : A.mb;
            int ldam = plasma_tile_mmain(A, m);
            int ldbm = plasma_tile_mmain(B, m);
            if (m < A.nt) {
                int Y = m == A.nt-1 ? A.n-m*A.nb : A.nb;
                core_omp_zlacpy(
                    PlasmaUpper,
                    X, Y, A.mb,
                    A(m, m), ldam,
                    B(m, m), ldbm);
            }
            for (int n = m+1; n < A.nt; n++) {
                int Y = n == A.nt-1 ? A.n-n*A.nb : A.nb;
                core_omp_zlacpy(
                    PlasmaGeneral,
                    X, Y, A.mb,
                    A(m, n), ldam,
                    B(m, n), ldbm);
            }
        }
        break;
    //=============
    // PlasmaLower
    //=============
    case PlasmaLower:
        for (int m = 0; m < A.mt; m++) {
            int X = m == A.mt-1 ? A.m-m*A.mb : A.mb;
            int ldam = plasma_tile_mmain(A, m);
            int ldbm = plasma_tile_mmain(B, m);
            if (m < A.nt) {
                int Y = m == A.nt-1 ? A.n-m*A.nb : A.nb;
                core_omp_zlacpy(
                    PlasmaLower,
                    X, Y, A.mb,
                    A(m, m), ldam,
                    B(m, m), ldbm);
            }
            for (int n = 0; n < imin(m, A.nt); n++) {
                int Y = n == A.nt-1 ? A.n-n*A.nb : A.nb;
                core_omp_zlacpy(
                    PlasmaGeneral,
                    X, Y, A.mb,
                    A(m, n), ldam,
                    B(m, n), ldbm);
            }
        }
        break;
    //===============
    // PlasmaGeneral
    //===============
    case PlasmaGeneral:
    default:
        for (int m = 0; m < A.mt; m++) {
            int X = m == A.mt-1 ? A.m-m*A.mb : A.mb;
            int ldam = plasma_tile_mmain(A, m);
            int ldbm = plasma_tile_mmain(B, m);
            for (int n = 0; n < A.nt; n++) {
                int Y = n == A.nt-1 ? A.n-n*A.nb : A.nb;
                core_omp_zlacpy(
                    PlasmaGeneral,
                    X, Y, A.mb,
                    A(m, n), ldam,
                    B(m, n), ldbm);
            }
        }
    }
}
