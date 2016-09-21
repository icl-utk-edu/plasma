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

#define A(m,n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)
#define B(m,n) (plasma_complex64_t*)plasma_tile_addr(B, m, n)

/***************************************************************************//**
 * Parallel tile matrix-matrix addition.
 * @see plasma_omp_ztradd
 ******************************************************************************/
void plasma_pztradd(plasma_enum_t uplo, plasma_enum_t transA,
                    plasma_complex64_t alpha,  plasma_desc_t A,
                    plasma_complex64_t beta,   plasma_desc_t B,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    int tempmm, tempnn, tempmn, tempnm;
    int m, n;
    int ldam, ldan, ldbm, ldbn;

    // Check sequence status
    if (sequence->status != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorSequence);
        return;
    }

    switch (uplo) {
    case PlasmaLower:
        if (transA == PlasmaNoTrans) {
            for (n = 0; n < imin(B.mt,B.nt); n++) {
                tempnm = plasma_tile_mdim(B, n);
                tempnn = plasma_tile_ndim(B, n);
                ldan = plasma_tile_mdim(A, n);
                ldbn = plasma_tile_mdim(B, n);
                core_omp_ztradd(
                    uplo, transA,
                    tempnm, tempnn,
                    alpha, A(n, n), ldan,
                    beta,  B(n, n), ldbn);

                for (m = n+1; m < B.mt; m++) {
                    tempmm = plasma_tile_mdim(B, m);
                    ldam = plasma_tile_mdim(A, m);
                    ldbm = plasma_tile_mdim(B, m);
                    core_omp_zgeadd(
                        transA, 
                        tempmm, tempnn,
                        alpha, A(m, n), ldam,
                        beta,  B(m, n), ldbm);
                }
            }
        }
        else {
            for (n = 0; n < imin(B.mt,B.nt); n++) {
                tempnm = plasma_tile_mdim(B, n);
                tempnn = plasma_tile_ndim(B, n);
                ldan = plasma_tile_mdim(A, n);
                ldbn = plasma_tile_mdim(B, n);
                core_omp_ztradd(
                    uplo, transA,
                    tempnm, tempnn,
                    alpha, A(n, n), ldan,
                    beta,  B(n, n), ldbn);

                for (m = n+1; m < B.mt; m++) {
                    tempmm = plasma_tile_mdim(B, m);
                    ldbm = plasma_tile_mdim(B, m);
                    core_omp_zgeadd(
                        transA,
                        tempmm, tempnn,
                        alpha, A(n, m), ldan,
                        beta,  B(m, n), ldbm);
                }
            }
        }
        break;
    case PlasmaUpper:
        if (transA == PlasmaNoTrans) {
            for (m = 0; m < imin(B.mt,B.nt); m++) {
                tempmm = plasma_tile_mdim(B, m);
                tempmn = plasma_tile_ndim(B, m);
                ldam = plasma_tile_mdim(A, m);
                ldbm = plasma_tile_mdim(B, m);
                core_omp_ztradd(
                    uplo, transA,
                    tempmm, tempmn,
                    alpha, A(m, m), ldam,
                    beta,  B(m, m), ldbm);

                for (n = m+1; n < B.nt; n++) {
                    tempnn = plasma_tile_ndim(B, n);
                    core_omp_zgeadd(
                        transA,
                        tempmm, tempnn,
                        alpha, A(m, n), ldam,
                        beta,  B(m, n), ldbm);
                }
            }
        }
        else {
            for (m = 0; m < imin(B.mt,B.nt); m++) {
                tempmm = plasma_tile_mdim(B, m);
                tempmn = plasma_tile_ndim(B, m);
                ldam = plasma_tile_mdim(A, m);
                ldbm = plasma_tile_mdim(B, m);
                core_omp_ztradd(
                    uplo, transA,
                    tempmm, tempmn,
                    alpha, A(m, m), ldam,
                    beta,  B(m, m), ldbm);

                for (n = m+1; n < B.nt; n++) {
                    tempnn = plasma_tile_ndim(B, n);
                    ldan = plasma_tile_mdim(A, n);
                    core_omp_zgeadd(
                        transA,
                        tempmm, tempnn,
                        alpha, A(n, m), ldan,
                        beta,  B(m, n), ldbm);
                }
            }
        }
        break;
    case PlasmaGeneral:
    default:
        if (transA == PlasmaNoTrans) {
            for (m = 0; m < B.mt; m++) {
                tempmm = m == B.mt-1 ? B.m-B.mb*m : B.nb;
                ldam = plasma_tile_mdim(A, m);
                ldbm = plasma_tile_mdim(B, m);
                for (n = 0; n < B.nt; n++) {
                    tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                    core_omp_zgeadd(
                        transA, tempmm, tempnn,
                        alpha, A(m, n), ldam,
                        beta,  B(m, n), ldbm);
                }
            }
        }
        else {
            for (m = 0; m < B.mt; m++) {
                tempmm = plasma_tile_mdim(B, m);
                ldam = plasma_tile_mdim(A, m);
                ldbm = plasma_tile_mdim(B, m);
                for (n = 0; n < B.nt; n++) {
                    tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
                    ldan = plasma_tile_mdim(A, n);
                    core_omp_zgeadd(
                        transA, tempmm, tempnn,
                        alpha, A(n, m), ldan,
                        beta,  B(m, n), ldbm);
                }
            }
        }
    }
}
