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
#include "plasma_types.h"
#include "plasma_internal.h"
#include "plasma_core_blas_z.h"

#define A(m, n) ((plasma_complex64_t*) plasma_tile_addr(A, m, n))
#define T(m, n) ((plasma_complex64_t*) plasma_tile_addr(T, m, n))
/***************************************************************************//**
 *  Parallel tile BAND Tridiagonal Reduction
 **/
void plasma_pzhe2hb(plasma_enum_t uplo,
                    plasma_desc_t A, plasma_desc_t T,
                    plasma_workspace_t work,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Check sequence status.
    if (sequence->status != PlasmaSuccess)
        return;

    // Case nb > n only 1 tile
    if (A.mt > A.m)
        return;

    // Set inner blocking from the plasma context
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    int ib = plasma->ib;

    if (uplo == PlasmaLower) {
        for (int k = 0; k < A.nt-1; ++k) {
            int nvak = plasma_tile_nview(A, k+1);
            int ldak = plasma_tile_mmain(A, k+1);
            plasma_core_omp_zgeqrt(
                nvak, A.nb, ib,
                A(k+1, k), ldak,
                T(k+1, k), T.mb,
                work,
                sequence, request);

            // LEFT and RIGHT on the symmetric diagonal block
            plasma_core_omp_zherfb(
                PlasmaLower,
                nvak, nvak, ib,
                A(k+1,   k), ldak,
                T(k+1,   k), T.mb,
                A(k+1, k+1), ldak,
                work,
                sequence, request);

            // RIGHT on the remaining tiles until the bottom
            for (int m = k+2; m < A.mt ; ++m) {
                int mvam = plasma_tile_mview(A, m);
                int ldam = plasma_tile_mmain(A, m);
                plasma_core_omp_zunmqr(
                    PlasmaRight, PlasmaNoTrans,
                    mvam, A.nb, nvak, ib,
                    A(k+1,   k), ldak,
                    T(k+1,   k), T.mb,
                    A(m  , k+1), ldam,
                    work,
                    sequence, request);
            }

            for (int m = k+2; m < A.mt; ++m) {
                int mvam = plasma_tile_mview(A, m);
                int ldam = plasma_tile_mmain(A, m);
                plasma_core_omp_ztsqrt(
                    mvam, A.nb, ib,
                    A(k+1, k), ldak,
                    A(m  , k), ldam,
                    T(m  , k), T.mb,
                    work,
                    sequence, request);

                // LEFT
                for (int i = k+2; i < m; ++i) {
                    int ldai = plasma_tile_mmain(A, i);
                    plasma_core_omp_ztsmqr_hetra1(
                        PlasmaLeft, Plasma_ConjTrans,
                        A.mb, A.nb, mvam, A.nb, A.nb, ib,
                        A(i, k+1), ldai,
                        A(m,   i), ldam,
                        A(m,   k), ldam,
                        T(m,   k), T.mb,
                        work,
                        sequence, request);
                }

                // RIGHT
                for (int j = m+1; j < A.mt ; ++j) {
                    int mvaj = plasma_tile_mview(A, j);
                    int ldaj = plasma_tile_mmain(A, j);
                    plasma_core_omp_ztsmqr(
                        PlasmaRight, PlasmaNoTrans,
                        mvaj, A.nb, mvaj, mvam, A.nb, ib,
                        A(j, k+1), ldaj,
                        A(j,   m), ldaj,
                        A(m,   k), ldam,
                        T(m,   k), T.mb,
                        work,
                        sequence, request);
                }

                // LEFT->RIGHT
                plasma_core_omp_ztsmqr_corner(
                    A.nb, A.nb, mvam, A.nb,
                    mvam, mvam, A.nb, ib,
                    A(k+1, k+1), ldak,
                    A(m  , k+1), ldam,
                    A(m  ,   m), ldam,
                    A(m  ,   k), ldam,
                    T(m  ,   k), T.mb,
                    work,
                    sequence, request);
           }
       }
    }
    else {
        for (int k = 0; k < A.nt-1; ++k) {
            int nvak = plasma_tile_nview(A, k+1);
            int ldak  = plasma_tile_mmain(A, k);
            int ldak1 = plasma_tile_mmain(A, k+1);
            plasma_core_omp_zgelqt(
                A.nb, nvak, ib,
                A(k, k+1), ldak,
                T(k, k+1), T.mb,
                work,
                sequence, request);

            // RIGHT and LEFT on the symmetric diagonal block
            plasma_core_omp_zherfb(
                PlasmaUpper,
                nvak, nvak, ib,
                A(k,   k+1), ldak,
                T(k,   k+1), T.mb,
                A(k+1, k+1), ldak1,
                work,
                sequence, request);

            // LEFT on the remaining tiles until the left side
            for (int n = k+2; n < A.nt ; ++n) {
                int nvan = plasma_tile_nview(A, n);
                plasma_core_omp_zunmlq(
                    PlasmaLeft, PlasmaNoTrans,
                    A.nb, nvan, nvak, ib,
                    A(k,   k+1), ldak,
                    T(k,   k+1), T.mb,
                    A(k+1,   n), ldak1,
                    work,
                    sequence, request);
            }

            for (int n = k+2; n < A.nt; ++n) {
                int nvan = plasma_tile_nview(A, n);
                int ldan = plasma_tile_nmain(A, n);
                plasma_core_omp_ztslqt(
                    A.nb, nvan, ib,
                    A(k, k+1), ldak,
                    A(k,   n), ldak,
                    T(k,   n), T.mb,
                    work,
                    sequence, request);

                // RIGHT
                for (int i = k+2; i < n; ++i) {
                    int ldai = plasma_tile_nmain(A, i);

                    plasma_core_omp_ztsmlq_hetra1(
                        PlasmaRight, Plasma_ConjTrans,
                        A.mb, A.nb, A.nb, nvan, A.nb, ib,
                        A(k+1, i), ldak1,
                        A(i,   n), ldai,
                        A(k,   n), ldak,
                        T(k,   n), T.mb,
                        work,
                        sequence, request);
                }

                // LEFT
                for (int j = n+1; j < A.nt ; ++j) {
                    int nvaj = plasma_tile_nview(A, j);
                    plasma_core_omp_ztsmlq(
                        PlasmaLeft, PlasmaNoTrans,
                        A.nb, nvaj, nvan, nvaj, A.nb, ib,
                        A(k+1, j), ldak1,
                        A(n,   j), ldan,
                        A(k,   n), ldak,
                        T(k,   n), T.mb,
                        work,
                        sequence, request);
                }

                // RIGHT->LEFT
                plasma_core_omp_ztsmlq_corner(
                    A.nb, A.nb, A.nb, nvan,
                    nvan, nvan, A.nb, ib,
                    A(k+1, k+1), ldak1,
                    A(k+1,   n), ldak1,
                    A(n  ,   n), ldan,
                    A(k  ,   n), ldak,
                    T(k  ,   n), T.mb,
                    work,
                    sequence, request);
           }
       }
    }
}
