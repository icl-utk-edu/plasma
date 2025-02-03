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

#define A(i_, j_) ((plasma_complex64_t*) plasma_tile_addr(A, i_, j_))
#define T(i_, j_) ((plasma_complex64_t*) plasma_tile_addr(T, i_, j_))

/***************************************************************************//**
 *  Parallel tile Hermitian full to Hermitian band reduction
 **/
void plasma_pzhe2hb(
    plasma_enum_t uplo,
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

            // LEFT and RIGHT on the Hermitian diagonal block
            plasma_core_omp_zherfb(
                PlasmaLower,
                nvak, nvak, ib,
                A(k+1,   k), ldak,
                T(k+1,   k), T.mb,
                A(k+1, k+1), ldak,
                work,
                sequence, request);

            // RIGHT on the remaining tiles until the bottom
            for (int i = k+2; i < A.mt; ++i) {
                int mvai = plasma_tile_mview(A, i);
                int ldai = plasma_tile_mmain(A, i);
                plasma_core_omp_zunmqr(
                    PlasmaRight, PlasmaNoTrans,
                    mvai, A.nb, nvak, ib,
                    A(k+1,   k), ldak,
                    T(k+1,   k), T.mb,
                    A(i,   k+1), ldai,
                    work,
                    sequence, request);
            }

            for (int i = k+2; i < A.mt; ++i) {
                int mvai = plasma_tile_mview(A, i);
                int ldai = plasma_tile_mmain(A, i);
                plasma_core_omp_ztsqrt(
                    mvai, A.nb, ib,
                    A(k+1, k), ldak,
                    A(i,   k), ldai,
                    T(i,   k), T.mb,
                    work,
                    sequence, request);

                // LEFT
                for (int j = k+2; j < i; ++j) {
                    int ldaj = plasma_tile_mmain(A, j);
                    plasma_core_omp_ztsmqr_conj_trans(
                        PlasmaLeft, Plasma_ConjTrans,
                        A.mb, A.nb, mvai, A.nb, A.nb, ib,
                        A(j, k+1), ldaj,  // conj-trans of A(k+1, j)
                        A(i,   j), ldai,
                        A(i,   k), ldai,
                        T(i,   k), T.mb,
                        work,
                        sequence, request);
                }

                // RIGHT
                for (int j = i+1; j < A.mt; ++j) {
                    int mvaj = plasma_tile_mview(A, j);
                    int ldaj = plasma_tile_mmain(A, j);
                    plasma_core_omp_ztsmqr(
                        PlasmaRight, PlasmaNoTrans,
                        mvaj, A.nb, mvaj, mvai, A.nb, ib,
                        A(j, k+1), ldaj,
                        A(j,   i), ldaj,
                        A(i,   k), ldai,
                        T(i,   k), T.mb,
                        work,
                        sequence, request);
                }

                // LEFT->RIGHT
                plasma_core_omp_ztsmqr_corner(
                    A.nb, A.nb, mvai, A.nb,
                    mvai, mvai, A.nb, ib,
                    A(k+1, k+1), ldak,
                    A(i,   k+1), ldai,
                    A(i,     i), ldai,
                    A(i,     k), ldai,
                    T(i,     k), T.mb,
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

            // RIGHT and LEFT on the Hermitian diagonal block
            plasma_core_omp_zherfb(
                PlasmaUpper,
                nvak, nvak, ib,
                A(k,   k+1), ldak,
                T(k,   k+1), T.mb,
                A(k+1, k+1), ldak1,
                work,
                sequence, request);

            // LEFT on the remaining tiles until the left side
            for (int j = k+2; j < A.nt; ++j) {
                int nvaj = plasma_tile_nview(A, j);
                plasma_core_omp_zunmlq(
                    PlasmaLeft, PlasmaNoTrans,
                    A.nb, nvaj, nvak, ib,
                    A(k,   k+1), ldak,
                    T(k,   k+1), T.mb,
                    A(k+1,   j), ldak1,
                    work,
                    sequence, request);
            }

            for (int j = k+2; j < A.nt; ++j) {
                int nvaj = plasma_tile_nview(A, j);
                int ldaj = plasma_tile_nmain(A, j);

                plasma_core_omp_ztslqt(
                    A.nb, nvaj, ib,
                    A(k, k+1), ldak,
                    A(k,   j), ldak,
                    T(k,   j), T.mb,
                    work,
                    sequence, request);

                // RIGHT
                for (int i = k+2; i < n; ++i) {
                    int ldai = plasma_tile_nmain(A, i);

                    plasma_core_omp_ztsmlq_conj_trans(
                        PlasmaRight, Plasma_ConjTrans,
                        A.mb, A.nb, A.nb, nvaj, A.nb, ib,
                        A(k+1, i), ldak1,  // conj-trans of A(i, k+1)
                        A(i,   j), ldai,
                        A(k,   j), ldak,
                        T(k,   j), T.mb,
                        work,
                        sequence, request);
                }

                // LEFT
                for (int i = j+1; i < A.nt; ++i) {
                    int nvai = plasma_tile_nview(A, i);
                    plasma_core_omp_ztsmlq(
                        PlasmaLeft, PlasmaNoTrans,
                        A.nb, nvai, nvaj, nvai, A.nb, ib,
                        A(k+1, i), ldak1,
                        A(j,   i), ldaj,
                        A(k,   j), ldak,
                        T(k,   j), T.mb,
                        work,
                        sequence, request);
                }

                // RIGHT->LEFT
                plasma_core_omp_ztsmlq_2sided(
                    A.nb, A.nb, A.nb, nvaj,
                    nvaj, nvaj, A.nb, ib,
                    A(k+1, k+1), ldak1,
                    A(k+1,   j), ldak1,
                    A(j,     j), ldaj,
                    A(k,     j), ldak,
                    T(k,     j), T.mb,
                    work,
                    sequence, request);
            }
        }
    }
}
