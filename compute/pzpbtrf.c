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
#include "core_blas.h"

#define A(m, n) ((plasma_complex64_t*)plasma_tile_addr(A, m, n))

/***************************************************************************//**
 *  Parallel tile Cholesky factorization of a band matrix.
 * @see plasma_omp_zgbtrf
 ******************************************************************************/
void plasma_pzpbtrf(plasma_enum_t uplo, plasma_desc_t A,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Check sequence status.
    if (sequence->status != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorSequence);
        return;
    }

    if (uplo == PlasmaLower) {
        //==============
        // PlasmaLower
        //==============
        for (int k = 0; k < A.mt; k++) {
            int mvak  = plasma_tile_mview(A, k);
            int ldakk = BLKLDD_BAND(uplo, A, k, k);
            core_omp_zpotrf(
                PlasmaLower, mvak,
                A(k, k), ldakk,
                A.nb*k,
                sequence, request);
            for (int m = k+1; m < imin(A.nt, k+A.klt); m++) {
                int mvam  = plasma_tile_mview(A, m);
                int ldamk = BLKLDD_BAND(uplo, A, m, k);
                int ldamm = BLKLDD_BAND(uplo, A, m, m);
                core_omp_ztrsm(
                    PlasmaRight, PlasmaLower,
                    PlasmaConjTrans, PlasmaNonUnit,
                    mvam, mvak,
                    1.0, A(k, k), ldakk,
                         A(m, k), ldamk,
                    sequence, request);
                core_omp_zherk(
                    PlasmaLower, PlasmaNoTrans,
                    mvam, A.mb,
                    -1.0, A(m, k), ldamk,
                     1.0, A(m, m), ldamm,
                    sequence, request);
                for (int n = imax(k+1, m-A.klt); n < m; n++) {
                    int nvan  = plasma_tile_nview(A, n);
                    int ldank = BLKLDD_BAND(uplo, A, n, k);
                    int ldamn = BLKLDD_BAND(uplo, A, m, n);
                    core_omp_zgemm(
                        PlasmaNoTrans, PlasmaConjTrans,
                        mvam, nvan, mvak,
                        -1.0, A(m, k), ldamk,
                              A(n, k), ldank,
                         1.0, A(m, n), ldamn,
                        sequence, request);
                }
            }
        }
    }
    else {
        //==============
        // PlasmaUpper
        //==============
        for (int k = 0; k < A.nt; k++) {
            int mvak  = plasma_tile_mview(A, k);
            int ldakk = BLKLDD_BAND(uplo, A, k, k);
            core_omp_zpotrf(
                PlasmaUpper, mvak,
                A(k, k), ldakk,
                A.nb*k,
                sequence, request);
            for (int m = k+1; m < imin(A.nt, k+A.kut); m++) {
                int mvam  = plasma_tile_mview(A, m);
                int ldakm = BLKLDD_BAND(uplo, A, k, m);
                int ldamm = BLKLDD_BAND(uplo, A, m, m);
                core_omp_ztrsm(
                    PlasmaLeft, PlasmaUpper,
                    PlasmaConjTrans, PlasmaNonUnit,
                    A.nb, mvam,
                    1.0, A(k, k), ldakk,
                         A(k, m), ldakm,
                    sequence, request);
                core_omp_zherk(
                    PlasmaUpper, PlasmaConjTrans,
                    mvam, A.mb,
                    -1.0, A(k, m), ldakm,
                     1.0, A(m, m), ldamm,
                    sequence, request);
                for (int n = imax(k+1, m-A.kut); n < m; n++) {
                    int ldakn = BLKLDD_BAND(uplo, A, k, n);
                    int ldanm = BLKLDD_BAND(uplo, A, n, m);
                    core_omp_zgemm(
                        PlasmaConjTrans, PlasmaNoTrans,
                        A.mb, mvam, A.mb,
                        -1.0, A(k, n), ldakn,
                              A(k, m), ldakm,
                         1.0, A(n, m), ldanm,
                        sequence, request);
                }
            }
        }
    }
}
