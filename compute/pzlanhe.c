/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> c
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
 *  Parallel tile calculation of max, one, infinity or Frobenius matrix norm
 *  for a Hermitian matrix.
 ******************************************************************************/
void plasma_pzlanhe(plasma_enum_t norm, plasma_enum_t uplo,
                    plasma_desc_t A, double *work, double *value,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    switch (norm) {
    double stub;
    double *workspace;
    double *scale;
    double *sumsq;
    //================
    // PlasmaMaxNorm
    //================
    case PlasmaMaxNorm:
        for (int m = 0; m < A.mt; m++) {
            int mvam = plasma_tile_mview(A, m);
            int ldam = plasma_tile_mmain(A, m);
            if (uplo == PlasmaLower) {
                for (int n = 0; n < m; n++) {
                    int nvan = plasma_tile_nview(A, n);
                    plasma_core_omp_zlange(PlasmaMaxNorm,
                                    mvam, nvan,
                                    A(m, n), ldam,
                                    &stub, &work[A.mt*n+m],
                                    sequence, request);
                }
            }
            else { // PlasmaUpper
                for (int n = m+1; n < A.nt; n++) {
                    int nvan = plasma_tile_nview(A, n);
                    plasma_core_omp_zlange(PlasmaMaxNorm,
                                    mvam, nvan,
                                    A(m, n), ldam,
                                    &stub, &work[A.mt*n+m],
                                    sequence, request);
                }
            }
            plasma_core_omp_zlanhe(PlasmaMaxNorm, uplo,
                            mvam,
                            A(m, m), ldam,
                            &stub, &work[A.mt*m+m],
                            sequence, request);
        }
        #pragma omp taskwait
        plasma_core_omp_dlansy(PlasmaMaxNorm, uplo,
                        A.nt,
                        work, A.mt,
                        &stub, value,
                        sequence, request);
        break;
    //================
    // PlasmaOneNorm
    //================
    case PlasmaOneNorm:
    case PlasmaInfNorm:
        for (int m = 0; m < A.mt; m++) {
            int mvam = plasma_tile_mview(A, m);
            int ldam = plasma_tile_mmain(A, m);
            if (uplo == PlasmaLower) {
                for (int n = 0; n < m; n++) {
                    int nvan = plasma_tile_nview(A, n);
                    plasma_core_omp_zlange_aux(PlasmaOneNorm,
                                        mvam, nvan,
                                        A(m, n), ldam,
                                        &work[A.n*m+n*A.nb],
                                        sequence, request);
                    plasma_core_omp_zlange_aux(PlasmaInfNorm,
                                        mvam, nvan,
                                        A(m, n), ldam,
                                        &work[A.n*n+m*A.nb],
                                        sequence, request);
                }
            }
            else { // PlasmaUpper
                for (int n = m+1; n < A.nt; n++) {
                    int nvan = plasma_tile_nview(A, n);
                    plasma_core_omp_zlange_aux(PlasmaOneNorm,
                                        mvam, nvan,
                                        A(m, n), ldam,
                                        &work[A.n*m+n*A.nb],
                                        sequence, request);
                    plasma_core_omp_zlange_aux(PlasmaInfNorm,
                                        mvam, nvan,
                                        A(m, n), ldam,
                                        &work[A.n*n+m*A.nb],
                                        sequence, request);
                }
            }
            plasma_core_omp_zlanhe_aux(PlasmaOneNorm, uplo,
                                mvam,
                                A(m, m), ldam,
                                &work[A.n*m+m*A.nb],
                                sequence, request);
        }
        #pragma omp taskwait
        workspace = work + A.mt*A.n;
        plasma_core_omp_dlange(PlasmaInfNorm,
                        A.n, A.mt,
                        work, A.n,
                        workspace, value,
                        sequence, request);
        break;
    //======================
    // PlasmaFrobeniusNorm
    //======================
    case PlasmaFrobeniusNorm:
        scale = work;
        sumsq = work + A.mt*A.nt;
        for (int m = 0; m < A.mt; m++) {
            int mvam = plasma_tile_mview(A, m);
            int ldam = plasma_tile_mmain(A, m);
            if (uplo == PlasmaLower) {
                for (int n = 0; n < m; n++) {
                    int nvan = plasma_tile_nview(A, n);
                    plasma_core_omp_zgessq(mvam, nvan,
                                    A(m, n), ldam,
                                    &scale[A.mt*n+m], &sumsq[A.mt*n+m],
                                    sequence, request);
                }
            }
            else { // PlasmaUpper
                for (int n = m+1; n < A.nt; n++) {
                    int nvan = plasma_tile_nview(A, n);
                    plasma_core_omp_zgessq(mvam, nvan,
                                    A(m, n), ldam,
                                    &scale[A.mt*m+n], &sumsq[A.mt*m+n],
                                    sequence, request);
                }
            }
            plasma_core_omp_zhessq(uplo,
                            mvam,
                            A(m, m), ldam,
                            &scale[A.mt*m+m], &sumsq[A.mt*m+m],
                            sequence, request);
        }
        #pragma omp taskwait
        plasma_core_omp_dsyssq_aux(A.mt, A.nt,
                            scale, sumsq,
                            value,
                            sequence, request);
        break;
    }
}
