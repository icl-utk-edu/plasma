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
#include <plasma_core_blas.h>
#include "core_lapack.h"

#define A(m, n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)

/***************************************************************************//**
 *  Parallel tile calculation of max, one, infinity or Frobenius matrix norm
 *  for a general band matrix.
 ******************************************************************************/
void plasma_pzlangb(plasma_enum_t norm,
                    plasma_desc_t A, double *work, double *value,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    double stub;
    int wcnt = 0;
    int ldwork, klt, kut;
    double *workspace, *scale, *sumsq;

    switch (norm) {
    //================
    // PlasmaMaxNorm
    //================
    case PlasmaMaxNorm:
        wcnt = 0;
        for (int n = 0; n < A.nt; n++ ) {
            int nvan = plasma_tile_nview(A, n);
            int m_start = (imax(0, n*A.nb-A.ku)) / A.nb;
            int m_end = (imin(A.m-1, (n+1)*A.nb+A.kl-1)) / A.nb;
            for (int m = m_start; m <= m_end; m++ ) {
                int ldam = plasma_tile_mmain_band(A, m, n);
                int mvam = plasma_tile_mview(A, m);
                plasma_core_omp_zlange(PlasmaMaxNorm,
                                mvam, nvan,
                                A(m, n), ldam,
                                &stub, &work[wcnt],
                                sequence, request);
                wcnt++;
            }
        }

        #pragma omp taskwait
        plasma_core_omp_dlange(PlasmaMaxNorm,
                        1, wcnt,
                        work, 1,
                        &stub, value,
                        sequence, request);
        break;
    //================
    // PlasmaOneNorm
    //================
    case PlasmaOneNorm:
        // # of tiles in upper band (not including diagonal)
        kut  = (A.ku+A.nb-1)/A.nb;
        // # of tiles in lower band (not including diagonal)
        klt  = (A.kl+A.nb-1)/A.nb;
        ldwork = kut+klt+1;
        for (int n = 0; n < A.nt; n++ ) {
            int nvan = plasma_tile_nview(A, n);
            int m_start = (imax(0, n*A.nb-A.ku)) / A.nb;
            int m_end = (imin(A.m-1, (n+1)*A.nb+A.kl-1)) / A.nb;
            for (int m = m_start; m <= m_end; m++ ) {
                int ldam = plasma_tile_mmain_band(A, m, n);
                int mvam = plasma_tile_mview(A, m);
                plasma_core_omp_zlange_aux(PlasmaOneNorm,
                                    mvam, nvan,
                                    A(m,n), ldam,
                                    &work[(m-m_start)*A.n+n*A.nb],
                                    sequence, request);
            }
        }
        #pragma omp taskwait
        workspace = &work[A.n*ldwork];
        plasma_core_omp_dlange(PlasmaInfNorm,
                        A.n, ldwork,
                        work, A.n,
                        workspace, value,
                        sequence, request);
        break;
    //================
    // PlasmaInfNorm
    //================
    case PlasmaInfNorm:
        ldwork = A.mb*A.mt;
        for (int n = 0; n < A.nt; n++ ) {
            int nvan = plasma_tile_nview(A, n);
            int m_start = (imax(0, n*A.nb-A.ku)) / A.nb;
            int m_end = (imin(A.m-1, (n+1)*A.nb+A.kl-1)) / A.nb;
            for (int m = m_start; m <= m_end; m++ ) {
                int ldam = plasma_tile_mmain_band(A, m, n);
                int mvam = plasma_tile_mview(A, m);
                plasma_core_omp_zlange_aux(PlasmaInfNorm,
                                    mvam, nvan,
                                    A(m,n), ldam,
                                    &work[m*A.mb+n*ldwork],
                                    sequence, request);
            }
        }
        #pragma omp taskwait
        //nwork = A.nt;
        workspace = &work[ldwork*A.nt];
        plasma_core_omp_dlange(PlasmaInfNorm,
                        ldwork, A.nt,
                        work, ldwork,
                        workspace, value,
                        sequence, request);
        break;
    //======================
    // PlasmaFrobeniusNorm
    //======================
    case PlasmaFrobeniusNorm:
        kut  = (A.ku+A.nb-1)/A.nb; // # of tiles in upper band (not including diagonal)
        klt  = (A.kl+A.nb-1)/A.nb;    // # of tiles in lower band (not including diagonal)
        ldwork = kut+klt+1;
        scale = work;
        sumsq = &work[ldwork*A.nt];
        for (int n = 0; n < A.nt; n++ ) {
            int nvan = plasma_tile_nview(A, n);
            int m_start = (imax(0, n*A.nb-A.ku)) / A.nb;
            int m_end = (imin(A.m-1, (n+1)*A.nb+A.kl-1)) / A.nb;

            for (int m = m_start; m <= m_end; m++ ) {
                int ldam = plasma_tile_mmain_band(A, m, n);
                int mvam = plasma_tile_mview(A, m);
                plasma_core_omp_zgessq(mvam, nvan,
                                A(m,n), ldam,
                                &scale[n*ldwork+m-m_start],
                                &sumsq[n*ldwork+m-m_start],
                                sequence, request);
            }
        }
        #pragma omp taskwait
        plasma_core_omp_dgessq_aux(ldwork*A.nt, scale, sumsq,
                            value, sequence, request);
        break;
    default:
        assert(0);
    }
}
