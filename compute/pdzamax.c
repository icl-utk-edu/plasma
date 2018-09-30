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

#define A(m, n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)

/******************************************************************************/
void plasma_pdzamax(plasma_enum_t colrow,
                    plasma_desc_t A, double *work, double *values,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    switch (colrow) {
    //===================
    // PlasmaColumnwise
    //===================
    case PlasmaColumnwise:
        for (int m = 0; m < A.mt; m++) {
            int mvam = plasma_tile_mview(A, m);
            int ldam = plasma_tile_mmain(A, m);
            for (int n = 0; n < A.nt; n++) {
                int nvan = plasma_tile_nview(A, n);
                plasma_core_omp_dzamax(PlasmaColumnwise,
                                mvam, nvan,
                                A(m, n), ldam,
                                &work[A.n*m+n*A.nb],
                                sequence, request);
            }
        }
        #pragma omp taskwait
        plasma_core_omp_damax(PlasmaRowwise,
                       A.n, A.mt,
                       work, A.n,
                       values,
                       sequence, request);
        break;
    //================
    // PlasmaRowwise
    //================
    case PlasmaRowwise:
        for (int m = 0; m < A.mt; m++) {
            int mvam = plasma_tile_mview(A, m);
            int ldam = plasma_tile_mmain(A, m);
            for (int n = 0; n < A.nt; n++) {
                int nvan = plasma_tile_nview(A, n);
                plasma_core_omp_dzamax(PlasmaRowwise,
                                mvam, nvan,
                                A(m, n), ldam,
                                &work[A.m*n+m*A.mb],
                                sequence, request);
            }
        }
        #pragma omp taskwait
        plasma_core_omp_damax(PlasmaRowwise,
                       A.m, A.nt,
                       work, A.m,
                       values,
                       sequence, request);
    }
}
