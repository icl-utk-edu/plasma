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

#define A(m, n) ((plasma_complex64_t*)plasma_tile_addr(A, m, n))

/***************************************************************************//**
 *  Initializes the matrix A to beta on the diagonal and alpha on the
 *  offdiagonals. Applies alpha correctly for any shape of the submatrix
 *  described by A, but applies beta correctly only for submatrices aligned
 *  with the diagonal of the main matrix (A.i = A.j).
 **/
void plasma_pzlaset(plasma_enum_t uplo,
                    plasma_complex64_t alpha, plasma_complex64_t beta,
                    plasma_desc_t A,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    int i, j;
    int m, n;

    int lm1 = A.gm/A.mb;
    int ln1 = A.gn/A.nb;

    for (i = 0; i < A.mt; i++) {
        if (i == 0 && i == A.mt-1)
            m = A.m;
        else if (i == 0)
            m = A.mb-A.i%A.mb;
        else if (i == A.mt-1)
            m = (A.i+A.m+A.mb-1)%A.mb+1;
        else
            m = A.mb;

        for (j = 0; j < A.nt; j++) {
            if (j == 0 && j == A.nt-1)
                n = A.n;
            else if (j == 0)
                n = A.nb-A.j%A.nb;
            else if (j == A.nt-1)
                n = (A.j+A.n+A.nb-1)%A.nb+1;
            else
                n = A.nb;

            if (uplo == PlasmaGeneral ||
               (uplo == PlasmaLower && i >= j) ||
               (uplo == PlasmaUpper && i <= j))
                plasma_core_omp_zlaset(i == j ? uplo : PlasmaGeneral,
                                A.i/A.mb+i == lm1 ? A.gm-lm1*A.mb : A.mb,
                                A.j/A.nb+j == ln1 ? A.gn-ln1*A.nb : A.nb,
                                i == 0 ? A.i%A.mb : 0,
                                j == 0 ? A.j%A.nb : 0,
                                m,
                                n,
                                alpha,
                                i != j ? alpha : beta,
                                A(i, j));
        }
    }
}
