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

#include "plasma.h"
#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_tuning.h"
#include "plasma_types.h"
#include "plasma_workspace.h"

/***************************************************************************//**
 *
 ******************************************************************************/
int plasma_zgetrs(int n, int nrhs,
                  plasma_complex64_t *pA, int lda, int *ipiv,
                  plasma_complex64_t *pB, int ldb)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    if (n < 0) {
        plasma_error("illegal value of n");
        return -1;
    }
    if (nrhs < 0) {
        plasma_error("illegal value of nrhs");
        return -2;
    }
    if (lda < imax(1, n)) {
        plasma_error("illegal value of lda");
        return -4;
    }
    if (ldb < imax(1, n)) {
        plasma_error("illegal value of ldb");
        return -7;
    }

    // quick return
    if (imin(n, nrhs) == 0)
        return PlasmaSuccess;

    // Tune parameters.
    if (plasma->tuning)
        plasma_tune_trsm(plasma, PlasmaComplexDouble, n, n);

    // Set tiling parameters.
    int nb = plasma->nb;

    // Initialize barrier.
    plasma_barrier_init(&plasma->barrier);

    // Create tile matrix.
    plasma_desc_t A;
    plasma_desc_t B;
    int retval;
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        n, n, 0, 0, n, n, &A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        return retval;
    }
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        n, nrhs, 0, 0, n, nrhs, &B);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        return retval;
    }

    // Initialize sequence.
    plasma_sequence_t sequence;
    retval = plasma_sequence_init(&sequence);

    // Initialize request.
    plasma_request_t request;
    retval = plasma_request_init(&request);

    #pragma omp parallel
    #pragma omp master
    {
        // Translate to tile layout.
        plasma_omp_zge2desc(pA, lda, A, &sequence, &request);
        plasma_omp_zge2desc(pB, ldb, B, &sequence, &request);

        // Call the tile async function.
        plasma_omp_zgetrs(A, ipiv, B, &sequence, &request);

        // Translate back to LAPACK layout.
        plasma_omp_zdesc2ge(B, pB, ldb, &sequence, &request);
    }

    // Free matrix A in tile layout.
    plasma_desc_destroy(&A);
    plasma_desc_destroy(&B);

    // Return status.
    int status = sequence.status;
    return status;
}

/***************************************************************************//**
 *
 ******************************************************************************/
void plasma_omp_zgetrs(plasma_desc_t A, int *ipiv,
                       plasma_desc_t B,
                       plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }

    // Check input arguments.
    if (plasma_desc_check(A) != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        plasma_error("invalid A");
        return;
    }
    if (plasma_desc_check(B) != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        plasma_error("invalid B");
        return;
    }
    if (sequence == NULL) {
        plasma_fatal_error("NULL sequence");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (request == NULL) {
        plasma_fatal_error("NULL request");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }

    // quick return
    if (A.n == 0 || B.n == 0)
        return;

    // Call the parallel functions.
    plasma_pzgeswp(PlasmaRowwise, B, ipiv, 1, sequence, request);

    plasma_pztrsm(PlasmaLeft, PlasmaLower, PlasmaNoTrans, PlasmaUnit,
                  1.0, A,
                       B,
                  sequence, request);

    plasma_pztrsm(PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit,
                  1.0, A,
                       B,
                  sequence, request);
}
