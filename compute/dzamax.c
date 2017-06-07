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
#include "plasma_types.h"

/******************************************************************************/
int plasma_dzamax(plasma_enum_t colrow,
                  int m, int n,
                  plasma_complex64_t *pA, int lda, double *values)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    // Check input arguments.
    if ((colrow != PlasmaColumnwise) && (colrow != PlasmaRowwise)) {
        plasma_error("illegal value of colrow");
        return -1;
    }
    if (m < 0) {
        plasma_error("illegal value of m");
        return -2;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -3;
    }
    if (lda < imax(1, m)) {
        plasma_error("illegal value of lda");
        return -5;
    }

    // quick return
    if (imin(n, m) == 0)
        return PlasmaSuccess;

    // Set tiling parameters.
    int nb = plasma->nb;

    // Create tile matrices.
    plasma_desc_t A;
    int retval;
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        m, n, 0, 0, m, n, &A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        return retval;
    }

    // Allocate workspace.
    double *work;
    switch (colrow) {
    case PlasmaColumnwise:
        work = (double*)malloc((size_t)A.mt*A.n*sizeof(double));
        break;
    case PlasmaRowwise:
        work = (double*)malloc((size_t)A.m*A.nt*sizeof(double));
        break;
    }
    if (work == NULL) {
        plasma_error("malloc() failed");
        return PlasmaErrorOutOfMemory;
    }

    // Initialize sequence.
    plasma_sequence_t sequence;
    retval = plasma_sequence_init(&sequence);

    // Initialize request.
    plasma_request_t request;
    retval = plasma_request_init(&request);

    // asynchronous block
    #pragma omp parallel
    #pragma omp master
    {
        // Translate to tile layout.
        plasma_omp_zge2desc(pA, lda, A, &sequence, &request);

        // Call tile async function.
        plasma_omp_dzamax(colrow, A, work, values, &sequence, &request);
    }
    // implicit synchronization

    free(work);

    // Free matrix in tile layout.
    plasma_desc_destroy(&A);

    // Return status.
    int status = sequence.status;
    return status;
}

/******************************************************************************/
void plasma_omp_dzamax(plasma_enum_t colrow, plasma_desc_t A,
                       double *work, double *values,
                       plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }

    // Check input arguments.
    if ((colrow != PlasmaColumnwise) && (colrow != PlasmaRowwise)) {
        plasma_error("illegal value of colrow");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (plasma_desc_check(A) != PlasmaSuccess) {
        plasma_error("invalid descriptor A");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (sequence == NULL) {
        plasma_error("NULL sequence");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (request == NULL) {
        plasma_error("NULL request");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }

    // quick return
    if (imin(A.m, A.n) == 0)
        return;

    // Call the parallel function.
    plasma_pdzamax(colrow, A, work, values, sequence, request);
}
