/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions mixed zc -> ds
 *
 **/

#include "plasma.h"
#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_types.h"

/***************************************************************************//**
 *
 * @ingroup plasma_lag2
 *
 *  Converts m-by-n matrix As from complex single to complex double precision.
 *
 *******************************************************************************
 *
 * @param[in] m
 *          The number of rows of the matrix As. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix As. n >= 0.
 *
 * @param[in] pAs
 *          The ldas-by-n matrix As in single complex precision.
 *
 * @param[in] ldas
 *          The leading dimension of the array As. ldas >= max(1,m).
 *
 * @param[out] pA
 *          On exit, the lda-by-n matrix A in double complex precision.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,m).
 *
 *******************************************************************************
 *
 * @retval PlasmaSuccess successful exit
 *
 *******************************************************************************
 *
 * @sa plasma_omp_clag2z
 * @sa plasma_zlag2c
 * @sa plasma_dlag2s
 * @sa plasma_slag2d
 *
 ******************************************************************************/
int plasma_clag2z(int m, int n,
                  plasma_complex32_t *pAs, int ldas,
                  plasma_complex64_t *pA,  int lda)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    // Check input arguments.
    if (m < 0) {
        plasma_error("illegal value of m");
        return -1;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -2;
    }
    if (ldas < imax(1, m)) {
        plasma_error("illegal value of ldas");
        return -4;
    }
    if (lda < imax(1, m)) {
        plasma_error("illegal value of lda");
        return -6;
    }

    // quick return
    if (imin(n, m) == 0)
      return PlasmaSuccess;

    // Set tiling parameters.
    int nb = plasma->nb;

    // Create tile matrices.
    plasma_desc_t As;
    plasma_desc_t A;
    int retval;
    retval = plasma_desc_general_create(PlasmaComplexFloat, nb, nb,
                                        m, n, 0, 0, m, n, &As);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        return retval;
    }
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        m, n, 0, 0, m, n, &A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        plasma_desc_destroy(&As);
        return retval;
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
        plasma_omp_cge2desc(pAs, ldas, As, &sequence, &request);
        plasma_omp_zge2desc(pA,  lda,  A,  &sequence, &request);

        // Call tile async function.
        plasma_omp_clag2z(As, A, &sequence, &request);

        // Translate back to LAPACK layout.
        plasma_omp_cdesc2ge(As, pAs, ldas, &sequence, &request);
        plasma_omp_zdesc2ge(A,  pA,  lda,  &sequence, &request);
    }
    // implicit synchronization

    // Free matrices in tile layout.
    plasma_desc_destroy(&As);
    plasma_desc_destroy(&A);

    // Return status.
    int status = sequence.status;
    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_lag2
 *
 *  Converts m-by-n matrix A from single complex to double complex precision.
 *  Non-blocking tile version of plasma_clag2z(). May return before the
 *  computation is finished. Operates on matrices stored by tiles. All matrices
 *  are passed through descriptors. All dimensions are taken from the
 *  descriptors. Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] As
 *          Descriptor of matrix As.
 *
 * @param[out] A
 *          Descriptor of matrix A.
 *
 * @param[in] sequence
 *          Identifies the sequence of function calls that this call belongs to
 *          (for completion checks and exception handling purposes). Check the
 *          sequence->status for errors.
 *
 * @param[out] request
 *          Identifies this function call (for exception handling purposes).
 *
 * @retval void
 *          Errors are returned by setting sequence->status and
 *          request->status to error values. The sequence->status and
 *          request->status should never be set to PlasmaSuccess (the
 *          initial values) since another async call may be setting a
 *          failure value at the same time.
 *
 *******************************************************************************
 *
 * @sa plasma_clag2z
 * @sa plasma_omp_zlag2c
 * @sa plasma_omp_dlag2s
 * @sa plasma_omp_slag2d
 *
 ******************************************************************************/
void plasma_omp_clag2z(plasma_desc_t As, plasma_desc_t A,
                       plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Get PLASMA context
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }

    // Check input arguments.
    if (plasma_desc_check(As) != PlasmaSuccess) {
        plasma_error("invalid As");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (plasma_desc_check(A) != PlasmaSuccess) {
        plasma_error("invalid A");
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
    if (imin(As.m, As.n) == 0)
        return;

    // Call the parallel function.
    plasma_pclag2z(As, A, sequence, request);
}
