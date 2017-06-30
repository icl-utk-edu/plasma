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
 * @ingroup plasma_getri
 *
 *  Computes the inverse of a matrix A using the LU factorization computed
 *  by plasma_zgetrf.
 *
 *******************************************************************************
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in,out] pA
 *          On entry, the LU factors computed by plasma_zgetrf.
 *          On exit, the inverse of A, overwriting the factors.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,n).
 *
 * @param[in] ipiv
 *          The pivot indices computed by plasma_zgetrf.
 *
 *******************************************************************************
 *
 * @retval PLASMA_SUCCESS successful exit
 * @retval < 0 if -i, the i-th argument had an illegal value
 * @retval > 0 if i, the (i,i) element of the factor U or L is
 *         zero, and the inverse could not be computed.
 *
 *******************************************************************************
 *
 * @sa plasma_cgetri
 * @sa plasma_dgetri
 * @sa plasma_sgetri
 *
 ******************************************************************************/
int plasma_zgetri(int n, plasma_complex64_t *pA, int lda, int *ipiv)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    // Check input arguments.
    if (n < 0) {
        plasma_error("illegal value of n");
        return -1;
    }
    if (lda < imax(1, n)) {
        plasma_error("illegal value of lda");
        return -3;
    }
    // quick return
    if (imax(n, 0) == 0)
        return PlasmaSuccess;

    // Tune parameters.
    if (plasma->tuning)
        plasma_tune_getrf(plasma, PlasmaComplexDouble, n, n);

    // Set tiling parameters.
    int nb = plasma->nb;

    // Create tile matrix.
    plasma_desc_t A;
    plasma_desc_t W;
    int retval;
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        n, n, 0, 0, n, n, &A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        return retval;
    }
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        n, nb, 0, 0, n, nb, &W);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        plasma_desc_destroy(&A);
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
        plasma_omp_zge2desc(pA, lda, A, &sequence, &request);

        // Perform computation.
        plasma_omp_zgetri(A, ipiv, W, &sequence, &request);

        // Translate back to LAPACK layout.
        plasma_omp_zdesc2ge(A, pA, lda, &sequence, &request);
    }
    // implicit synchronization

    // Free matrices in tile layout.
    plasma_desc_destroy(&W);
    plasma_desc_destroy(&A);

    // Return status.
    int status = sequence.status;
    return status;
}

/***************************************************************************//**
 *
 * Computes the inverse of a matrix A using the LU factorization.
 * Non-blocking tile version of plasma_zgbsv().
 * Operates on matrices stored by tiles.
 * All matrices are passed through descriptors.
 * All dimensions are taken from the descriptors.
 * Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] A
 *          On entry, the LU factors computed by plasma_zgetrf.
 *          On exit, the inverse of A, overwriting the factors.
 *
 * @param[in] ipiv
 *          The pivot indices computed by plasma_zgetrf.
 *
 * @param[out] W
 *          Workspace of dimension (n, nb)
 *
 * @param[in] sequence
 *          Identifies the sequence of function calls that this call belongs to
 *          (for completion checks and exception handling purposes).  Check
 *          the sequence->status for errors.
 *
 * @param[out] request
 *          Identifies this function call (for exception handling purposes).
 *
 * @retval void
 *          Errors are returned by setting sequence->status and
 *          request->status to error values.  The sequence->status and
 *          request->status should never be set to PlasmaSuccess (the
 *          initial values) since another async call may be setting a
 *          failure value at the same time.
 *
 *******************************************************************************
 *
 * @sa plasma_zgetri
 * @sa plasma_omp_zgetri
 * @sa plasma_omp_cgetri
 * @sa plasma_omp_dgetri
 * @sa plasma_omp_sgetri
 *
 ******************************************************************************/
void plasma_omp_zgetri(plasma_desc_t A, int *ipiv, plasma_desc_t W,
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
    if (plasma_desc_check(A) != PlasmaSuccess) {
        plasma_error("invalid A");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (plasma_desc_check(W) != PlasmaSuccess) {
        plasma_error("invalid W");
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
    if (A.n == 0)
        return;

    // Invert triangular part.
    plasma_pztrtri(PlasmaUpper, PlasmaNonUnit, A, sequence, request);

    // Compute product of inverse of the upper and lower triangles.
    plasma_pzgetri_aux(A, W, sequence, request);

    // Apply pivot.
    plasma_pzgeswp(PlasmaColumnwise, A, ipiv, -1, sequence, request);
}
