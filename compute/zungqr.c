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
 * @ingroup plasma_ungqr
 *
 *  Generates an m-by-n matrix Q with orthonormal columns, which
 *  is defined as the first n columns of a product of the elementary reflectors
 *  returned by plasma_zgeqrf.
 *
 *******************************************************************************
 *
 * @param[in] m
 *          The number of rows of the matrix Q. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix Q. m >= n >= 0.
 *
 * @param[in] k
 *          The number of columns of elementary tile reflectors whose product
 *          defines the matrix Q.
 *          n >= k >= 0.
 *
 * @param[in] pA
 *          Details of the QR factorization of the original matrix A as returned
 *          by plasma_zgeqrf, where the k first columns are the reflectors.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,m).
 *
 * @param[in] T
 *          Auxiliary factorization data, computed by plasma_zgeqrf.
 *
 * @param[out] pQ
 *          On exit, pointer to the m-by-n matrix Q.
 *
 * @param[in] ldq
 *          The leading dimension of the array Q. ldq >= max(1,m).
 *
 *******************************************************************************
 *
 * @retval PlasmaSuccess successful exit
 * @retval < 0 if -i, the i-th argument had an illegal value
 *
 *******************************************************************************
 *
 * @sa plasma_omp_zungqr
 * @sa plasma_cungqr
 * @sa plasma_dorgqr
 * @sa plasma_sorgqr
 * @sa plasma_zgeqrf
 *
 ******************************************************************************/
int plasma_zungqr(int m, int n, int k,
                  plasma_complex64_t *pA, int lda,
                  plasma_desc_t T,
                  plasma_complex64_t *pQ, int ldq)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    // Check input arguments.
    if (m < 0) {
        plasma_error("illegal value of m");
        return -1;
    }
    if (n < 0 || n > m) {
        plasma_error("illegal value of n");
        return -2;
    }
    if (k < 0 || k > n) {
        plasma_error("illegal value of k");
        return -3;
    }
    if (lda < imax(1, m)) {
        plasma_error("illegal value of lda");
        return -5;
    }
    if (ldq < imax(1, m)) {
        plasma_error("illegal value of ldq");
        return -8;
    }

    // quick return
    if (n <= 0)
        return PlasmaSuccess;

    // Tune parameters.
    if (plasma->tuning)
        plasma_tune_geqrf(plasma, PlasmaComplexDouble, m, n);

    // Set tiling parameters.
    int ib = plasma->ib;
    int nb = plasma->nb;

    // Create tile matrices.
    plasma_desc_t A;
    plasma_desc_t Q;
    int retval;
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        m, n, 0, 0, m, k, &A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        return retval;
    }
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        m, n, 0, 0, m, k, &Q);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        plasma_desc_destroy(&A);
        return retval;
    }

    // Allocate workspace.
    plasma_workspace_t work;
    size_t lwork = ib*nb;  // unmqr: work
    retval = plasma_workspace_create(&work, lwork, PlasmaComplexDouble);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_workspace_create() failed");
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
        plasma_omp_zge2desc(pQ, ldq, Q, &sequence, &request);

        // Call the tile async function.
        plasma_omp_zungqr(A, T, Q, work, &sequence, &request);

        // Translate Q back to LAPACK layout.
        plasma_omp_zdesc2ge(Q, pQ, ldq, &sequence, &request);
    }
    // implicit synchronization

    plasma_workspace_destroy(&work);

    // Free matrices in tile layout.
    plasma_desc_destroy(&A);
    plasma_desc_destroy(&Q);

    // Return status.
    int status = sequence.status;
    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_ungqr
 *
 *  Non-blocking tile version of plasma_zungqr().
 *  May return before the computation is finished.
 *  Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] A
 *          Descriptor of matrix A.
 *          A is stored in the tile layout.
 *
 * @param[in] T
 *          Descriptor of matrix T.
 *          Auxiliary factorization data, computed by plasma_zgeqrf.
 *
 * @param[out] Q
 *          Descriptor of matrix Q. On exit, matrix Q stored in the tile layout.
 *
 * @param[in] work
 *          Workspace for the auxiliary arrays needed by some coreblas kernels.
 *          For multiplication by Q contains preallocated space for work
 *          arrays. Allocated by the plasma_workspace_create function.
 *
 * @param[in] sequence
 *          Identifies the sequence of function calls that this call belongs to
 *          (for completion checks and exception handling purposes).
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
 * @sa plasma_zungqr
 * @sa plasma_omp_cungqr
 * @sa plasma_omp_dorgqr
 * @sa plasma_omp_sorgqr
 * @sa plasma_omp_zgeqrf
 *
 ******************************************************************************/
void plasma_omp_zungqr(plasma_desc_t A, plasma_desc_t T, plasma_desc_t Q,
                       plasma_workspace_t work,
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
    if (plasma_desc_check(T) != PlasmaSuccess) {
        plasma_error("invalid T");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (plasma_desc_check(Q) != PlasmaSuccess) {
        plasma_error("invalid Q");
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
    if (Q.n <= 0)
        return;

    // Set Q to identity.
    plasma_pzlaset(PlasmaGeneral, 0.0, 1.0, Q, sequence, request);

    // Construct Q.
    if (plasma->householder_mode == PlasmaTreeHouseholder) {
        plasma_pzungqr_tree(A, T, Q, work, sequence, request);
    }
    else {
        plasma_pzungqr(A, T, Q, work, sequence, request);
    }
}
