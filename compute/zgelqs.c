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
 * @ingroup plasma_gelqs
 *
 *  Computes a minimum-norm solution min | A*X - B | using the
 *  LQ factorization A = L*Q computed by plasma_zgelqf.
 *
 *******************************************************************************
 *
 * @param[in] m
 *          The number of rows of the matrix A. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix A. n >= m >= 0.
 *
 * @param[in] nrhs
 *          The number of columns of B. nrhs >= 0.
 *
 * @param[in] pA
 *          Details of the LQ factorization of the original matrix A as returned
 *          by plasma_zgelqf.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= m.
 *
 * @param[in] T
 *          Auxiliary factorization data, computed by plasma_zgelqf.
 *
 * @param[in,out] pB
 *          On entry, pointer to the m-by-nrhs right hand side matrix B.
 *          On exit, the n-by-nrhs solution matrix X.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= n.
 *
 *******************************************************************************
 *
 * @retval PlasmaSuccess successful exit
 * @retval < 0 if -i, the i-th argument had an illegal value
 *
 *******************************************************************************
 *
 * @sa plasma_omp_zgelqs
 * @sa plasma_cgelqs
 * @sa plasma_dgelqs
 * @sa plasma_sgelqs
 * @sa plasma_zgelqf
 *
 ******************************************************************************/
int plasma_zgelqs(int m, int n, int nrhs,
                  plasma_complex64_t *pA, int lda,
                  plasma_desc_t T,
                  plasma_complex64_t *pB, int ldb)
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
    if (n < 0 || m > n) {
        plasma_error("illegal value of n");
        return -2;
    }
    if (nrhs < 0) {
        plasma_error("illegal value of nrhs");
        return -3;
    }
    if (lda < imax(1, m)) {
        plasma_error("illegal value of lda");
        return -5;
    }
    if (ldb < imax(1, imax(1, n))) {
        plasma_error("illegal value of ldb");
        return -8;
    }

    // quick return
    if (m == 0 || n == 0 || nrhs == 0)
        return PlasmaSuccess;

    // Tune parameters.
    if (plasma->tuning)
        plasma_tune_gelqf(plasma, PlasmaComplexDouble, m, n);

    // Set tiling parameters.
    int ib = plasma->ib;
    int nb = plasma->nb;

    // Create tile matrices.
    plasma_desc_t A;
    plasma_desc_t B;
    int retval;
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        m, n, 0, 0, m, n, &A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        return retval;
    }
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        n, nrhs, 0, 0, n, nrhs, &B);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        plasma_desc_destroy(&A);
        return retval;
    }

    // Allocate workspace.
    plasma_workspace_t work;
    size_t lwork = ib*nb;  // unmlq: work
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
        plasma_omp_zge2desc(pB, ldb, B, &sequence, &request);

        // Call the tile async function.
        plasma_omp_zgelqs(A, T, B, work, &sequence, &request);

        // Translate back to LAPACK layout.
        plasma_omp_zdesc2ge(B, pB, ldb, &sequence, &request);
    }
    // implicit synchronization

    plasma_workspace_destroy(&work);

    // Free matrices in tile layout.
    plasma_desc_destroy(&A);
    plasma_desc_destroy(&B);

    // Return status.
    int status = sequence.status;
    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_gelqs
 *
 *  Computes a minimum-norm solution using previously computed LQ factorization.
 *  Non-blocking tile version of plasma_zgelqs().
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
 *          Auxiliary factorization data, computed by plasma_zgelqf.
 *
 * @param[in,out] B
 *          Descriptor of matrix B.
 *          On entry, right-hand side matrix B in the tile layout.
 *          On exit, solution matrix X in the tile layout.
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
 * @sa plasma_zgelqs
 * @sa plasma_omp_cgelqs
 * @sa plasma_omp_dgelqs
 * @sa plasma_omp_sgelqs
 * @sa plasma_omp_zgelqf
 *
 ******************************************************************************/
void plasma_omp_zgelqs(plasma_desc_t A, plasma_desc_t T,
                       plasma_desc_t B, plasma_workspace_t work,
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
        plasma_error("invalid descriptor A");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (plasma_desc_check(T) != PlasmaSuccess) {
        plasma_error("invalid descriptor T");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (plasma_desc_check(B) != PlasmaSuccess) {
        plasma_error("invalid descriptor B");
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
    if (A.m == 0 || A.n == 0 || B.n == 0)
        return;

    // Zero the trailing block of the right-hand-side matrix.
    // B has less rows than X.
    plasma_pzlaset(PlasmaGeneral, 0.0, 0.0,
                   plasma_desc_view(B, A.m, 0, A.n - A.m, B.n),
                   sequence, request);

    // Solve L * Y = B.
    plasma_pztrsm(PlasmaLeft, PlasmaLower, PlasmaNoTrans, PlasmaNonUnit,
                  1.0, plasma_desc_view(A, 0, 0, A.m, A.m),
                       plasma_desc_view(B, 0, 0, A.m, B.n),
                  sequence, request);

    // Find X = Q^H * Y.
    if (plasma->householder_mode == PlasmaTreeHouseholder) {
        plasma_pzunmlq_tree(PlasmaLeft, Plasma_ConjTrans,
                            A, T, B, work,
                            sequence, request);
    }
    else {
        plasma_pzunmlq(PlasmaLeft, Plasma_ConjTrans,
                       A, T, B, work,
                       sequence, request);
    }
}
