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
 * @ingroup plasma_unmlq
 *
 *  Overwrites the general complex m-by-n matrix C with
 *
 *                                 side = PlasmaLeft     side = PlasmaRight
 *  trans = PlasmaNoTrans             Q * C                 C * Q
 *  trans = Plasma_ConjTrans        Q^H * C                 C * Q^H
 *
 *  where Q is an orthogonal (or unitary) matrix defined as the product of k
 *  elementary reflectors
 *
 *        Q = H(1) H(2) . . . H(k)
 *
 *  as returned by plasma_zgelqf. Q is of order m if side = PlasmaLeft
 *  and of order n if side = PlasmaRight.
 *
 *******************************************************************************
 *
 * @param[in] side
 *          Intended usage:
 *          - PlasmaLeft:  apply Q or Q^H from the left;
 *          - PlasmaRight: apply Q or Q^H from the right.
 *
 * @param[in] trans
 *          Intended usage:
 *          - PlasmaNoTrans:    apply Q;
 *          - Plasma_ConjTrans: apply Q^H.
 *
 * @param[in] m
 *          The number of rows of the matrix C. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix C. n >= 0.
 *
 * @param[in] k
 *          The number of rows of elementary tile reflectors whose product
 *          defines the matrix Q.
 *          If side == PlasmaLeft,  m >= k >= 0.
 *          If side == PlasmaRight, n >= k >= 0.
 *
 * @param[in] pA
 *          Details of the LQ factorization of the original matrix A as returned
 *          by plasma_zgelqf.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,k).
 *
 * @param[in] T
 *          Auxiliary factorization data, computed by plasma_zgelqf.
 *
 * @param[in,out] pC
 *          On entry, pointer to the m-by-n matrix C.
 *          On exit, C is overwritten by Q*C, Q^H*C, C*Q, or C*Q^H.
 *
 * @param[in] ldc
 *          The leading dimension of the array C. ldc >= max(1,m).
 *
 *******************************************************************************
 *
 * @retval PlasmaSuccess successful exit
 * @retval < 0 if -i, the i-th argument had an illegal value
 *
 *******************************************************************************
 *
 * @sa plasma_omp_zunmlq
 * @sa plasma_cunmlq
 * @sa plasma_dormlq
 * @sa plasma_sormlq
 * @sa plasma_zgelqf
 *
 ******************************************************************************/
int plasma_zunmlq(plasma_enum_t side, plasma_enum_t trans,
                  int m, int n, int k,
                  plasma_complex64_t *pA, int lda,
                  plasma_desc_t T,
                  plasma_complex64_t *pC, int ldc)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    // Check input arguments.
    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        plasma_error("illegal value of side");
        return -1;
    }
    if ((trans != Plasma_ConjTrans) && (trans != PlasmaNoTrans)) {
        plasma_error("illegal value of trans");
        return -2;
    }
    if (m < 0) {
        plasma_error("illegal value of m");
        return -3;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -4;
    }

    int an;
    if (side == PlasmaLeft) {
        an = m;
    }
    else {
        an = n;
    }

    if ((k < 0) || (k > an)) {
        plasma_error("illegal value of k");
        return -5;
    }
    if (lda < imax(1, k)) {
        plasma_error("illegal value of lda");
        return -7;
    }
    if (ldc < imax(1, m)) {
        plasma_error("illegal value of ldc");
        return -10;
    }

    // quick return
    if (m == 0 || n == 0 || k == 0)
        return PlasmaSuccess;

    // Tune parameters.
    if (plasma->tuning)
        plasma_tune_gelqf(plasma, PlasmaComplexDouble, m, n);

    // Set tiling parameters.
    int ib = plasma->ib;
    int nb = plasma->nb;

    // Create tile matrices.
    plasma_desc_t A;
    plasma_desc_t C;
    int retval;
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        k, an, 0, 0, k, an, &A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        return retval;
    }
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        m, n, 0, 0, m, n, &C);
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
        plasma_omp_zge2desc(pC, ldc, C, &sequence, &request);

        // Call the tile async function.
        plasma_omp_zunmlq(side, trans,
                          A, T, C, work,
                          &sequence, &request);

        // Translate back to LAPACK layout.
        plasma_omp_zdesc2ge(C, pC, ldc, &sequence, &request);
    }
    // implicit synchronization

    plasma_workspace_destroy(&work);

    // Free matrices in tile layout.
    plasma_desc_destroy(&A);
    plasma_desc_destroy(&C);

    // Return status.
    int status = sequence.status;
    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_unmlq
 *
 *  Non-blocking tile version of plasma_zunmlq().
 *  May return before the computation is finished.
 *  Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] side
 *          Intended usage:
 *          - PlasmaLeft:  apply Q or Q^H from the left;
 *          - PlasmaRight: apply Q or Q^H from the right.
 *
 * @param[in] trans
 *          Intended usage:
 *          - PlasmaNoTrans:    apply Q;
 *          - Plasma_ConjTrans: apply Q^H.
 *
 * @param[in] A
 *          Descriptor of matrix A stored in the tile layout.
 *          Details of the QR factorization of the original matrix A as returned
 *          by plasma_zgeqrf.
 *
 * @param[in] T
 *          Descriptor of matrix T.
 *          Auxiliary factorization data, computed by plasma_zgeqrf.
 *
 * @param[in,out] C
 *          Descriptor of matrix C.
 *          On entry, the m-by-n matrix C.
 *          On exit, C is overwritten by Q*C, Q^H*C, C*Q, or C*Q^H.
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
 * @sa plasma_zunmlq
 * @sa plasma_omp_cunmlq
 * @sa plasma_omp_dormlq
 * @sa plasma_omp_sormlq
 * @sa plasma_omp_zgelqf
 *
 ******************************************************************************/
void plasma_omp_zunmlq(plasma_enum_t side, plasma_enum_t trans,
                       plasma_desc_t A, plasma_desc_t T, plasma_desc_t C,
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
    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        plasma_error("invalid value of side");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if ((trans != Plasma_ConjTrans) && (trans != PlasmaNoTrans)) {
        plasma_error("invalid value of trans");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
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
    if (plasma_desc_check(C) != PlasmaSuccess) {
        plasma_error("invalid C");
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
    if (C.m == 0 || C.n == 0 || A.m == 0 || A.n == 0)
        return;

    // Call the parallel function.
    if (plasma->householder_mode == PlasmaTreeHouseholder) {
        plasma_pzunmlq_tree(side, trans,
                            A, T, C,
                            work, sequence, request);
    }
    else {
        plasma_pzunmlq(side, trans,
                       A, T, C,
                       work, sequence, request);
    }
}
