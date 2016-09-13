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

#include "plasma_types.h"
#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_z.h"

/***************************************************************************//**
 *
 * @ingroup plasma_unmqr
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
 *  as returned by PLASMA_zgeqrf. Q is of order m if side = PlasmaLeft
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
 *          - PlasmaNoTrans:    No transpose, apply Q;
 *          - Plasma_ConjTrans: Transpose, apply Q^H.
 *
 * @param[in] m
 *          The number of rows of the matrix C. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix C. n >= 0.
 *
 * @param[in] k
 *          The number of elementary reflectors whose product defines
 *          the matrix Q.
 *          If side == PlasmaLeft,  m >= k >= 0.
 *          If side == PlasmaRight, n >= k >= 0.
 *
 * @param[in] A
 *          Details of the QR factorization of the original matrix A as returned
 *          by PLASMA_zgeqrf.
 *
 * @param[in] lda
 *          The leading dimension of the array A.
 *          If side == PlasmaLeft,  lda >= max(1,m).
 *          If side == PlasmaRight, lda >= max(1,n).
 *
 * @param[in] descT
 *          Auxiliary factorization data, computed by PLASMA_zgeqrf.
 *
 * @param[in,out] C
 *          On entry, the m-by-n matrix C.
 *          On exit, C is overwritten by Q*C, Q^H*C, C*Q, or C*Q^H.
 *
 * @param[in] ldc
 *          The leading dimension of the array C. ldc >= max(1,m).
 *
 *******************************************************************************
 *
 * @retval PLASMA_SUCCESS successful exit
 * @retval < 0 if -i, the i-th argument had an illegal value
 *
 *******************************************************************************
 *
 * @sa plasma_omp_zunmqr
 * @sa PLASMA_cunmqr
 * @sa PLASMA_dormqr
 * @sa PLASMA_sormqr
 * @sa PLASMA_zgeqrf
 *
 ******************************************************************************/
int PLASMA_zunmqr(PLASMA_enum side, PLASMA_enum trans, int m, int n, int k,
                  plasma_complex64_t *A, int lda,
                  plasma_desc_t *descT,
                  plasma_complex64_t *C, int ldc)
{
    int ib, nb;
    int retval;
    int status;

    plasma_desc_t descA, descC;

    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }

    int am;
    if (side == PlasmaLeft) {
        am = m;
    }
    else {
        am = n;
    }

    // Check input arguments
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
    if ((k < 0) || (k > am)) {
        plasma_error("illegal value of k");
        return -5;
    }
    if (lda < imax(1, am)) {
        plasma_error("illegal value of lda");
        return -7;
    }
    if (ldc < imax(1, m)) {
        plasma_error("illegal value of ldc");
        return -10;
    }
    // Quick return
    if (m == 0 || n == 0 || k == 0)
        return PLASMA_SUCCESS;

    // Tune NB & IB depending on M, K & N; Set NB
    //status = plasma_tune(PLASMA_FUNC_ZGELS, M, K, N);
    //if (status != PLASMA_SUCCESS) {
    //    plasma_error("PLASMA_zunmqr", "plasma_tune() failed");
    //    return status;
    //}
    ib = plasma->ib;
    nb = plasma->nb;

    // Initialize tile matrix descriptors.
    descA = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, lda, k, 0, 0, am, k);

    descC = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, ldc, n, 0, 0, m, n);

    // Allocate matrices in tile layout.
    retval = plasma_desc_mat_alloc(&descA);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_desc_mat_alloc() failed");
        return retval;
    }
    retval = plasma_desc_mat_alloc(&descC);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_desc_mat_alloc() failed");
        plasma_desc_mat_free(&descA);
        return retval;
    }

    // Allocate workspace.
    plasma_workspace_t work;
    size_t lwork = ib*nb;  // unmqr: work
    retval = plasma_workspace_alloc(&work, lwork, PlasmaComplexDouble);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_workspace_alloc() failed");
        return retval;
    }

    // Create sequence.
    plasma_sequence_t *sequence = NULL;
    retval = plasma_sequence_create(&sequence);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_sequence_create() failed");
        return retval;
    }

    // Initialize request.
    plasma_request_t request = PLASMA_REQUEST_INITIALIZER;

    #pragma omp parallel
    #pragma omp master
    {
        // the Async functions are submitted here.  If an error occurs
        // (at submission time or at run time) the sequence->status
        // will be marked with an error.  After an error, the next
        // Async will not _insert_ more tasks into the runtime.  The
        // sequence->status can be checked after each call to _Async
        // or at the end of the parallel region.

        // Translate to tile layout.
        PLASMA_zcm2ccrb_Async(A, lda, &descA, sequence, &request);
        PLASMA_zcm2ccrb_Async(C, ldc, &descC, sequence, &request);

        // Call the tile async function.
        plasma_omp_zunmqr(side, trans, &descA, descT, &descC,
                                 &work, sequence, &request);

        // Translate back to LAPACK layout.
        // this does not seem needed for A
        //PLASMA_zccrb2cm_Async(&descA, A, lda, sequence, &request);
        PLASMA_zccrb2cm_Async(&descC, C, ldc, sequence, &request);
    }
    // implicit synchronization

    plasma_workspace_free(&work);

    // Free matrices in tile layout.
    plasma_desc_mat_free(&descA);
    plasma_desc_mat_free(&descC);

    // Return status.
    status = sequence->status;
    plasma_sequence_destroy(sequence);
    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_unmqr
 *
 *  Non-blocking tile version of PLASMA_zunmqr().
 *  May return before the computation is finished.
 *  Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
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
 *          by PLASMA_zgeqrf.
 *
 * @param[in] T
 *          Descriptor of matrix T.
 *          Auxiliary factorization data, computed by PLASMA_zgeqrf.
 *
 * @param[in,out] C
 *          Descriptor of matrix C.
 *          On entry, the m-by-n matrix C.
 *          On exit, C is overwritten by Q*C, Q^H*C, C*Q, or C*Q^H.
 *
 * @param[in] work
 *          Workspace for the auxiliary arrays needed by some coreblas kernels.
 *          For multiplication by Q contains preallocated space for WORK
 *          arrays. Allocated by the plasma_workspace_alloc function.
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
 *          request->status should never be set to PLASMA_SUCCESS (the
 *          initial values) since another async call may be setting a
 *          failure value at the same time.
 *
 *******************************************************************************
 *
 * @sa PLASMA_zunmqr
 * @sa plasma_omp_cunmqr
 * @sa plasma_omp_dormqr
 * @sa plasma_omp_sormqr
 * @sa plasma_omp_zgeqrf
 *
 ******************************************************************************/
void plasma_omp_zunmqr(PLASMA_enum side, PLASMA_enum trans,
                       plasma_desc_t *A, plasma_desc_t *T, plasma_desc_t *C,
                       plasma_workspace_t *work,
                       plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    // Check input arguments
    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        plasma_error("strange side - neither PlasmaLeft nor PlasmaRight");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if ((trans != Plasma_ConjTrans) && (trans != PlasmaNoTrans)) {
        plasma_error(
            "strange trans - neither Plasma_ConjTrans nor PlasmaTrans");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (plasma_desc_check(A) != PLASMA_SUCCESS) {
        plasma_error("invalid descriptor A");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (plasma_desc_check(T) != PLASMA_SUCCESS) {
        plasma_error("invalid descriptor T");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (plasma_desc_check(C) != PLASMA_SUCCESS) {
        plasma_error("invalid descriptor C");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (sequence == NULL) {
        plasma_error("NULL sequence");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (request == NULL) {
        plasma_error("NULL request");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (A->nb != A->mb || C->nb != C->mb) {
        plasma_error("only square tiles supported");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    // Check sequence status.
    if (sequence->status != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);
        return;
    }

    // Quick return
    // (m == 0 || n == 0 || k == 0)
    if (C->m == 0 || C->n == 0 || imin(A->m, A->n) == 0)
        return;

    plasma_pzunmqr(side, trans,
                   *A, *C, *T,
                   work, sequence, request);
}
