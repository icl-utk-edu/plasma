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
 *                               side = 'PlasmaLeft'     side = 'PlasmaRight'
 *  trans = 'PlasmaNoTrans':          Q * C                 C * Q
 *  trans = 'Plasma_ConjTrans':     Q^H * C                 C * Q^H
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
 * @sa PLASMA_zunmqr_Tile_Async
 * @sa PLASMA_cunmqr
 * @sa PLASMA_dormqr
 * @sa PLASMA_sormqr
 * @sa PLASMA_zgeqrf
 *
 ******************************************************************************/
int PLASMA_zunmqr(PLASMA_enum side, PLASMA_enum trans, int m, int n, int k,
                  PLASMA_Complex64_t *A, int lda,
                  PLASMA_desc *descT,
                  PLASMA_Complex64_t *C, int ldc)
{
    int nb;
    int retval;
    int status;

    PLASMA_desc descA, descC;

    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }

    int am;
    if ( side == PlasmaLeft ) {
        am = m;
    } else {
        am = n;
    }

    // Check input arguments
    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        plasma_error("illegal value of side");
        return -1;
    }
    if ((trans != Plasma_ConjTrans) && (trans != PlasmaNoTrans)){
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
    // Quick return - currently NOT equivalent to LAPACK's:
    if (imin(m, imin(n, k)) == 0)
        return PLASMA_SUCCESS;

    // Tune NB & IB depending on M, K & N; Set NB
    //status = plasma_tune(PLASMA_FUNC_ZGELS, M, K, N);
    //if (status != PLASMA_SUCCESS) {
    //    plasma_error("PLASMA_zunmqr", "plasma_tune() failed");
    //    return status;
    //}
    
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

    // Create sequence.
    PLASMA_sequence *sequence = NULL;
    retval = plasma_sequence_create(&sequence);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_sequence_create() failed");
        return retval;
    }

    // Initialize request.
    PLASMA_request request = PLASMA_REQUEST_INITIALIZER;

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
        if (sequence->status == PLASMA_SUCCESS)
            PLASMA_zcm2ccrb_Async(C, ldc, &descC, sequence, &request);

        // Call the tile async function.
        if (sequence->status == PLASMA_SUCCESS) {
            PLASMA_zunmqr_Tile_Async(side, trans, &descA, descT, &descC,
                                     sequence, &request);
        }

        // Translate back to LAPACK layout.
        // this does not seem needed for A
        //if (sequence->status == PLASMA_SUCCESS)
        //    PLASMA_zccrb2cm_Async(&descA, A, lda, sequence, &request);
        if (sequence->status == PLASMA_SUCCESS)
            PLASMA_zccrb2cm_Async(&descC, C, ldc, sequence, &request);
    } // pragma omp parallel block closed

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
 * @param[in] descA
 *          Descriptor of matrix A stored in the tile layout.
 *          Details of the QR factorization of the original matrix A as returned
 *          by PLASMA_zgeqrf.
 *
 * @param[in] descT
 *          Descriptor of matrix T.
 *          Auxiliary factorization data, computed by PLASMA_zgeqrf.
 *
 * @param[in,out] descC
 *          Descriptor of matrix C.
 *          On entry, the m-by-n matrix C.
 *          On exit, C is overwritten by Q*C, Q^H*C, C*Q, or C*Q^H.
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
 * @sa PLASMA_cunmqr_Tile_Async
 * @sa PLASMA_dormqr_Tile_Async
 * @sa PLASMA_sormqr_Tile_Async
 * @sa PLASMA_zgeqrf_Tile_Async
 *
 ******************************************************************************/
void PLASMA_zunmqr_Tile_Async(PLASMA_enum side, PLASMA_enum trans,
                              PLASMA_desc *descA, PLASMA_desc *descT, 
                              PLASMA_desc *descC,
                              PLASMA_sequence *sequence, 
                              PLASMA_request *request)
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
    if ((trans != Plasma_ConjTrans) && (trans != PlasmaNoTrans)){
        plasma_error(
            "strange trans - neither Plasma_ConjTrans nor PlasmaTrans");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (plasma_desc_check(descA) != PLASMA_SUCCESS) {
        plasma_error("invalid descriptor A");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (plasma_desc_check(descT) != PLASMA_SUCCESS) {
        plasma_error("invalid descriptor T");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (plasma_desc_check(descC) != PLASMA_SUCCESS) {
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
    if (descA->nb != descA->mb || descC->nb != descC->mb) {
        plasma_error("only square tiles supported");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    // Check sequence status.
    if (sequence->status != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);
        return;
    }

    // quick return was commented in 2.8.0
    //if (imin(m, imin(n, k)) == 0)
    //    return PLASMA_SUCCESS;

    plasma_pzunmqr(side, trans,
                   *descA, *descC, *descT,
                   sequence, request);
    
    return;
}
