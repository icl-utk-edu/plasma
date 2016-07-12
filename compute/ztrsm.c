/**
 *
 * @file ztrsm.c
 *
 *  PLASMA computational routines
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.8.0
 * @author Mawussi Zounon
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
 * @ingroup plasma_trsm
 *
 *  Performs one of the matrix equations
 *
 *    \f[ op( A )\times X  = \alpha B, \f] or
 *    \f[ op( X )\times A  = \alpha B, \f]
 *
 *  where op( X ) is one of:
 *          - op( X ) = X   or
 *          - op( X ) = X^T or
 *          - op( X ) = X^H,
 *
 *  alpha is a scalar, A and are B m by n matrices.
 *  The matrix X is overwritten by B.
 *
 *******************************************************************************
 *
 * @param[in] side
 *          - PlasmaLeft:  A*X = B,
 *          - PlasmaRight: X*A = B.
 *
 * @param[in] uplo
 *          - PlasmaUpper: A is upper triangular,
 *          - PlasmaLower: A is lower triangular.
 *
 * @param[in] transA
 *          - PlasmaNoTrans:   A is not transposed,
 *          - PlasmaTrans:     A is transposed,
 *          - PlasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] diag
 *          - PlasmaNonUnit: A is non unit,
 *          - PlasmaUnit:    A us unit.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in] nrhs
 *          The number of columns of the matrix B. nrhs >= 0.
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          The triangular matrix. If uplo = PlasmaUpper, the leading n-by-n
 *          upper triangular part of the array A contains the upper triangular
 *          matrix, and the strictly lower triangular part of A is not
 *          referenced. If uplo = PlasmaLower, the leading n-by-n lower
 *          triangular part of A contains the lower triangular matrix,
 *          and the strictly upper triangular part of A is not referenced.
 *          If diag = PlasmaUnit, the diagonal elements of A are also not
 *          referenced and are assumed to be 1.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,n).
 *
 * @param[in,out] B
 *          On entry, the n-by-nrhs right hand side matrix B.
 *          On exit, if return value = 0, the n-by-nrhs solution matrix X.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1,n).
 *
 *
 *******************************************************************************
 *
 * @retval PLASMA_SUCCESS successful exit
 *
 *******************************************************************************
 *
 * @sa PLASMA_ztrsm_Tile_Async
 * @sa PLASMA_ctrsm
 * @sa PLASMA_dtrsm
 * @sa PLASMA_strsm
 *
 ******************************************************************************/
int PLASMA_ztrsm(PLASMA_enum side, PLASMA_enum uplo,
                 PLASMA_enum transA, PLASMA_enum diag,
                 int n, int nrhs, PLASMA_Complex64_t alpha,
                 PLASMA_Complex64_t *A, int lda,
                 PLASMA_Complex64_t *B, int ldb)
{
    int An;
    int nb;
    int retval;
    int status;

    PLASMA_desc descA;
    PLASMA_desc  descB;

    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }

    // Check input arguments
    if ((side != PlasmaLeft) &&
        (side != PlasmaRight)) {
        plasma_error("illegal value of side");
        return -1;
    }
    if ((uplo != PlasmaUpper) &&
            (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo");
        return -2;
    }
    if ((transA != PlasmaConjTrans) &&
        (transA != PlasmaNoTrans) &&
        (transA != PlasmaTrans )) {
        plasma_error("illegal value of transA");
        return -3;
    }
    if ((diag != PlasmaUnit) &&
            (diag != PlasmaNonUnit)) {
        plasma_error("illegal value of diag");
        return -4;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -5;
    }
    if (nrhs < 0) {
        plasma_error("illegal value of nrhs");
        return -6;
    }
    if (lda < imax(1, An)) {
        plasma_error("illegal value of lda");
        return -8;
    }
    if (ldb < imax(1, n)) {
        plasma_error("illegal value of ldb");
        return -10;
    }

    if (side == PlasmaLeft) {
        An = n;
    }
    else {
        An = nrhs;
    }

    // quick return
    if ((n == 0) || (nrhs == 0))
        return PLASMA_SUCCESS;

    // Tune.
    // if (plasma_tune(PLASMA_FUNC_ZTRSM, n, nrhs, 0) != PLASMA_SUCCESS) {
    //     plasma_error("plasma_tune() failed");
    //     return status;
    // }
    nb = plasma->nb;

    // Initialize tile matrix descriptors.
    descA = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, An, An, 0, 0, An, An);

    descB = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, n, nrhs, 0, 0, n, nrhs);

    // Allocate matrices in tile layout.
    retval = plasma_desc_mat_alloc(&descA);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_desc_mat_alloc() failed");
        return retval;
    }
    retval = plasma_desc_mat_alloc(&descB);
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
            PLASMA_zcm2ccrb_Async(B, ldb, &descB, sequence, &request);

        // Call the tile async function.
        if (sequence->status == PLASMA_SUCCESS) {
            PLASMA_ztrsm_Tile_Async(side, uplo,
                                    transA, diag,
                                    alpha, &descA,
                                           &descB,
                                    sequence, &request);
        }

        // Translate back to LAPACK layout.
        if (sequence->status == PLASMA_SUCCESS)
            PLASMA_zccrb2cm_Async(&descB, B, ldb, sequence, &request);
    } // pragma omp parallel block closed

    // Check for errors in the async execution
    if (sequence->status != PLASMA_SUCCESS)
        return sequence->status;

    // Free matrices in tile layout.
    plasma_desc_mat_free(&descA);
    plasma_desc_mat_free(&descB);

    // Return status.
    status = sequence->status;
    plasma_sequence_destroy(sequence);
    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_trsm
 *
 *  Computes triangular solve.
 *  Non-blocking tile version of PLASMA_ztrsm().
 *  Operates on matrices stored by tiles.
 *  All matrices are passed through descriptors.
 *  All dimensions are taken from the descriptors.
 *  Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] side
 *          - PlasmaLeft:  A*X = B,
 *          - PlasmaRight: X*A = B.
 *
 * @param[in] uplo
 *          - PlasmaUpper: A is uppert triangular,
 *          - PlasmaLower: A is lower triangular.
 *
 * @param[in] transA
 *          - PlasmaNoTrans:   A is transposed,
 *          - PlasmaTrans:     A is not transposed,
 *          - PlasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] diag
 *          - PlasmaNonUnit: A is non unit,
 *          - PlasmaUnit:    A us unit.
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          Descriptor of matrix A.
 *
 * @param[in] B
 *          Descriptor of matrix B.
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
 *          request->status should never be set to PLASMA_SUCCESS (the
 *          initial values) since another async call may be setting a
 *          failure value at the same time.
 *
 *******************************************************************************
 *
 * @sa PLASMA_ztrsm
 * @sa PLASMA_ctrsm_Tile_Async
 * @sa PLASMA_dtrsm_Tile_Async
 * @sa PLASMA_strsm_Tile_Async
 *
 ******************************************************************************/
void PLASMA_ztrsm_Tile_Async(PLASMA_enum side, PLASMA_enum uplo,
                             PLASMA_enum transA, PLASMA_enum diag,
                             PLASMA_Complex64_t alpha, PLASMA_desc *A,
                                                       PLASMA_desc *B,
                             PLASMA_sequence *sequence, PLASMA_request *request)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    // Check input arguments
    if ((side != PlasmaLeft) &&
        (side != PlasmaRight)) {
        plasma_error("illegal value of side");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if ((uplo != PlasmaUpper) &&
        (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if ((transA != PlasmaConjTrans) &&
        (transA != PlasmaNoTrans) &&
        (transA != PlasmaTrans)) {
        plasma_error("illegal value of transA");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if ((diag != PlasmaUnit) &&
        (diag != PlasmaNonUnit)) {
        plasma_error("illegal value of diag");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (plasma_desc_check(A) != PLASMA_SUCCESS) {
        plasma_error("invalid A");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (plasma_desc_check(B) != PLASMA_SUCCESS) {
        plasma_error("invalid B");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    if (A->nb != A->mb || B->nb != B->mb) {
        plasma_error("only square tiles supported");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    // quick return
    if ((B->m == 0) || (B->n == 0))
      return;

    // Call the parallel function.
    plasma_pztrsm(side, uplo,
                  transA, diag,
                  alpha, *A,
                         *B,
                  sequence, request);

    return;
}
