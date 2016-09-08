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
 * @ingroup plasma_tradd
 *
 *  Performs an addition of two trapezoidal matrices similarly to the
 * 'pztradd()' function from the PBLAS library:
 *
 *    \f[ B = \alpha * op( A ) + \beta * B, \f]
 *
 *  where op( X ) is one of:
 *    \f[ op( X ) = X,   \f]
 *    \f[ op( X ) = X^T, \f]
 *    \f[ op( X ) = X^H, \f]
 *
 *  alpha and beta are scalars and A, B are matrices with op( A ) an m-by-n or
 *  n-by-m matrix depending on the value of transA and B an m-by-n matrix.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          Specifies the shape of op( A ) and B matrices:
 *          - PlasmaFull:  op( A ) and B are general matrices.
 *          - PlasmaUpper: op( A ) and B are upper trapezoidal matrices.
 *          - PlasmaLower: op( A ) and B are lower trapezoidal matrices.
 *
 * @param[in] transA
 *          Specifies whether the matrix A is non-transposed, transposed, or
 *          conjugate transposed
 *          - PlasmaNoTrans:   op( A ) = A
 *          - PlasmaTrans:     op( A ) = A^T
 *          - PlasmaConjTrans: op( A ) = A^H
 *
 * @param[in] m
 *          Number of rows of the matrices op( A ) and B.
 *          m >= 0.
 *
 * @param[in] n
 *          Number of columns of the matrices op( A ) and B.
 *          n >= 0.
 *
 * @param[in] alpha
 *          Scalar factor of A.
 *
 * @param[in] A
 *          Matrix of size lda-by-k, where k is n when transA == PlasmaNoTrans
 *          and m otherwise.
 *
 * @param[in] lda
 *          Leading dimension of the array A. lda >= max(1,l), where l is m
 *          when transA = PlasmaNoTrans and n otherwise.
 *
 * @param[in] beta
 *          Scalar factor of B.
 *
 * @param[in,out] B
 *          Matrix of size ldb-by-n.
 *          On exit, B = alpha * op( A ) + beta * B
 *
 * @param[in] ldb
 *          Leading dimension of the array B.
 *          ldb >= max(1,m).
 *
 *******************************************************************************
 *
 * @retval PLASMA_SUCCESS successful exit
 *
 *******************************************************************************
 *
 * @sa PLASMA_ztradd_Tile_Async
 * @sa PLASMA_ctradd
 * @sa PLASMA_dtradd
 * @sa PLASMA_stradd
 *
 ******************************************************************************/
int PLASMA_ztradd(PLASMA_enum uplo, PLASMA_enum transA, int m, int n,
                  PLASMA_Complex64_t  alpha,
                  PLASMA_Complex64_t *A, int lda,
                  PLASMA_Complex64_t  beta,
                  PLASMA_Complex64_t *B, int ldb)
{
    int Am, An;
    int Bm, Bn;
    int nb;
    int retval;
    int status;

    PLASMA_desc descA;
    PLASMA_desc descB;

    PLASMA_Complex64_t zzero = 0.0;
    PLASMA_Complex64_t zone  = 1.0;

    // Get PLASMA context
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }

    // Check input arguments
    if ((uplo != PlasmaFull) &&
        (uplo != PlasmaUpper) &&
        (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo");
        return -1;
    }

    if ((transA != PlasmaNoTrans) &&
        (transA != PlasmaTrans) &&
        (transA != PlasmaConjTrans)) {
        plasma_error("illegal value of transA");
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

    if (A == NULL) {
        plasma_error("NULL A");
        return -6;
    }

    if (transA == PlasmaNoTrans) {
        Am = m;
        An = n;
    }
    else {
        Am = n;
        An = m;
    }

    Bm = m;  Bn = n;

    if (lda < imax(1, Am)) {
        plasma_error("illegal value of lda");
        return -7;
    }

    if (B == NULL) {
        plasma_error("NULL B");
        return -9;
    }

    if (ldb < imax(1, Bm)) {
        plasma_error("illegal value of ldb");
        return -10;
    }

    // quick return
    if (m == 0 || n == 0 || (alpha == zzero && beta == zone))
        return PLASMA_SUCCESS;

    // Tune
    // if (plasma_tune(PLASMA_FUNC_ZTRADD, m, n, 0) != PLASMA_SUCCESS) {
    //     plasma_error("plasma_tune() failed");
    //     return status;
    // }
    nb = plasma->nb;

    // Initialize tile matrix descriptors
    descA = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, Am, An, 0, 0, Am, An);

    descB = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, Bm, Bn, 0, 0, Bm, Bn);

    // Allocate matrices in tile layout
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

    // Create sequence
    PLASMA_sequence *sequence = NULL;
    retval = plasma_sequence_create(&sequence);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_sequence_create() failed");
        return retval;
    }

    // Initialize request
    PLASMA_request request = PLASMA_REQUEST_INITIALIZER;

    // Asynchronous block
    #pragma omp parallel
    #pragma omp master
    {
        // Translate to tile layout
        PLASMA_zcm2ccrb_Async(A, lda, &descA, sequence, &request);
        PLASMA_zcm2ccrb_Async(B, ldb, &descB, sequence, &request);

        // Call tile async function
        if (sequence->status == PLASMA_SUCCESS) {
            PLASMA_ztradd_Tile_Async(uplo, transA, alpha, &descA, beta, &descB,
                                     sequence, &request);
        }

        // Revert to LAPACK layout
        PLASMA_zccrb2cm_Async(&descB, B, ldb, sequence, &request);
    }
    // Implicit synchronization

    // Deallocate memory in tile layout
    plasma_desc_mat_free(&descA);
    plasma_desc_mat_free(&descB);

    // Return status
    status = sequence->status;
    plasma_sequence_destroy(sequence);
    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_tradd
 *
 *  Performs an addition of two trapezoidal matrices similarly to the
 * 'pztradd()' function from the PBLAS library. Non-blocking tile version of
 *  PLASMA_ztradd(). May return before the computation is finished. Operates
 *  on matrices stored by tiles. All matrices are passed through descriptors.
 *  All dimensions are taken from the descriptors. Allows for pipelining of
 *  operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          Specifies the shape of op( A ) and B matrices:
 *          - PlasmaFull:  op( A ) and B are general matrices.
 *          - PlasmaUpper: op( A ) and B are upper trapezoidal matrices.
 *          - PlasmaLower: op( A ) and B are lower trapezoidal matrices.
 *
 * @param[in] transA
 *          Specifies whether the matrix A is non-transposed, transposed, or
 *          conjugate transposed
 *          - PlasmaNoTrans:   op( A ) = A
 *          - PlasmaTrans:     op( A ) = A^T
 *          - PlasmaConjTrans: op( A ) = A^H
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          Descriptor of matrix A.
 *
 * @param[in] beta
 *          The scalar beta.
 *
 * @param[in,out] B
 *          Descriptor of matrix B.
 *
 * @param[in] sequence
 *          Identifies the sequence of function calls that this call belongs to
 *         (for completion checks and exception handling purposes). Check the
 *          sequence->status for errors.
 *
 * @param[out] request
 *          Identifies this function call (for exception handling purposes).
 *
 * @retval void
 *          Errors are returned by setting sequence->status and
 *          request->status to error values. The sequence->status and
 *          request->status should never be set to PLASMA_SUCCESS (the
 *          initial values) since another async call may be setting a
 *          failure value at the same time.
 *
 *******************************************************************************
 *
 * @sa PLASMA_ztradd
 * @sa PLASMA_ctradd_Tile_Async
 * @sa PLASMA_dtradd_Tile_Async
 * @sa PLASMA_stradd_Tile_Async
 *
 ******************************************************************************/
void PLASMA_ztradd_Tile_Async(PLASMA_enum uplo, PLASMA_enum transA,
                              PLASMA_Complex64_t alpha, PLASMA_desc *A,
                              PLASMA_Complex64_t beta,  PLASMA_desc *B,
                              PLASMA_sequence *sequence,
                              PLASMA_request  *request)
{
    // Get PLASMA context
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    // Check input arguments
    if ((uplo != PlasmaFull) &&
        (uplo != PlasmaUpper) &&
        (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    if ((transA != PlasmaNoTrans) &&
        (transA != PlasmaTrans) &&
        (transA != PlasmaConjTrans)) {
        plasma_error("illegal value of transA");
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

    // quick return
    int Am = transA == PlasmaNoTrans ? A->m : A->n;
    PLASMA_Complex64_t zzero = (PLASMA_Complex64_t)0.0;

    if ((alpha == zzero || Am == 0) && beta == 1.0)
        return;

    // Call parallel function
    plasma_pztradd(uplo, transA, alpha, *A, beta, *B, sequence, request);
}
