/**
 *
 * @file zherk.c
 *
 *  PLASMA computational routine.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley, Univ. of Colorado Denver and
 *  Univ. of Manchester.
 *
 * @version 3.0.0
 * @author Pedro V. Lara
 * @date 2016-05-16
 * @precisions normal z -> c
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
 * @ingroup PLASMA_Complex64_t
 *
 *  Performs one of the hermitian rank k operations
 *
 *    \f[ C = \alpha [ op( A ) \times conjg( op( A )' )] + \beta C \f],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = conjg( X' )
 *
 *  where alpha and beta are real scalars, C is an n-by-n hermitian
 *  matrix and A is an n-by-k matrix in the first case and a k-by-n
 *  matrix in the second case.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of C is stored;
 *          - PlasmaLower: Lower triangle of C is stored.
 *
 * @param[in] trans
 *          Specifies whether the matrix A is transposed or conjugate transposed:
 *          - PlasmaNoTrans:   A is not transposed;
 *          - PlasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] n
 *          The number of n specifies the order of the matrix C. n must be at least zero.
 *
 * @param[in] k
 *          The number of k specifies the number of columns of the matrix op( A ).
 *
 * @param[in] alpha
 *          The number of alpha specifies the scalar alpha.
 *
 * @param[in] A
 *          A is a lda-by-ka matrix, where ka is k when trans = PlasmaNoTrans,
 *          and is n otherwise.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda must be at least
 *          max( 1, n ) if trans == PlasmaNoTrans, otherwise lda must
 *          be at least max( 1, k ).
 *
 * @param[in] beta
 *          beta specifies the scalar beta
 *
 * @param[in,out] C
 *          C is a ldc-by-n matrix.
 *          On exit, the array uplo part of the matrix is overwritten
 *          by the uplo part of the updated matrix.
 *
 * @param[in] ldc
 *          The leading dimension of the array C. ldc >= max( 1, n ).
 *
 *******************************************************************************
 *
 * @retval PLASMA_SUCCESS successful exit
 *
 *******************************************************************************
 *
 * @sa PLASMA_zherk_Tile_Async
 * @sa PLASMA_cherk
 * @sa PLASMA_dherk
 * @sa PLASMA_sherk
 *
 ******************************************************************************/
int PLASMA_zherk(PLASMA_enum uplo, PLASMA_enum trans, int n, int k,
                 double alpha, PLASMA_Complex64_t *A, int lda,
                 double beta,  PLASMA_Complex64_t *C, int ldc)
{
    int Am, An;
    int nb;
    int retval;
    int status;

    PLASMA_desc descA;
    PLASMA_desc descC;

    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }

    // Check input arguments.
    if ((uplo != PlasmaUpper) && (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo");
        return -1;
    }
    if ((trans != PlasmaNoTrans) && (trans != PlasmaConjTrans)) {
        plasma_error("illegal value of trans");
        return -2;
    }
    if ( trans == PlasmaNoTrans ) {
        Am = n; An = k;
    } else {
        Am = k; An = n;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -3;
    }
    if (k < 0) {
        plasma_error("illegal value of k");
        return -4;
    }
    if (lda < imax(1, Am)) {
        plasma_error("illegal value of lda");
        return -7;
    }
    if (ldc < imax(1, n)) {
        plasma_error("illegal value of ldc");
        return -10;
    }

    /* Quick return */
    if (n == 0 ||
        ((alpha == 0.0 || k == 0.0) && beta == 1.0))
        return PLASMA_SUCCESS;

    // Tune
    // status = plasma_tune(PLASMA_FUNC_ZSYRK, n, k, 0);
    // if (status != PLASMA_SUCCESS) {
    //     plasma_error("PLASMA_zherk", "plasma_tune() failed");
    //     return status;
    // }

    
    /* Set NT & KT */
    nb = plasma->nb;

    // Initialize tile matrix descriptors.
    descA = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, Am, An, 0, 0, Am, An);

    descC = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, n, n, 0, 0, n, n);

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
        /* the Async functions are submitted here.  If an error occurs
           (at submission time or at run time) the sequence->status
           will be marked with an error.  After an error, the next
           Async will not _insert_ more tasks into the runtime.  The
           sequence->status can be checked after each call to _Async
           or at the end of the parallel region. */

        // Translate to tile layout.
        PLASMA_zcm2ccrb_Async(A, lda, &descA, sequence, &request);
        if (sequence->status == PLASMA_SUCCESS) 
            PLASMA_zcm2ccrb_Async(C, ldc, &descC, sequence, &request);

        // Call the tile async function.
        if (sequence->status == PLASMA_SUCCESS) {
            PLASMA_zherk_Tile_Async(uplo, trans,
                                    alpha, &descA,
                                    beta, &descC,
                                    sequence, &request);
        }

        // Translate back to LAPACK layout.
        if (sequence->status == PLASMA_SUCCESS) 
            PLASMA_zccrb2cm_Async(&descC, C, ldc, sequence, &request);
    } /* pragma omp parallel block closed  */

    // Check for errors in the async execution
    if (sequence->status != PLASMA_SUCCESS)
        return sequence->status;

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
 * @ingroup PLASMA_Complex64_t_Tile_Async
 *
 *  Performs rank-k update.
 *  Tile equivalent of PLASMA_zherk().
 *  Operates on matrices stored by tiles.
 *  All matrices are passed through descriptors.
 *  All dimensions are taken from the descriptors.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of C is stored;
 *          - PlasmaLower: Lower triangle of C is stored.
 *
 * @param[in] trans
 *          Specifies whether the matrix A is transposed or conjugate transposed:
 *          - PlasmaNoTrans:   A is not transposed;
 *          - PlasmaConjTrans: A is conjugated transposed.
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
 * @param[in,out] C
 *          Descriptor of matrix C.
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
 * @sa PLASMA_zherk
 * @sa PLASMA_zherk_Tile_Async
 * @sa PLASMA_cherk_Tile_Async
 * @sa PLASMA_dherk_Tile_Async
 * @sa PLASMA_sherk_Tile_Async
 *
 ******************************************************************************/
void PLASMA_zherk_Tile_Async(PLASMA_enum uplo, PLASMA_enum trans,
                            double alpha, PLASMA_desc *A,
                            double beta,  PLASMA_desc *C,
                            PLASMA_sequence *sequence, PLASMA_request *request)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    // Check input arguments.
    if ((uplo != PlasmaUpper) &&
        (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if ((trans != PlasmaNoTrans) &&
        (trans != PlasmaConjTrans)) {
        plasma_error("illegal value of trans");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (plasma_desc_check(A) != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        plasma_error("invalid A");
        return;
    }
    if (plasma_desc_check(C) != PLASMA_SUCCESS) {
        plasma_error("invalid C");
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

    int Am, An, Ai, Aj, Amb, Anb;
    int Bm, Bn, Bi, Bj, Bmb, Bnb;

    if (trans == PlasmaNoTrans) {
        Am  = A->m;
        An  = A->n;
        Amb = A->mb;
        Anb = A->nb;
        Ai  = A->i;
        Aj  = A->j;
    }
    else {
        Am  = A->n;
        An  = A->m;
        Amb = A->nb;
        Anb = A->mb;
        Ai  = A->j;
        Aj  = A->i;
    }

    if (C->mb != C->nb) {
        plasma_error("only square tiles supported");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (Amb != C->mb) {
        plasma_error("tile sizes have to match");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (C->m != C->n) {
        plasma_error("only square matrix C is supported");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (Am != C->m) {
        plasma_error("size of matrices have to match");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    // Check sequence status.
    if (sequence->status != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);
        return;
    }

    // quick return
    if (C->m == 0 || 
	((alpha == 0.0 || An == 0) && beta == 1.0))
        return;

    // Call the parallel function.
    plasma_pzherk(uplo, trans,
                  alpha, *A,
                  beta, *C,
                  sequence, request);

    return;
}
