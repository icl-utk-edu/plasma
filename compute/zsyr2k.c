/**
 *
 * @file zsyr2k.c
 *
 *  PLASMA computational routine.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley, Univ. of Colorado Denver and
 *  Univ. of Manchester.
 *
 * @version 3.0.0
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
 * @ingroup plasma_syr2k
 *
 *  PLASMA_zsyr2k - Performs one of the symmetric rank 2k operations
 *
 *    \f[ C = \alpha A \times B^T + \alpha B \times A^T + \beta C \f],
 *    or
 *    \f[ C = \alpha A^T \times B + \alpha B^T \times A + \beta C \f],
 *
 *  where alpha and beta are complex scalars, C is an n-by-n symmetric
 *  matrix, and A and B are n-by-k matrices in the first case and k-by-n
 *  matrices in the second case.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of C is stored;
 *          - PlasmaLower: Lower triangle of C is stored.
 *
 * @param[in] trans
 *          - PlasmaNoTrans:
 *            \f[ C = \alpha A \times B^T + \alpha B \times A^T + \beta C \f];
 *          - PlasmaTrans:
 *            \f[ C = \alpha A^T \times B + \alpha B^T \times A + \beta C \f].
 *
 * @param[in] n
 *          The order of the matrix C. n must be at least zero.
 *
 * @param[in] k
 *          The number of columns of the A and B matrices
 *          with trans = PlasmaNoTrans. Or the number of rows of the A
 *          and B matrices with trans = PlasmaTrans.
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          A  lda-by-ka matrix, where ka is k when trans = PlasmaNoTrans,
 *          and is n otherwise.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda must be at least
 *          max( 1, n ), otherwise lda must be at least max( 1, k ).
 *
 * @param[in] B
 *          A ldb-by-kb matrix, where kb is k when trans = PlasmaNoTrans,
 *          and is n otherwise.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb must be at least
 *          max( 1, n ), otherwise ldb must be at least max( 1, k ).
 *
 * @param[in] beta
 *          The scalar beta.
 *
 * @param[in,out] C
 *          A ldc-by-n matrix.
 *          On exit, the array uplo part of the matrix is overwritten
 *          by the uplo part of the updated matrix.
 *
 * @param[in] ldc
 *          The leading dimension of the array C. ldc >= max( 1, n ).
 *
 *******************************************************************************
 *
 * @retval  PLASMA_SUCCESS successful exit
 *
 *******************************************************************************
 *
 * @sa PLASMA_zsyr2k_Tile_Async
 * @sa PLASMA_csyr2k
 * @sa PLASMA_dsyr2k
 * @sa PLASMA_ssyr2k
 *
 ******************************************************************************/
int PLASMA_zsyr2k(PLASMA_enum uplo, PLASMA_enum trans,
                  int n, int k,
                  PLASMA_Complex64_t alpha,
                  PLASMA_Complex64_t *A, int lda,
                  PLASMA_Complex64_t *B, int ldb,
                  PLASMA_Complex64_t beta, PLASMA_Complex64_t *C, int ldc)
{
    int Am, An;
    int Bm, Bn;
    int nb;
    int retval;
    int status;
    PLASMA_Complex64_t zzero = 0.0;

    PLASMA_desc descA;
    PLASMA_desc descB;
    PLASMA_desc descC;

    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }

    // Check input arguments.
    if ((uplo != PlasmaUpper) &&
        (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo");
        return -1;
    }
    if ((trans != PlasmaNoTrans) &&
        (trans != PlasmaTrans)) {
        plasma_error("illegal value of trans");
        return -2;
    }
    if (trans == PlasmaNoTrans) {
        Am = n;
        An = k;
        Bm = n;
        Bn = k;
    }
    else {
        Am = k;
        An = n;
        Bm = k;
        Bn = n;
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
    if (ldb < imax(1, Bm)) {
        plasma_error("illegal value of ldb");
        return -9;
    }
    if (ldc < imax(1, n)) {
        plasma_error("illegal value of ldc");
        return -12;
    }

    // quick return
    if (n == 0 || ((alpha == zzero || k == 0.0) && beta == (double)1.0))
        return PLASMA_SUCCESS;

    // Tune
    // status = plasma_tune(PLASMA_FUNC_ZSYR2K, n, k, 0);
    // if (status != PLASMA_SUCCESS) {
    //     plasma_error("plasma_tune() failed");
    //     return status;
    // }

    // Set NT & KT
    nb = plasma->nb;
    // Initialize tile matrix descriptors.
    descA = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, Am, An, 0, 0, Am, An);

    descB = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, Bm, Bn, 0, 0, Bm, Bn);

    descC = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, n, n, 0, 0, n, n);

    // Allocate matrices in tile layout.
    retval = plasma_desc_mat_alloc(&descA);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_desc_mat_alloc() failed");
        return retval;
    }
    retval = plasma_desc_mat_alloc(&descB);
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
        plasma_fatal_error("plasma_sequence_create() failed");
        return retval;
    }
    // Initialize request.
    PLASMA_request request = PLASMA_REQUEST_INITIALIZER;

    #pragma omp parallel
    #pragma omp master
    {
        // The Async functions are submitted here.  If an error occurs
        //   (at submission time or at run time) the sequence->status
        //   will be marked with an error.  After an error, the next
        //   Async will not _insert_ more tasks into the runtime.  The
        //   sequence->status can be checked after each call to _Async
        //   or at the end of the parallel region.

        // Translate to tile layout.
        PLASMA_zcm2ccrb_Async(A, lda, &descA, sequence, &request);
        if (sequence->status == PLASMA_SUCCESS)
            PLASMA_zcm2ccrb_Async(B, ldb, &descB, sequence, &request);
        if (sequence->status == PLASMA_SUCCESS)
            PLASMA_zcm2ccrb_Async(C, ldc, &descC, sequence, &request);

        // Call the tile async function.
        if (sequence->status == PLASMA_SUCCESS) {
            PLASMA_zsyr2k_Tile_Async(uplo, trans,
                                     alpha, &descA,
                                     &descB, beta,
                                     &descC, sequence,
                                     &request);
        }

        // Translate back to LAPACK layout.
        if (sequence->status == PLASMA_SUCCESS)
            PLASMA_zccrb2cm_Async(&descC, C, ldc, sequence, &request);
    } // pragma omp parallel block closed

    // Check for errors in the async execution
    if (sequence->status != PLASMA_SUCCESS)
        return sequence->status;

    // Free matrices in tile layout.
    plasma_desc_mat_free(&descA);
    plasma_desc_mat_free(&descB);
    plasma_desc_mat_free(&descC);

    // Return status.
    status = sequence->status;
    plasma_sequence_destroy(sequence);
    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_syr2k
 *
 *  Performs rank 2k update.
 *  Non-blocking tile version of PLASMA_zsyr2k().
 *  May return before the computation is finished.
 *  Operates on matrices stored by tiles.
 *  All matrices are passed through descriptors.
 *  All dimensions are taken from the descriptors.
 *  Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of C is stored;
 *          - PlasmaLower: Lower triangle of C is stored.
 *
 * @param[in] trans
 *          - PlasmaNoTrans:
 *            \f[ C = \alpha A \times B^T + \alpha B \times A^T + \beta C \f];
 *          - PlasmaTrans:
 *            \f[ C = \alpha A^T \times B + \alpha B^T \times A + \beta C \f].
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          Descriptor of matrix A.
 *
 *@param[in] B
 *          Descriptor of matrix B.
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
 * @sa PLASMA_zsyr2k
 * @sa PLASMA_zsyr2k_Tile_Async
 * @sa PLASMA_csyr2k_Tile_Async
 *
 ******************************************************************************/
void PLASMA_zsyr2k_Tile_Async(PLASMA_enum uplo, PLASMA_enum trans,
                              PLASMA_Complex64_t alpha, PLASMA_desc *A,
                                                        PLASMA_desc *B,
                              PLASMA_Complex64_t beta,  PLASMA_desc *C,
                              PLASMA_sequence *sequence, PLASMA_request *request)
{
    PLASMA_Complex64_t zzero = 0.0;
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    // Check input arguments.
    if ((uplo != PlasmaUpper) && (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if ((trans != PlasmaNoTrans) && (trans != PlasmaTrans)) {
        plasma_error("illegal value of trans");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (plasma_desc_check(A) != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        plasma_error("invalid A");
        return;
    }

    if (plasma_desc_check(B) != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        plasma_error("invalid B");
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

    int Am, An, Amb;

    if (trans == PlasmaNoTrans) {
        Am  = A->m;
        An  = A->n;
        Amb = A->mb;
    }
    else {
        Am  = A->n;
        An  = A->m;
        Amb = A->nb;
    }

    if (C->mb != C->nb) {
        plasma_error("only square tiles for C are supported");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if ((B->mb != A->mb) || (B->nb != A->nb) || (Amb != C->mb)) {
        plasma_error("tile sizes mismatch");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (C->m != C->n) {
        plasma_error("only square matrix C is supported");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if ((B->m != A->m) || (B->n != A->n) || (Am != C->m)) {
        plasma_error("matrix sizes mismatch");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    // Check sequence status.
    if (sequence->status != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);
        return;
    }

    // quick return
    if (C->m == 0 || ((alpha == zzero || An == 0) && beta == (double)1.0))
        return;

    // Call the parallel function.
    plasma_pzsyr2k(uplo, trans,
                   alpha, *A,
                          *B,
                    beta, *C,
                   sequence, request);
    return;
}
