/**
 *
 * @file zhemm.c
 *
 *  PLASMA computational routine.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of Manchester, Univ. of California Berkeley and
 *  Univ. of Colorado Denver
 *
 * @version 3.0.0
 * @author Samuel D. Relton
 * @date 2016-05-17
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
 * @ingroup plasma_hemm
 *
 *  Performs one of the matrix-matrix operations
 *
 *     \f[ C = \alpha \times A \times B + \beta \times C \f]
 *  or
 *     \f[ C = \alpha \times B \times A + \beta \times C \f]
 *
 *  where alpha and beta are scalars, A is an hemmetric matrix and  B and
 *  C are m by n matrices.
 *
 *******************************************************************************
 *
 * @param[in] side
 *          Specifies whether the hemmetric matrix A appears on the
 *          left or right in the operation as follows:
 *          - PlasmaLeft:      \f[ C = \alpha \times A \times B + \beta \times C \f]
 *          - PlasmaRight:     \f[ C = \alpha \times B \times A + \beta \times C \f]
 *
 * @param[in] uplo
 *          Specifies whether the upper or lower triangular part of
 *          the hemmetric matrix A is to be referenced as follows:
 *          - PlasmaLower:     Only the lower triangular part of the
 *                             hemmetric matrix A is to be referenced.
 *          - PlasmaUpper:     Only the upper triangular part of the
 *                             hemmetric matrix A is to be referenced.
 *
 * @param[in] m
 *          Specifies the number of rows of the matrix C. M >= 0.
 *
 * @param[in] n
 *          Specifies the number of columns of the matrix C. N >= 0.
 *
 * @param[in] alpha
 *          Specifies the scalar alpha.
 *
 * @param[in] A
 *          A is a LDA-by-ka matrix, where ka is M when side = PlasmaLeft,
 *          and is N otherwise. Only the uplo triangular part is referenced.
 *
 * @param[in] lda
 *          The leading dimension of the array A. LDA >= max(1,ka).
 *
 * @param[in] B
 *          B is a LDB-by-N matrix, where the leading M-by-N part of
 *          the array B must contain the matrix B.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. LDB >= max(1,M).
 *
 * @param[in] beta
 *          Specifies the scalar beta.
 *
 * @param[in,out] C
 *          C is a LDC-by-N matrix.
 *          On exit, the array is overwritten by the M by N updated matrix.
 *
 * @param[in] ldc
 *          The leading dimension of the array C. LDC >= max(1,M).
 *
 *******************************************************************************
 *
 * @retval  PLASMA_SUCCESS successful exit
 *
 *******************************************************************************
 *
 * @sa PLASMA_zhemm_Tile_Async
 * @sa PLASMA_chemm
 * @sa PLASMA_dhemm
 * @sa PLASMA_shemm
 *
 ******************************************************************************/
int PLASMA_zhemm(PLASMA_enum side, PLASMA_enum uplo, int m, int n,
                 PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int lda,
                                           PLASMA_Complex64_t *B, int ldb,
                 PLASMA_Complex64_t beta,  PLASMA_Complex64_t *C, int ldc)
{
    int Am;
    int nb;
    int retval;
    int status;

    PLASMA_desc descA;
    PLASMA_desc descB;
    PLASMA_desc descC;

    PLASMA_Complex64_t zzero = 0.0;
    PLASMA_Complex64_t zone  = 1.0;

    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }

    // Check input arguments.
    if ( (side != PlasmaLeft) && (side != PlasmaRight) ){
        plasma_error("illegal value of side");
        return -1;
    }
    if ((uplo != PlasmaLower) && (uplo != PlasmaUpper)) {
        plasma_error("illegal value of uplo");
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
        return -7;
    }

    if (side == PlasmaLeft)
    {
        Am = m;
    }
    else
    {
        Am = n;
    }

    if (lda < imax(1, Am)) {
        plasma_error("illegal value of lda");
        return -8;
    }
    if (B == NULL) {
        plasma_error("NULL B");
        return -9;
    }
    if (ldb < imax(1, m)) {
        plasma_error("illegal value of ldb");
        return -10;
    }
    if (C == NULL) {
        plasma_error("NULL C");
        return -12;
    }
    if (ldc < imax(1, m)) {
        plasma_error("illegal value of ldc");
        return -13;
    }

    // quick return
    if (m == 0 || n == 0 || (alpha == zzero && beta == zone))
        return PLASMA_SUCCESS;

    nb = plasma->nb;

    // Initialize tile matrix descriptors.
    descA = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, Am, Am, 0, 0, Am, Am);
    descB = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, m, n, 0, 0, m, n);
    descC = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, m, n, 0, 0, m, n);

    // Allocate matrices in tile layout.
    retval = plasma_desc_mat_alloc(&descA);
    if (retval != PLASMA_SUCCESS)
    {
        plasma_error("plasma_desc_mat_alloc() failed");
        return retval;
    }
    retval = plasma_desc_mat_alloc(&descB);
    if (retval != PLASMA_SUCCESS)
    {
        plasma_error("plasma_desc_mat_alloc() failed");
        plasma_desc_mat_free(&descA);
        return retval;
    }
    retval = plasma_desc_mat_alloc(&descC);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_desc_mat_alloc() failed");
        plasma_desc_mat_free(&descA);
        plasma_desc_mat_free(&descB);
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
            PLASMA_zcm2ccrb_Async(B, ldb, &descB, sequence, &request);
        if (sequence->status == PLASMA_SUCCESS)
            PLASMA_zcm2ccrb_Async(C, ldc, &descC, sequence, &request);

        // Call the tile async function.
        if (sequence->status == PLASMA_SUCCESS) {
            PLASMA_zhemm_Tile_Async(side, uplo,
                                    alpha, &descA,
                                    &descB,
                                    beta, &descC,
                                    sequence, &request);
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
 * @ingroup plasma_hemm
 *
 *  Performs hemmetric matrix multiplication.
 *  Non-blocking equivalent of PLASMA_zhemm_Tile().
 *  May return before the computation is finished.
 *  Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] sequence
 *          Identifies the sequence of function calls that this call belongs to
 *          (for completion checks and exception handling purposes).
 *
 * @param[out] request
 *          Identifies this function call (for exception handling purposes).
 *
 *******************************************************************************
 *
 * @sa PLASMA_zhemm
 * @sa PLASMA_zhemm_Tile
 * @sa PLASMA_chemm_Tile_Async
 * @sa PLASMA_dhemm_Tile_Async
 * @sa PLASMA_shemm_Tile_Async
 *
 ******************************************************************************/
void PLASMA_zhemm_Tile_Async(PLASMA_enum side, PLASMA_enum uplo,
                             PLASMA_Complex64_t alpha, PLASMA_desc *A, PLASMA_desc *B,
                             PLASMA_Complex64_t beta,  PLASMA_desc *C,
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
    if ((side != PlasmaLeft) &&
        (side != PlasmaRight)){
        plasma_error("illegal value of side");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if ((uplo != PlasmaLower) &&
        (uplo != PlasmaUpper)) {
        plasma_error("illegal value of uplo");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (plasma_desc_check(A) != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        plasma_error("invalid A");
        return;
    }
    if (plasma_desc_check(B) != PLASMA_SUCCESS) {
        plasma_error("invalid B");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
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

    if (A->mb != C->mb || A->nb != B->nb || B->nb != C->nb) {
        plasma_error("tile size mismatch");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (side == PlasmaLeft) {
        if (A->m != B->m || A->n != B->m) {
            plasma_error("matrix size mismatch");
            plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
            return;
        }
    }
    else {
        if (A->m != B->n || A->n != B->n) {
            plasma_error("matrix size mismatch");
            plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
            return;
        }
    }
    if (B->m != C->m || B->n != C->n) {
        plasma_error("matrix size mismatch");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (A->i%A->mb != C->i%C->mb ||
        B->j%B->nb != C->j%C->nb || A->j%A->nb != B->i%B->mb) {
        plasma_error("start indexes have to match");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    // Check sequence status.
    if (sequence->status != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);
        return;
    }

    // quick return
    if (C->m == 0 || C->n == 0 || ((alpha == 0.0 || A->n == 0) && beta == 1.0))
        return;

    // Call the parallel function.
    plasma_pzhemm(side, uplo,
                  alpha, *A,
                  *B,
                  beta,  *C,
                  sequence, request);
    return;
}
