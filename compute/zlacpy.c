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
 * @ingroup plasma_zlacpy
 *
 *  Copies full or triangular part of a two-dimensional m-by-n matrix A to
 *  another m-by-n matrix B.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          Specifies the part of the matrix A to be copied to B.
 *            - PlasmaFull:  Full rectangular matrix A
 *            - PlasmaUpper: Upper triangular part of A
 *            - PlasmaLower: Lower triangular part of A
 *
 * @param[in] m
 *          The number of rows of the matrix A. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix A. n >= 0.
 *
 * @param[in] A
 *          The m-by-n matrix A. If uplo = PlasmaUpper, only the upper trapezium
 *          is accessed; if uplo = PlasmaLower, only the lower trapezium is
 *          accessed.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,m).
 *
 * @param[out] B
 *          The m-by-n matrix B.
 *          On exit, B = A in the locations specified by uplo.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1,m).
 *
 *******************************************************************************
 *
 * @retval PLASMA_SUCCESS successful exit
 *
 *******************************************************************************
 *
 * @sa PLASMA_zlacpy_Tile_Async
 * @sa PLASMA_clacpy
 * @sa PLASMA_dlacpy
 * @sa PLASMA_slacpy
 *
 ******************************************************************************/
int PLASMA_zlacpy(PLASMA_enum uplo, int m, int n,
                  PLASMA_Complex64_t *A, int lda,
                  PLASMA_Complex64_t *B, int ldb)
{
    int nb;
    int retval;
    int status;
    plasma_context_t *plasma;
    PLASMA_sequence  *sequence = NULL;
    PLASMA_request    request  = PLASMA_REQUEST_INITIALIZER;
    PLASMA_desc descA, descB;

    // Get PLASMA context
    plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }

    // Check input arguments
    if ((uplo != PlasmaFull)  &&
        (uplo != PlasmaUpper) &&
        (uplo != PlasmaLower)) {

        plasma_error("illegal value of uplo");
        return -1;
    }

    if (m < 0) {
        plasma_error("illegal value of m");
        return -2;
    }

    if (n < 0) {
        plasma_error("illegal value of N");
        return -3;
    }

    if (lda < imax(1, m)) {
        plasma_error("illegal value of LDA");
        return -5;
    }

    if (ldb < imax(1, m)) {
        plasma_error("illegal value of LDB");
        return -7;
    }

    /* Quick return */
    if (imin(n, m) == 0)
      return PLASMA_SUCCESS;

    // Tune
    // if (plasma_tune(PLASMA_FUNC_ZGEMM, m, n, 0) != PLASMA_SUCCESS) {
    //     plasma_error("plasma_tune() failed");
    //     return status;
    // }
    nb = plasma->nb;

    // Initialize tile matrix descriptors
    descA = plasma_desc_init(PlasmaComplexDouble, nb, nb, nb*nb,
                             m, n, 0, 0, m, n);

    descB = plasma_desc_init(PlasmaComplexDouble, nb, nb, nb*nb,
                             m, n, 0, 0, m, n);

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
    retval = plasma_sequence_create(&sequence);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_sequence_create() failed");
        return retval;
    }

    // Asynchronous block
    #pragma omp parallel
    #pragma omp master
    {
        // Translate to tile layout
        PLASMA_zcm2ccrb_Async(A, lda, &descA, sequence, &request);
        PLASMA_zcm2ccrb_Async(B, ldb, &descB, sequence, &request);

        // Call tile async function
        if (sequence->status == PLASMA_SUCCESS) {
            PLASMA_zlacpy_Tile_Async(uplo, &descA, &descB, sequence, &request);
        }

        // Revert to LAPACK layout
        PLASMA_zccrb2cm_Async(&descA, A, lda, sequence, &request);
        PLASMA_zccrb2cm_Async(&descB, B, ldb, sequence, &request);

    }
    // Implicit synchronization

    // Deallocate memory in tile layout
    plasma_desc_mat_free(&descA);
    plasma_desc_mat_free(&descB);

    // Destroy sequence
    plasma_sequence_destroy(sequence);

    // Return status
    status = sequence->status;
    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_zlacpy
 *
 *  Copies full or triangular part of a two-dimensional m-by-n matrix A to
 *  another m-by-n matrix B. Non-blocking tile version of PLASMA_zlacpy(). May
 *  return before the computation is finished. Operates on matrices stored by
 *  tiles. All matrices are passed through descriptors. All dimensions are
 *  taken from the descriptors. Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          Specifies the part of the matrix A to be copied to B.
 *            - PlasmaFull:  Full rectangular matrix A
 *            - PlasmaUpper: Upper triangular part of A
 *            - PlasmaLower: Lower triangular part of A
 *
 * @param[in] A
 *          Descriptor of matrix A.
 *
 * @param[out] B
 *          Descriptor of matrix B.
 *
 * @param[in] sequence
 *          Identifies the sequence of function calls that this call belongs to
 *          (for completion checks and exception handling purposes). Check the
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
 * @sa PLASMA_zlacpy
 * @sa PLASMA_clacpy_Tile_Async
 * @sa PLASMA_dlacpy_Tile_Async
 * @sa PLASMA_slacpy_Tile_Async
 *
 ******************************************************************************/
void PLASMA_zlacpy_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B,
                             PLASMA_sequence *sequence, PLASMA_request *request)
{
    PLASMA_desc descA;
    PLASMA_desc descB;
    plasma_context_t *plasma;

    // Get PLASMA context
    plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    // Check input arguments
    if ((uplo != PlasmaFull)  &&
        (uplo != PlasmaUpper) &&
        (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    // Check descriptors for correctness
    if (plasma_desc_check(A) != PLASMA_SUCCESS) {
        plasma_error("invalid A");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    else {
        descA = *A;
    }

    if (plasma_desc_check(B) != PLASMA_SUCCESS) {
        plasma_error("invalid B");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    else {
        descB = *B;
    }

    if (descA.nb != descA.mb) {
        plasma_error("only square tiles supported");
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

    // Quick return
    if (imin(descA.m, descA.n) == 0)
        return;

    // Call parallel function
    plasma_pzlacpy(uplo, descA, descB, sequence, request);
}
