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
 * @ingroup plasma_lacpy
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
 * @retval PlasmaSuccess successful exit
 *
 *******************************************************************************
 *
 * @sa plasma_omp_zlacpy
 * @sa PLASMA_clacpy
 * @sa PLASMA_dlacpy
 * @sa PLASMA_slacpy
 *
 ******************************************************************************/
int PLASMA_zlacpy(plasma_enum_t uplo, int m, int n,
                  plasma_complex64_t *A, int lda,
                  plasma_complex64_t *B, int ldb)
{
    int nb;
    int retval;
    int status;
    plasma_context_t  *plasma;
    plasma_sequence_t *sequence = NULL;
    plasma_request_t   request  = PLASMA_REQUEST_INITIALIZER;
    plasma_desc_t descA, descB;

    // Get PLASMA context
    plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
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
        plasma_error("illegal value of n");
        return -3;
    }

    if (lda < imax(1, m)) {
        plasma_error("illegal value of lda");
        return -5;
    }

    if (ldb < imax(1, m)) {
        plasma_error("illegal value of ldb");
        return -7;
    }

    // Quick return
    if (imin(n, m) == 0)
      return PlasmaSuccess;

    // Tune
    // if (plasma_tune(PLASMA_FUNC_ZLACPY, m, n, 0) != PlasmaSuccess) {
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
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_mat_alloc() failed");
        return retval;
    }

    retval = plasma_desc_mat_alloc(&descB);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_mat_alloc() failed");
        plasma_desc_mat_free(&descA);
        return retval;
    }

    // Create sequence
    retval = plasma_sequence_create(&sequence);
    if (retval != PlasmaSuccess) {
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
        if (sequence->status == PlasmaSuccess) {
            plasma_omp_zlacpy(uplo, &descA, &descB, sequence, &request);
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
 * @ingroup plasma_lacpy
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
 *          request->status should never be set to PlasmaSuccess (the
 *          initial values) since another async call may be setting a
 *          failure value at the same time.
 *
 *******************************************************************************
 *
 * @sa PLASMA_zlacpy
 * @sa PLASMA_omp_clacpy
 * @sa PLASMA_omp_dlacpy
 * @sa PLASMA_omp_slacpy
 *
 ******************************************************************************/
void plasma_omp_zlacpy(plasma_enum_t uplo, plasma_desc_t *A, plasma_desc_t *B,
                       plasma_sequence_t *sequence, plasma_request_t *request)
{
    plasma_desc_t descA;
    plasma_desc_t descB;
    plasma_context_t *plasma;

    // Get PLASMA context
    plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }

    // Check input arguments
    if ((uplo != PlasmaFull)  &&
        (uplo != PlasmaUpper) &&
        (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }

    // Check descriptors for correctness
    if (plasma_desc_check(A) != PlasmaSuccess) {
        plasma_error("invalid A");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    else {
        descA = *A;
    }

    if (plasma_desc_check(B) != PlasmaSuccess) {
        plasma_error("invalid B");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    else {
        descB = *B;
    }

    if (descA.nb != descA.mb) {
        plasma_error("only square tiles supported");
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

    // Quick return
    if (imin(descA.m, descA.n) == 0)
        return;

    // Call parallel function
    plasma_pzlacpy(uplo, descA, descB, sequence, request);
}
