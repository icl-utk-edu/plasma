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

/***************************************************************************//**
 *
 * @ingroup plasma_lacpy
 *
 *  Copies general rectangular or upper or lower triangular part of
 *  a two-dimensional m-by-n matrix A to another m-by-n matrix B.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          Specifies the part of the matrix A to be copied to B.
 *            - PlasmaGeneral: General rectangular matrix A
 *            - PlasmaUpper:   Upper triangular part of A
 *            - PlasmaLower:   Lower triangular part of A
 *
 * @param[in] transa
 *          - PlasmaNoTrans:   A is not transposed,
 *          - PlasmaTrans:     A is transposed,
 *          - PlasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] m
 *          The number of rows of the matrix A. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix A. n >= 0.
 *
 * @param[in] pA
 *          The m-by-n matrix A. If uplo = PlasmaUpper, only the upper trapezium
 *          is accessed; if uplo = PlasmaLower, only the lower trapezium is
 *          accessed.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,m).
 *
 * @param[out] pB
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
 * @sa plasma_clacpy
 * @sa plasma_dlacpy
 * @sa plasma_slacpy
 *
 ******************************************************************************/
int plasma_zlacpy(plasma_enum_t uplo, plasma_enum_t transa,
                  int m, int n,
                  plasma_complex64_t *pA, int lda,
                  plasma_complex64_t *pB, int ldb)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    // Check input arguments.
    if ((uplo != PlasmaGeneral) &&
        (uplo != PlasmaUpper)   &&
        (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo");
        return -1;
    }
    if ((transa != PlasmaNoTrans) &&
        (transa != PlasmaTrans) &&
        (transa != PlasmaConjTrans)) {
        plasma_error("illegal value of transa");
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
    if (transa != PlasmaNoTrans && m != n) {
        plasma_error("illegal value of m and n");
        return -3;
    }
    if (lda < imax(1, m)) {
        plasma_error("illegal value of lda");
        return -6;
    }
    if (ldb < imax(1, (transa == PlasmaNoTrans ? m : n))) {
        plasma_error("illegal value of ldb");
        return -8;
    }

    // quick return
    if (imin(n, m) == 0)
      return PlasmaSuccess;

    // Tune parameters.
    if (plasma->tuning)
        plasma_tune_lacpy(plasma, PlasmaComplexDouble, m, n);

    // Set tiling parameters.
    int nb = plasma->nb;

    // Create tile matrices.
    plasma_desc_t A, B;
    int retval;
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        m, n, 0, 0, m, n, &A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_general_desc_create() failed");
        return retval;
    }
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        m, n, 0, 0, m, n, &B);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_general_desc_create() failed");
        plasma_desc_destroy(&A);
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
        plasma_omp_zge2desc(pB, ldb, B, &sequence, &request);

        // Call tile async function.
        plasma_omp_zlacpy(uplo, transa, A, B, &sequence, &request);

        // Translate back to LAPACK layout.
        plasma_omp_zdesc2ge(B, pB, ldb, &sequence, &request);
    }
    // implicit synchronization

    // Free matrices in tile layout.
    plasma_desc_destroy(&A);
    plasma_desc_destroy(&B);

    // Return status.
    int status = sequence.status;
    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_lacpy
 *
 *  Copies general rectangular or upper or lower triangular part of
 *  a two-dimensional m-by-n matrix A to another m-by-n matrix B. Non-blocking
 *  tile version of plasma_zlacpy(). May return before the computation is
 *  finished. Operates on matrices stored by tiles. All matrices are passed
 *  through descriptors. All dimensions are taken from the descriptors. Allows
 *  for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          Specifies the part of the matrix A to be copied to B.
 *            - PlasmaGeneral: General rectangular matrix A
 *            - PlasmaUpper:   Upper triangular part of A
 *            - PlasmaLower:   Lower triangular part of A
 *
 * @param[in] transa
 *          - PlasmaNoTrans:   A is not transposed,
 *          - PlasmaTrans:     A is transposed,
 *          - PlasmaConjTrans: A is conjugate transposed.
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
 * @sa plasma_zlacpy
 * @sa plasma_omp_clacpy
 * @sa plasma_omp_dlacpy
 * @sa plasma_omp_slacpy
 *
 ******************************************************************************/
void plasma_omp_zlacpy(plasma_enum_t uplo, plasma_enum_t transa,
                       plasma_desc_t A, plasma_desc_t B,
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
    if ((uplo != PlasmaGeneral) &&
        (uplo != PlasmaUpper)   &&
        (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if ((transa != PlasmaNoTrans) &&
        (transa != PlasmaTrans) &&
        (transa != PlasmaConjTrans)) {
        plasma_error("illegal value of transa");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (plasma_desc_check(A) != PlasmaSuccess) {
        plasma_error("invalid A");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (plasma_desc_check(B) != PlasmaSuccess) {
        plasma_error("invalid B");
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
    if (imin(A.m, A.n) == 0)
        return;

    // Call the parallel function.
    plasma_pzlacpy(uplo, transa, A, B, sequence, request);
}
