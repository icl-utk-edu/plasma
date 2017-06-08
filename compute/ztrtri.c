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
#include "plasma_workspace.h"

/***************************************************************************//**
 *
 * @ingroup plasma_trtri
 *
 *  Computes the inverse of an upper or lower triangular matrix A.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of A is stored;
 *          - PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] diag
 *          - PlasmaNonUnit: A is non-unit triangular;
 *          - PlasmaUnit:    A is unit triangular.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in,out] pA
 *          On entry, the triangular matrix A.  If uplo = 'U', the
 *          leading n-by-n upper triangular part of the array A
 *          contains the upper triangular matrix, and the strictly
 *          lower triangular part of A is not referenced.  If uplo =
 *          'L', the leading n-by-n lower triangular part of the array
 *          A contains the lower triangular matrix, and the strictly
 *          upper triangular part of A is not referenced.  If diag =
 *          'U', the diagonal elements of A are also not referenced and
 *          are assumed to be 1.  On exit, the (triangular) inverse of
 *          the original matrix, in the same storage format.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,n).
 *
 *******************************************************************************
 *
 * @retval PlasmaSuccess successful exit.
 * @retval < 0 if -i, the i-th argument had an illegal value.
 * @retval > 0 if i, A(i,i) is exactly zero.  The triangular
 *         matrix is singular and its inverse can not be computed.
 *
 *******************************************************************************
 *
 * @sa plasma_ctrtri
 * @sa plasma_dtrtri
 * @sa plasma_strtri
 *
 ******************************************************************************/
int plasma_ztrtri(plasma_enum_t uplo, plasma_enum_t diag,
                  int n, plasma_complex64_t *pA, int lda)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    // Check input arguments.
    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        plasma_error("illegal value of uplo");
        return -1;
    }
    if (diag != PlasmaUnit && diag != PlasmaNonUnit) {
        plasma_error("illegal value of diag");
        return -2;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -3;
    }
    if (lda < imax(1, n)) {
        plasma_error("illegal value of lda");
        return -5;
    }
    // quick return
    if (imax(n, 0) == 0)
        return PlasmaSuccess;

    // Tune parameters.
    if (plasma->tuning)
        plasma_tune_trtri(plasma, PlasmaComplexDouble, n);

    // Set tiling parameters.
    int nb = plasma->nb;

    // Create tile matrix.
    plasma_desc_t A;
    int retval;
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        n, n, 0, 0, n, n, &A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
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

        // Call the tile async function.
        plasma_omp_ztrtri(uplo, diag, A, &sequence, &request);

        // Translate back to LAPACK layout.
        plasma_omp_zdesc2ge(A, pA, lda, &sequence, &request);
    }
    // implicit synchronization

    // Free matrix A in tile layout.
    plasma_desc_destroy(&A);

    // Return status.
    int status = sequence.status;
    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_trtri
 *
 *  Computes the inverse of a complex upper or lower triangular matrix A.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of A is stored;
 *          - PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] diag
 *          - PlasmaNonUnit: A is non-unit triangular;
 *          - PlasmaUnit:    A is unit triangular.
 *
 * @param[in] A
 *          On entry, the triangular matrix A.  If uplo = 'U', the
 *          leading n-by-n upper triangular part of the array A
 *          contains the upper triangular matrix, and the strictly
 *          lower triangular part of A is not referenced.  If uplo =
 *          'L', the leading n-by-n lower triangular part of the array
 *          A contains the lower triangular matrix, and the strictly
 *          upper triangular part of A is not referenced.  If diag =
 *          'U', the diagonal elements of A are also not referenced and
 *          are assumed to be 1.  On exit, the (triangular) inverse of
 *          the original matrix, in the same storage format.
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
 *          request->status should never be set to PlasmaSuccess (the
 *          initial values) since another async call may be setting a
 *          failure value at the same time.
 *
 *******************************************************************************
 *
 * @sa plasma_ztrtri
 * @sa plasma_omp_ztrtri
 * @sa plasma_omp_ctrtri
 * @sa plasma_omp_dtrtri
 * @sa plasma_omp_strtri
 *
 ******************************************************************************/
void plasma_omp_ztrtri(plasma_enum_t uplo, plasma_enum_t diag,
                       plasma_desc_t A,
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
    if ((uplo != PlasmaUpper) &&
        (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if ((diag != PlasmaUnit) &&
        (diag != PlasmaNonUnit)) {
        plasma_error("illegal value of diag");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (plasma_desc_check(A) != PlasmaSuccess) {
        plasma_error("invalid A");
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
    if (A.n == 0)
        return;

    // Call the parallel function.
    plasma_pztrtri(uplo, diag, A, sequence, request);
}
