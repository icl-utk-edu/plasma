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
 * @ingroup plasma_lantr
 *
 *  Returns the norm of a trapezoidal or triangular matrix as
 *
 *     zlantr = ( max(abs(A(i,j))), NORM = PlasmaMaxNorm
 *              (
 *              ( norm1(A),         NORM = PlasmaOneNorm
 *              (
 *              ( normI(A),         NORM = PlasmaInfNorm
 *              (
 *              ( normF(A),         NORM = PlasmaFrobeniusNorm
 *
 *  where norm1 denotes the one norm of a matrix (maximum column sum),
 *  normI denotes the infinity norm of a matrix (maximum row sum) and
 *  normF denotes the Frobenius norm of a matrix (square root of sum
 *  of squares). Note that max(abs(A(i,j))) is not a consistent matrix
 *  norm.
 *
 *******************************************************************************
 *
 * @param[in] norm
 *          - PlasmaMaxNorm: max norm
 *          - PlasmaOneNorm: one norm
 *          - PlasmaInfNorm: infinity norm
 *          - PlasmaFrobeniusNorm: Frobenius norm
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of A is stored;
 *          - PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] diag
 *          - PlasmaNonUnit: A has non-unit diagonal,
 *          - PlasmaUnit:    A has unit diagonal.
 *
 * @param[in] m
 *          The number of rows of the matrix A. m >= 0. When m = 0,
 *          the returned value is set to zero.
 *
 * @param[in] n
 *          The number of columns of the matrix A. n >= 0. When n = 0,
 *          the returned value is set to zero.
 *
 * @param[in] pA
 *          The m-by-n trapezoidal matrix A.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,m).
 *
 *******************************************************************************
 *
 * @retval double
 *         The specified norm of the trapezoidal or triangular matrix A.
 *
 *******************************************************************************
 *
 * @sa plasma_omp_zlantr
 * @sa plasma_clantr
 * @sa plasma_dlantr
 * @sa plasma_slantr
 *
 ******************************************************************************/
double plasma_zlantr(plasma_enum_t norm, plasma_enum_t uplo, plasma_enum_t diag,
                     int m, int n,
                     plasma_complex64_t *pA, int lda)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    // Check input arguments.
    if ((norm != PlasmaMaxNorm) && (norm != PlasmaOneNorm) &&
        (norm != PlasmaInfNorm) && (norm != PlasmaFrobeniusNorm) ) {
        plasma_error("illegal value of norm");
        return -1;
    }
    if ((uplo != PlasmaUpper) &&
        (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo");
        return -2;
    }
    if ((diag != PlasmaUnit) &&
        (diag != PlasmaNonUnit)) {
        plasma_error("illegal value of diag");
        return -3;
    }
    if (m < 0) {
        plasma_error("illegal value of m");
        return -4;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -5;
    }
    if (lda < imax(1, m)) {
        printf("%d\n", lda);
        plasma_error("illegal value of lda");
        return -7;
    }

    // quick return
    if (imin(n, m) == 0)
      return 0.0;

    // Tune parameters.
    if (plasma->tuning)
        plasma_tune_lantr(plasma, PlasmaComplexDouble, m, n);


    // Set tiling parameters.
    int nb = plasma->nb;

    // Create tile matrices.
    plasma_desc_t A;
    int retval;
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        m, n, 0, 0, m, n, &A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        return retval;
    }

    // Allocate workspace.
    double *work = NULL;
    switch (norm) {
    case PlasmaMaxNorm:
        work = (double*)malloc((size_t)A.mt*A.nt*sizeof(double));
        break;
    case PlasmaOneNorm:
        work = (double*)calloc(((size_t)A.mt*A.n+A.n), sizeof(double));
        break;
    case PlasmaInfNorm:
        work = (double*)calloc(((size_t)A.nt*A.m+A.m), sizeof(double));
        break;
    case PlasmaFrobeniusNorm:
        work = (double*)calloc((size_t)2*A.mt*A.nt, sizeof(double));
        break;
    }
    if (work == NULL) {
        plasma_error("malloc() failed");
        return PlasmaErrorOutOfMemory;
    }

    // Initialize sequence.
    plasma_sequence_t sequence;
    retval = plasma_sequence_init(&sequence);

    // Initialize request.
    plasma_request_t request;
    retval = plasma_request_init(&request);

    double value;

    // asynchronous block
    #pragma omp parallel
    #pragma omp master
    {
        // Translate to tile layout.
        plasma_omp_zge2desc(pA, lda, A, &sequence, &request);

        // Call tile async function.
        plasma_omp_zlantr(norm, uplo, diag, A, work, &value,
                          &sequence, &request);
    }
    // implicit synchronization

    free(work);

    // Free matrix in tile layout.
    plasma_desc_destroy(&A);

    // Return the norm.
    return value;
}

/***************************************************************************//**
 *
 * @ingroup plasma_lantr
 *
 *  Calculates the max, one, infinity or Frobenius norm of a general matrix.
 *  Non-blocking equivalent of plasma_zlantr(). May return before the
 *  computation is finished. Operates on matrices stored by tiles. All matrices
 *  are passed through descriptors. All dimensions are taken from the
 *  descriptors. Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] norm
 *          - PlasmaMaxNorm: Max norm
 *          - PlasmaOneNorm: One norm
 *          - PlasmaInfNorm: Infinity norm
 *          - PlasmaFrobeniusNorm: Frobenius norm
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of A is stored;
 *          - PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] diag
 *          - PlasmaNonUnit: A has non-unit diagonal,
 *          - PlasmaUnit:    A has unit diagonal.
 *
 * @param[in] A
 *          The descriptor of matrix A.
 *
 * @param[out] work
 *          Workspace of size:
 *          - PlasmaMaxNorm: A.mt*A.nt
 *          - PlasmaOneNorm: A.mt*A.n + A.n
 *          - PlasmaInfNorm: A.mt*A.n + A.n
 *          - PlasmaFrobeniusNorm: 2*A.mt*A.nt
 *
 * @param[out] value
 *          The calculated value of the norm requested.
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
 *          request->status to error values. The sequence->status and
 *          request->status should never be set to PlasmaSuccess (the
 *          initial values) since another async call may be setting a
 *          failure value at the same time.
 *
 *******************************************************************************
 *
 * @sa plasma_zlantr
 * @sa plasma_omp_clantr
 * @sa plasma_omp_dlantr
 * @sa plasma_omp_slantr
 *
 ******************************************************************************/
void plasma_omp_zlantr(plasma_enum_t norm, plasma_enum_t uplo,
                       plasma_enum_t diag, plasma_desc_t A,
                       double *work, double *value,
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
    if ((norm != PlasmaMaxNorm) && (norm != PlasmaOneNorm) &&
        (norm != PlasmaInfNorm) && (norm != PlasmaFrobeniusNorm)) {
        plasma_error("illegal value of norm");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
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
        plasma_error("invalid descriptor A");
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
    if (imin(A.m, A.n) == 0) {
        *value = 0.0;
        return;
    }

    // Call the parallel function.
    plasma_pzlantr(norm, uplo, diag, A, work, value, sequence, request);
}
