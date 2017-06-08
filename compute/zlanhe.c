/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> c
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
 * @ingroup plasma_lanhe
 *
 *  Returns the norm of a Hermitian matrix as
 *
 *     zlanhe = ( max(abs(A(i,j))), NORM = PlasmaMaxNorm
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
 *          - PlasmaMaxNorm: Max norm
 *          - PlasmaOneNorm: One norm
 *          - PlasmaInfNorm: Infinity norm
 *          - PlasmaFrobeniusNorm: Frobenius norm
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of A is stored;
 *          - PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in,out] pA
 *          On entry, the Hermitian matrix A.
 *          If uplo = PlasmaUpper, the leading N-by-N upper triangular part of A
 *          contains the upper triangular part of the matrix A, and the strictly
 *          lower triangular part of A is not referenced.
 *          If uplo = PlasmaLower, the leading N-by-N lower triangular part of A
 *          contains the lower triangular part of the matrix A, and the strictly
 *          upper triangular part of A is not referenced.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,m).
 *
 *******************************************************************************
 *
 * @retval double
 *         The specified norm of the Hermitian matrix A.
 *
 *******************************************************************************
 *
 * @sa plasma_omp_zlanhe
 * @sa plasma_clanhe
 *
 ******************************************************************************/
double plasma_zlanhe(plasma_enum_t norm, plasma_enum_t uplo,
                     int n,
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
    if (n < 0) {
        plasma_error("illegal value of n");
        return -3;
    }
    if (lda < imax(1, n)) {
        plasma_error("illegal value of lda");
        return -5;
    }

    // quick return
    if (n == 0)
      return 0.0;

    // Tune parameters.
    if (plasma->tuning)
        plasma_tune_lansy(plasma, PlasmaComplexDouble, n);

    // Set tiling parameters.
    int nb = plasma->nb;

    // Create tile matrices.
    plasma_desc_t A;
    int retval;
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        n, n, 0, 0, n, n, &A);
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
    case PlasmaInfNorm:
        work = (double*)malloc(((size_t)A.mt*A.n+A.n)*sizeof(double));
        break;
    case PlasmaFrobeniusNorm:
        work = (double*)malloc((size_t)2*A.mt*A.nt*sizeof(double));
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
        plasma_omp_zlanhe(norm, uplo, A, work, &value, &sequence, &request);
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
 * @ingroup plasma_lanhe
 *
 *  Calculates the max, one, infinity or Frobenius norm of a Hermitian matrix.
 *  Non-blocking equivalent of plasma_zlanhe(). May return before the
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
 * @sa plasma_zlanhe
 * @sa plasma_omp_clanhe
 *
 ******************************************************************************/
void plasma_omp_zlanhe(plasma_enum_t norm, plasma_enum_t uplo, plasma_desc_t A,
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
    if (A.m == 0) {
        *value = 0.0;
        return;
    }

    // Call the parallel function.
    plasma_pzlanhe(norm, uplo, A, work, value, sequence, request);
}
