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
#include <string.h>

/***************************************************************************//**
 *
 * @ingroup plasma_hetrf
 *
 *  Factorize a Hermitian matrix A using a 'communication avoiding' Aasen's
 *  algorithm, followed by band LU factorization. The factorization has the form
 *
 *    \f[ A = P \times L \times T \times L^H \times P^H, \f]
 *    or
 *    \f[ A = P \times U^H \times T \times U \times P^H, \f]
 *
 *  where U is a unit-diagonal upper triangular matrix and L is a unit-diagonal
 *  lower triangular matrix, T is a band matrix, and P is a permutation matrix.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of A is stored;
 *          - PlasmaLower: Lower triangle of A is stored.
 *            TODO: only support Lower for now
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
 *          On exit, if return value = 0, the factor U or L from the Aasen's
 *          factorization A = (P*U^H)*T*(P*U^H)^H or A = (P*L)*T*(P*L)^H.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,n).
 *
 * @param[out] pT
 *          On exit, if return value = 0, the LU factors of the band matrix T.
 *
 * @param[in] ldt
 *          The leading dimension of the array T.
 *
 * @param[out] ipiv
 *          The pivot indices used by Aasen's algorithm; for 1 <= i <= min(m,n),
 *          row and column i of the matrix was interchanged with row and column ipiv(i).
 *
 * @param[out] ipiv2
 *          The pivot indices used by the band LU; for 1 <= i <= min(m,n),
 *          row and column i of the matrix was interchanged with row and column ipiv(i).
 *
 *******************************************************************************
 *
 * @retval PlasmaSuccess successful exit
 * @retval  < 0 if -i, the i-th argument had an illegal value
 * @retval  > 0 if i, the leading minor of order i of A is not
 *          positive definite, so the factorization could not
 *          be completed, and the solution has not been computed.
 *
 *******************************************************************************
 *
 * @sa plasma_omp_zhetrf
 * @sa plasma_chetrf
 * @sa plasma_dhetrf
 * @sa plasma_shetrf
 *
 ******************************************************************************/
int plasma_zhetrf(plasma_enum_t uplo,
                  int n,
                  plasma_complex64_t *pA, int lda, int *ipiv,
                  plasma_complex64_t *pT, int ldt, int *ipiv2)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    // Check input arguments.
    if (//(uplo != PlasmaUpper) &&
        (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo (Upper not supported, yet)");
        return -1;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -2;
    }
    if (lda < imax(1, n)) {
        plasma_error("illegal value of lda");
        return -4;
    }

    // quick return
    if (imax(n, 0) == 0)
        return PlasmaSuccess;

    // Tune parameters.
    if (plasma->tuning)
        plasma_tune_hetrf(plasma, PlasmaComplexDouble, n);

    // Set tiling parameters.
    int nb = plasma->nb;

    // Adjust max number of panel threads
    int max_panel_threads_gbtrf = 1;
    int max_panel_threads_hetrf = 1;
    if (plasma->max_panel_threads > 3) {
        max_panel_threads_gbtrf = 2;
    }
    max_panel_threads_hetrf = imax(1, plasma->max_panel_threads - max_panel_threads_gbtrf);
    plasma->max_panel_threads  = max_panel_threads_hetrf;

    // Initialize barrier
    plasma_barrier_init(&plasma->barrier);

    // Create tile matrix.
    plasma_desc_t A;
    plasma_desc_t T;
    plasma_desc_t W;
    int retval;
    retval = plasma_desc_triangular_create(PlasmaComplexDouble, uplo, nb, nb,
                                           n, n, 0, 0, n, n, &A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        return retval;
    }
    // band matrix (general band to prepare for band solve)
    retval = plasma_desc_general_band_create(PlasmaComplexDouble, PlasmaGeneral, nb, nb,
                                             ldt, n, 0, 0, n, n, nb, nb, &T);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_band_create() failed");
        return retval;
    }
    // workspace
    int tot = 3;
    int ldw = (1+(4+tot)*A.mt)*nb; // block column
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        ldw, nb, 0, 0, ldw, nb, &W);
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

    // Initialize data.
    memset(T.matrix, 0, ldt*n*sizeof(plasma_complex64_t));
    memset(W.matrix, 0, ldw*nb*sizeof(plasma_complex64_t));
    for (int i = 0; i < nb; i++) ipiv[i] = 1+i;

    // asynchronous block
    #pragma omp parallel
    #pragma omp master
    {
        // Translate to tile layout.
        plasma_omp_ztr2desc(pA, lda, A, &sequence, &request);
    }
    // implicit synchronization

    #pragma omp parallel
    #pragma omp master
    {
        // Call the tile async function to compute LTL^H factor of A,
        // where T is a band matrix
        plasma_omp_zhetrf(uplo, A, ipiv, T, ipiv2, W, &sequence, &request);
    }
    // implicit synchronization

    #pragma omp parallel
    #pragma omp master
    {
        // Translate back to LAPACK layout.
        plasma_omp_zdesc2tr(A, pA, lda, &sequence, &request);
        plasma_omp_zdesc2pb(T, pT, ldt, &sequence, &request);
    }
    // implicit synchronization

    // Free matrices in tile layout.
    plasma_desc_destroy(&A);
    plasma_desc_destroy(&T);
    plasma_desc_destroy(&W);

    // Return status.
    int status = sequence.status;
    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_hetrf
 *
 *  Factorize a Hermitian matrix.
 *  Non-blocking tile version of plasma_zhetrf().
 *  May return before the computation is finished.
 *  Operates on matrices stored by tiles.
 *  All matrices are passed through descriptors.
 *  All dimensions are taken from the descriptors.
 *  Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of A is stored;
 *          - PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] A
 *          On entry, the Hermitian matrix A.
 *          If uplo = PlasmaUpper, the leading N-by-N upper triangular part of A
 *          contains the upper triangular part of the matrix A, and the strictly
 *          lower triangular part of A is not referenced.
 *          If uplo = PlasmaLower, the leading N-by-N lower triangular part of A
 *          contains the lower triangular part of the matrix A, and the strictly
 *          upper triangular part of A is not referenced.
 *          On exit, if return value = 0, the factor U or L from the Cholesky
 *          factorization A = (P*U^H)*T(P*U^H)^H or A = (P*L)*T(P*L)^H.
 *
 * @param[out] T
 *          On exit, if return value = 0, the band matrix T of the factorization
 *          factorization A = (P*U^H)*T*(P*U^H)^H or A = (P*L)*T*(P*L)^H.
 *
 * @param[out] ipiv
 *          The pivot indices; for 1 <= i <= min(m,n), row and column i of the
 *          matrix was interchanged with row and column ipiv(i).
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
 * @sa plasma_zhetrf
 * @sa plasma_omp_zhetrf
 * @sa plasma_omp_chetrf
 * @sa plasma_omp_dhetrf
 * @sa plasma_omp_shetrf
 *
 ******************************************************************************/
void plasma_omp_zhetrf(plasma_enum_t uplo,
                       plasma_desc_t A, int *ipiv,
                       plasma_desc_t T, int *ipiv2,
                       plasma_desc_t W,
                       plasma_sequence_t *sequence,
                       plasma_request_t *request)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }

    // Check input arguments.
    if (//(uplo != PlasmaUpper) &&
        (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo (Upper not supported, yet)");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (plasma_desc_check(A) != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        plasma_error("invalid A");
        return;
    }
    if (sequence == NULL) {
        plasma_fatal_error("NULL sequence");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (request == NULL) {
        plasma_fatal_error("NULL request");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }

    // quick return
    if (A.m == 0)
        return;

    // Call the parallel function.
    plasma_pzhetrf_aasen(uplo, A, ipiv, T, W, sequence, request);
    plasma_pzgbtrf(T, ipiv2, sequence, request);
}
