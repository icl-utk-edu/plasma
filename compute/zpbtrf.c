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
 * @ingroup plasma_pbtrf
 *
 *  Performs the Cholesky factorization of an Hermitian positive matrix A,
 *
 *    \f[ A = L \times L^T \f] or \f[ A = U^T \times U \f]
 *
 *  if uplo = upper or lower, respectively, where L is lower triangular with
 *  positive diagonal elements, and U is upper triangular.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of A is stored;
 *          - PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] n
 *          The number of columns of the matrix A. n >= 0.
 *
 * @param[in] kd
 *          The number of subdiagonals within the band of A if uplo=upper,
 *          or the number of superdiagonals if uplo=lower. kd >= 0.
 *
 * @param[in,out] AB
 *          On entry, the upper or lower triangle of the Hermitian band
 *          matrix A, stored in the first KD+1 rows of the array.  The
 *          j-th column of A is stored in the j-th column of the array AB
 *          as follows:
 *          if UPLO = 'U', AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd) <= i <= j;
 *          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j <= i <= min(n,j+kd).
 * \n
 *          On exit, if INFO = 0, the triangular factor U or L from the
 *          Cholesky factorization A = U^H*U or A = L*L^H of the band
 *          matrix A, in the same storage format as A.
 *
 * @param[in] ldab
 *          The leading dimension of the array AB. ldab >= 2*kl+ku+1.
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
 * @sa plasma_omp_zpbtrf
 * @sa plasma_cpbtrf
 * @sa plasma_dpbtrf
 * @sa plasma_spbtrf
 *
 ******************************************************************************/
int plasma_zpbtrf(plasma_enum_t uplo,
                  int n, int kd,
                  plasma_complex64_t *pAB, int ldab)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    // Check input arguments.
    if ((uplo != PlasmaUpper) &&
        (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo");
        return -1;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -2;
    }
    if (kd < 0) {
        plasma_error("illegal value of kd");
        return -3;
    }
    if (ldab < kd+1) {
        plasma_error("illegal value of ldab");
        return -5;
    }

    // quick return
    if (imax(n, 0) == 0)
        return PlasmaSuccess;

    // Tune parameters.
    if (plasma->tuning)
        plasma_tune_pbtrf(plasma, PlasmaComplexDouble, n);

    // Set tiling parameters.
    int nb = plasma->nb;

    // Initialize tile matrix descriptors.
    int lm = nb*(1+(kd+nb-1)/nb);
    plasma_desc_t AB;
    int retval;
    retval = plasma_desc_general_band_create(PlasmaComplexDouble, uplo, nb, nb,
                                             lm, n, 0, 0, n, n, kd, kd, &AB);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_band_create() failed");
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
        plasma_omp_zpb2desc(pAB, ldab, AB, &sequence, &request);

        // Call the tile async function.
        plasma_omp_zpbtrf(uplo, AB, &sequence, &request);

        // Translate back to LAPACK layout.
        plasma_omp_zdesc2pb(AB, pAB, ldab, &sequence, &request);
    }
    // implicit synchronization

    // Free matrix A in tile layout.
    plasma_desc_destroy(&AB);

    // Return status.
    int status = sequence.status;
    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_pbtrf
 *
 *  Performs the Cholesky factorization of a Hermitian positive definite
 *  matrix.
 *  Non-blocking tile version of plasma_zpbtrf().
 *  May return before the computation is finished.
 *  Operates on matrices stored by tiles.
 *  All matrices are passed through descriptors.
 *  All dimensions are taken from the descriptors.
 *  Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] AB
 *          Descriptor of matrix AB.
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
 * @sa plasma_zpbtrf
 * @sa plasma_omp_zpbtrf
 * @sa plasma_omp_cpbtrf
 * @sa plasma_omp_dpbtrf
 * @sa plasma_omp_spbtrf
 *
 ******************************************************************************/
void plasma_omp_zpbtrf(plasma_enum_t uplo, plasma_desc_t AB,
                       plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
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
    if (plasma_desc_check(AB) != PlasmaSuccess) {
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
    if (AB.m == 0)
        return;

    // Call the parallel function.
    plasma_pzpbtrf(uplo, AB, sequence, request);
}
