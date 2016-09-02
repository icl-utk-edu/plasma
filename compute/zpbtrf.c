/**
 *
 * @file zpbtrf.c
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
 *          or the number of suuperdiagonals if uplo=lower. ku >= 0.
 *
 * @param[in,out] AB
 *          On entry, the upper or lower triangle of the Hermitian band
 *          matrix A, stored in the first KD+1 rows of the array.  The
 *          j-th column of A is stored in the j-th column of the array AB
 *          as follows:
 *          if UPLO = 'U', AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd)<=i<=j;
 *          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+kd).
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
 * @retval PLASMA_SUCCESS successful exit
 * @retval  < 0 if -i, the i-th argument had an illegal value
 * @retval  > 0 if i, the leading minor of order i of A is not
 *          positive definite, so the factorization could not
 *          be completed, and the solution has not been computed.
 *
 *******************************************************************************
 *
 * @sa PLASMA_zpbtrf_Tile_Async
 * @sa PLASMA_cpbtrf
 * @sa PLASMA_dpbtrf
 * @sa PLASMA_spbtrf
 *
 ******************************************************************************/
int PLASMA_zpbtrf(PLASMA_enum uplo,
                  int n, int kd,
                  PLASMA_Complex64_t *AB, int ldab)
{
    int nb;
    int retval;
    int status;

    PLASMA_desc descAB;

    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }

    // Check input arguments
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
        return PLASMA_SUCCESS;

    // Tune
    // status = plasma_tune(PLASMA_FUNC_ZGBSV, N, N, 0);
    // if (status != PLASMA_SUCCESS) {
    //     plasma_error("plasma_tune() failed");
    //     return status;
    // }

    nb = plasma->nb;
    // Initialize tile matrix descriptors.
    int lda = nb*(1+(kd+nb-1)/nb);
    descAB = plasma_desc_band_init(PlasmaComplexDouble, uplo, nb, nb,
                                   nb*nb, lda, n, 0, 0, n, n, kd, kd);

    // Allocate matrices in tile layout.
    retval = plasma_desc_mat_alloc(&descAB);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_desc_mat_alloc() failed");
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

    // The Async functions are submitted here.  If an error occurs
    // (at submission time or at run time) the sequence->status
    // will be marked with an error.  After an error, the next
    // Async will not _insert_ more tasks into the runtime.  The
    // sequence->status can be checked after each call to _Async
    // or at the end of the parallel region.
    //
    // Storage translation and factorization are split because
    // LU panel/pivot need synch on tiles in each column from/to
    // translation
    #pragma omp parallel
    #pragma omp master
    {
        // Translate to tile layout.
        PLASMA_zcm2ccrb_band_Async(uplo, AB, ldab, &descAB, sequence, &request);

        // Call the tile async function.
        if (sequence->status == PLASMA_SUCCESS) {
            PLASMA_zpbtrf_Tile_Async(uplo, &descAB, sequence, &request);
        }

        // Translate back to LAPACK layout.
        if (sequence->status == PLASMA_SUCCESS)
            PLASMA_zccrb2cm_band_Async(uplo, &descAB, AB, ldab, sequence, &request);
    } // pragma omp parallel block closed

    // Free matrix A in tile layout.
    plasma_desc_mat_free(&descAB);

    // Return status.
    status = sequence->status;
    plasma_sequence_destroy(sequence);
    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_pbtrf
 *
 *  Performs the Cholesky factorization of a Hermitian positive definite
 *  matrix.
 *  Non-blocking tile version of PLASMA_zpbtrf().
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
 *          request->status should never be set to PLASMA_SUCCESS (the
 *          initial values) since another async call may be setting a
 *          failure value at the same time.
 *
 *******************************************************************************
 *
 * @sa PLASMA_zpbtrf
 * @sa PLASMA_zpbtrf_Tile_Async
 * @sa PLASMA_cpbtrf_Tile_Async
 * @sa PLASMA_dpbtrf_Tile_Async
 * @sa PLASMA_spbtrf_Tile_Async
 *
 ******************************************************************************/
void PLASMA_zpbtrf_Tile_Async(PLASMA_enum uplo,
                              PLASMA_desc *AB,
                              PLASMA_sequence *sequence,
                              PLASMA_request *request)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    // Check input arguments.
    if (plasma_desc_band_check(uplo, AB) != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        plasma_error("invalid A");
        return;
    }
    if (sequence == NULL) {
        plasma_fatal_error("NULL sequence");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (request == NULL) {
        plasma_fatal_error("NULL request");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    if (AB->mb != AB->nb) {
        plasma_error("only square tiles supported");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    // Check sequence status.
    if (sequence->status != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);
        return;
    }

    // quick return
    if (AB->m == 0)
        return;

    // Call the parallel function.
    plasma_pzpbtrf(uplo, *AB, sequence, request);

    return;
}
