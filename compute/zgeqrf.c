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
 * @ingroup PLASMA_Complex64_t
 *
 *  Computes the tile QR factorization of a real or complex m-by-n matrix A.
 *  The factorization has the form
 *    \f[ A = Q \times R \f],
 *  where Q is a matrix with orthonormal columns and R is an upper triangular
 *  with positive diagonal.
 *
 *******************************************************************************
 *
 * @param[in] m
 *          The number of rows of the matrix A.
 *          m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix A.
 *          n >= 0.
 *
 * @param[in,out] A
 *          On entry, the m-by-n matrix A.
 *          On exit, the elements on and above the diagonal of the array contain
 *          the min(m,n)-by-n upper trapezoidal matrix R (R is upper triangular
 *          if m >= n); the elements below the diagonal represent the unitary
 *          matrix Q as a product of elementary reflectors stored by tiles.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,m).
 *
 * @param[out] descT
 *          On exit, auxiliary factorization data, required by PLASMA_zgeqrs to
 *          solve the system of equations.
 *
 *******************************************************************************
 *
 * @retval PLASMA_SUCCESS successful exit
 * @retval < 0 if -i, the i-th argument had an illegal value
 *
 *******************************************************************************
 *
 * @sa PLASMA_zgeqrf_Tile_Async
 * @sa PLASMA_cgeqrf
 * @sa PLASMA_dgeqrf
 * @sa PLASMA_sgeqrf
 * @sa PLASMA_zgeqrs
 * @sa PLASMA_zgels
 *
 ******************************************************************************/
int PLASMA_zgeqrf(int m, int n,
                  PLASMA_Complex64_t *A, int lda,
                  PLASMA_desc *descT)
{
    int nb;
    int retval;
    int status;

    PLASMA_desc descA;

    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }

    // Check input arguments.
    if (m < 0) {
        plasma_error("illegal value of m");
        return -1;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -2;
    }
    if (lda < imax(1, m)) {
        plasma_error("illegal value of lda");
        return -4;
    }

    // quick return
    if (imin(m, n) == 0)
        return PLASMA_SUCCESS;

    // Tune NB & IB depending on M, N & NRHS; Set NBNBSIZE
    //status = plasma_tune(PLASMA_FUNC_ZGELS, M, N, 0);
    //if (status != PLASMA_SUCCESS) {
    //    plasma_error("PLASMA_zgeqrf", "plasma_tune() failed");
    //    return status;
    //}
    nb = plasma->nb;

    // Initialize tile matrix descriptor.
    descA = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, m, n, 0, 0, m, n);

    // Allocate matrices in tile layout.
    retval = plasma_desc_mat_alloc(&descA);
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

    #pragma omp parallel
    #pragma omp master
    {
        // The Async functions are submitted here.  If an error occurs
        // (at submission time or at run time) the sequence->status
        // will be marked with an error.  After an error, the next
        // Async will not _insert_ more tasks into the runtime.  The
        // sequence->status can be checked after each call to _Async
        // or at the end of the parallel region.

        // Translate to tile layout.
        PLASMA_zcm2ccrb_Async(A, lda, &descA, sequence, &request);

        // Call the tile async function.
        if (sequence->status == PLASMA_SUCCESS) {
            PLASMA_zgeqrf_Tile_Async(&descA, descT, sequence, &request);
        }

        // Translate back to LAPACK layout.
        if (sequence->status == PLASMA_SUCCESS)
            PLASMA_zccrb2cm_Async(&descA, A, lda, sequence, &request);
    } // pragma omp parallel block closed

    // Free matrix A in tile layout.
    plasma_desc_mat_free(&descA);

    // Return status.
    status = sequence->status;
    plasma_sequence_destroy(sequence);
    return status;
}

/***************************************************************************//**
 *
 * @ingroup PLASMA_Complex64_t
 *
 *  Computes the tile QR factorization of a matrix.
 *  Non-blocking tile version of PLASMA_zgeqrf().
 *  May return before the computation is finished.
 *  Operates on matrices stored by tiles.
 *  All matrices are passed through descriptors.
 *  All dimensions are taken from the descriptors.
 *  Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in,out] descA
 *          Descriptor of matrix A.
 *          A is stored in the tile layout.
 *
 * @param[out] descT
 *          Descriptor of matrix descT.
 *          On exit, auxiliary factorization data, required by PLASMA_zgeqrs to
 *          solve the system of equations.
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
 *          request->status to error values.  The sequence->status and
 *          request->status should never be set to PLASMA_SUCCESS (the
 *          initial values) since another async call may be setting a
 *          failure value at the same time.
 *
 *******************************************************************************
 *
 * @sa PLASMA_zgeqrf
 * @sa PLASMA_cgeqrf_Tile_Async
 * @sa PLASMA_dgeqrf_Tile_Async
 * @sa PLASMA_sgeqrf_Tile_Async
 * @sa PLASMA_zgeqrs_Tile_Async
 * @sa PLASMA_zgeqrs_Tile_Async
 * @sa PLASMA_zgels_Tile_Async
 *
 ******************************************************************************/
void PLASMA_zgeqrf_Tile_Async(PLASMA_desc *descA, PLASMA_desc *descT,
                              PLASMA_sequence *sequence,
                              PLASMA_request *request)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    // Check input arguments.
    if (plasma_desc_check(descA) != PLASMA_SUCCESS) {
        plasma_error("invalid A");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (plasma_desc_check(descT) != PLASMA_SUCCESS) {
        plasma_error("invalid T");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
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
    if (descA->nb != descA->mb) {
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
    // Jakub S.: Why was it commented out in version 2.8.0 ?
    // I leave it like that till explained.
    //if (imin(m, n) == 0)
    //    return PLASMA_SUCCESS;

    // Call the parallel function.
    plasma_pzgeqrf(*descA, *descT, sequence, request);

    return;
}
