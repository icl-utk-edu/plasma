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
 * @ingroup plasma_ungqr
 *
 *  Generates an m-by-n matrix Q with orthonormal columns, which
 *  is defined as the first n columns of a product of the elementary reflectors
 *  returned by PLASMA_zgeqrf.
 *
 *******************************************************************************
 *
 * @param[in] m
 *          The number of rows of the matrix Q. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix Q. m >= n >= 0.
 *
 * @param[in] k
 *          The number of columns of elementary tile reflectors whose product
 *          defines the matrix Q.
 *          n >= k >= 0.
 *
 * @param[in] A
 *          Details of the QR factorization of the original matrix A as returned
 *          by PLASMA_zgeqrf, where the k first columns are the reflectors.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,m).
 *
 * @param[in] descT
 *          Auxiliary factorization data, computed by PLASMA_zgeqrf.
 *
 * @param[out] Q
 *          On exit, the m-by-n matrix Q.
 *
 * @param[in] ldq
 *          The leading dimension of the array Q. ldq >= max(1,m).
 *
 *******************************************************************************
 *
 * @retval PLASMA_SUCCESS successful exit
 * @retval < 0 if -i, the i-th argument had an illegal value
 *
 *******************************************************************************
 *
 * @sa PLASMA_zungqr_Tile_Async
 * @sa PLASMA_cungqr
 * @sa PLASMA_dorgqr
 * @sa PLASMA_sorgqr
 * @sa PLASMA_zgeqrf
 *
 ******************************************************************************/
int PLASMA_zungqr(int m, int n, int k,
                  PLASMA_Complex64_t *A, int lda,
                  PLASMA_desc *descT,
                  PLASMA_Complex64_t *Q, int ldq)
{
    int nb;
    int retval;
    int status;

    PLASMA_desc descA;
    PLASMA_desc descQ;

    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }

    // Check input arguments
    if (m < 0) {
        plasma_error("illegal value of m");
        return -1;
    }
    if (n < 0 || n > m) {
        plasma_error("illegal value of n");
        return -2;
    }
    if (k < 0 || k > n) {
        plasma_error("illegal value of k");
        return -3;
    }
    if (lda < imax(1, m)) {
        plasma_error("illegal value of lda");
        return -5;
    }
    if (ldq < imax(1, m)) {
        plasma_error("illegal value of ldq");
        return -8;
    }
    if (imin(m, imin(n, k)) == 0)
        return PLASMA_SUCCESS;

    // Tune NB & IB depending on M & N; Set NB
    //status = plasma_tune(PLASMA_FUNC_ZGELS, M, N, 0);
    //if (status != PLASMA_SUCCESS) {
    //    plasma_error("PLASMA_zungqr", "plasma_tune() failed");
    //    return status;
    //}
    nb = plasma->nb;

    // Initialize tile matrix descriptors.
    descA = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, lda, n, 0, 0, m, k);

    descQ = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, ldq, n, 0, 0, m, n);

    // Allocate matrices in tile layout.
    retval = plasma_desc_mat_alloc(&descA);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_desc_mat_alloc() failed");
        return retval;
    }
    retval = plasma_desc_mat_alloc(&descQ);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_desc_mat_alloc() failed");
        plasma_desc_mat_free(&descA);
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

    // asynchronous block
    #pragma omp parallel
    #pragma omp master
    {
        // Translate to tile layout.
        PLASMA_zcm2ccrb_Async(A, lda, &descA, sequence, &request);
        PLASMA_zcm2ccrb_Async(Q, ldq, &descQ, sequence, &request);

        // Call the tile async function.
        PLASMA_zungqr_Tile_Async(&descA, descT, &descQ, sequence, &request);

        // Translate Q back to LAPACK layout.
        PLASMA_zccrb2cm_Async(&descQ, Q, ldq, sequence, &request);
    }
    // implicit synchronization

    // Free matrices in tile layout.
    plasma_desc_mat_free(&descA);
    plasma_desc_mat_free(&descQ);

    // Return status.
    status = sequence->status;
    plasma_sequence_destroy(sequence);
    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_ungqr
 *
 *  Non-blocking tile version of PLASMA_zungqr().
 *  May return before the computation is finished.
 *  Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] descA
 *          Descriptor of matrix A.
 *          A is stored in the tile layout.
 *
 * @param[in] descT
 *          Descriptor of matrix T.
 *          Auxiliary factorization data, computed by PLASMA_zgeqrf.
 *
 * @param[out] descQ
 *          Descriptor of matrix Q. On exit, matrix Q stored in the tile layout.
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
 * @sa PLASMA_zungqr
 * @sa PLASMA_cungqr_Tile_Async
 * @sa PLASMA_dorgqr_Tile_Async
 * @sa PLASMA_sorgqr_Tile_Async
 * @sa PLASMA_zgeqrf_Tile_Async
 *
 ******************************************************************************/
void PLASMA_zungqr_Tile_Async(PLASMA_desc *descA, PLASMA_desc *descT,
                              PLASMA_desc *descQ,
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

    // Check input arguments
    if (plasma_desc_check(descA) != PLASMA_SUCCESS) {
        plasma_error("invalid descriptor A");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (plasma_desc_check(descT) != PLASMA_SUCCESS) {
        plasma_error("invalid descriptor T");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (plasma_desc_check(descQ) != PLASMA_SUCCESS) {
        plasma_error("invalid descriptor Q");
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
    if (descA->nb != descA->mb || descQ->nb != descQ->mb) {
        plasma_error("only square tiles supported");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    // quick return
    //if (n <= 0)
    //    return;

    // set ones to diagonal of Q
    plasma_pzlaset(PlasmaFull,
                   (PLASMA_Complex64_t)0.0, (PLASMA_Complex64_t)1.0, *descQ,
                   sequence, request);

    // construct Q
    plasma_pzungqr(*descA, *descQ, *descT, sequence, request);

    return;
}
