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
 * @ingroup plasma_unglq
 *
 *  Generates an m-by-n matrix Q with orthonormal rows, which is
 *  defined as the first m rows of a product of the elementary reflectors
 *  returned by PLASMA_zgelqf.
 *
 *******************************************************************************
 *
 * @param[in] m
 *          The number of rows of the matrix Q. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix Q. n >= m.
 *
 * @param[in] k
 *          The number of rows of elementary tile reflectors whose product
 *          defines the matrix Q.
 *          m >= k >= 0.
 *
 * @param[in] A
 *          Details of the LQ factorization of the original matrix A as returned
 *          by PLASMA_zgelqf.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,m).
 *
 * @param[in] descT
 *          Auxiliary factorization data, computed by PLASMA_zgelqf.
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
 * @sa plasma_omp_zunglq
 * @sa PLASMA_cunglq
 * @sa PLASMA_dorglq
 * @sa PLASMA_sorglq
 * @sa PLASMA_zgelqf
 *
 ******************************************************************************/
int PLASMA_zunglq(int m, int n, int k,
                  PLASMA_Complex64_t *A, int lda,
                  plasma_desc_t *descT,
                  PLASMA_Complex64_t *Q, int ldq)
{
    int ib, nb;
    int retval;
    int status;

    plasma_desc_t descA, descQ;

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
    if (n < m) {
        plasma_error("illegal value of n");
        return -2;
    }
    if (k < 0 || k > m) {
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
    // Quick return
    if (m <= 0)
        return PLASMA_SUCCESS;

    // Tune NB & IB depending on M & N; Set NB
    //status = plasma_tune(PLASMA_FUNC_ZGELS, M, N, 0);
    //if (status != PLASMA_SUCCESS) {
    //    plasma_error("PLASMA_zunglq", "plasma_tune() failed");
    //    return status;
    //}
    ib = plasma->ib;
    nb = plasma->nb;

    // Initialize tile matrix descriptors.
    descA = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, lda, n, 0, 0, k, n);

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

    // Allocate workspace.
    plasma_workspace_t work;
    size_t lwork = ib*nb;  // unmlq: work
    retval = plasma_workspace_alloc(&work, lwork, PlasmaComplexDouble);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_workspace_alloc() failed");
        return retval;
    }

    // Create sequence.
    plasma_sequence_t *sequence = NULL;
    retval = plasma_sequence_create(&sequence);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_sequence_create() failed");
        return retval;
    }

    // Initialize request.
    plasma_request_t request = PLASMA_REQUEST_INITIALIZER;

    #pragma omp parallel
    #pragma omp master
    {
        // the Async functions are submitted here.  If an error occurs
        // (at submission time or at run time) the sequence->status
        // will be marked with an error.  After an error, the next
        // Async will not _insert_ more tasks into the runtime.  The
        // sequence->status can be checked after each call to _Async
        // or at the end of the parallel region.

        // Translate to tile layout.
        PLASMA_zcm2ccrb_Async(A, lda, &descA, sequence, &request);
        PLASMA_zcm2ccrb_Async(Q, ldq, &descQ, sequence, &request);

        // Call the tile async function.
        plasma_omp_zunglq(&descA, descT, &descQ, &work, sequence, &request);

        // Translate Q back to LAPACK layout.
        PLASMA_zccrb2cm_Async(&descQ, Q, ldq, sequence, &request);
    }
    // implicit synchronization

    plasma_workspace_free(&work);

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
 * @ingroup plasma_unglq
 *
 *  Non-blocking tile version of PLASMA_zunglq().
 *  May return before the computation is finished.
 *  Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] A
 *          Descriptor of matrix A.
 *          A is stored in the tile layout.
 *
 * @param[in] T
 *          Descriptor of matrix T.
 *          Auxiliary factorization data, computed by PLASMA_zgelqf.
 *
 * @param[out] Q
 *          Descriptor of matrix Q. On exit, matrix Q stored in the tile layout.
 *
 * @param[in] work
 *          Workspace for the auxiliary arrays needed by some coreblas kernels.
 *          For multiplication by Q contains preallocated space for WORK
 *          arrays. Allocated by the plasma_workspace_alloc function.
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
 * @sa PLASMA_zunglq
 * @sa plasma_omp_cunglq
 * @sa plasma_omp_dorglq
 * @sa plasma_omp_sorglq
 * @sa plasma_omp_zgelqf
 *
 ******************************************************************************/
void plasma_omp_zunglq(plasma_desc_t *A, plasma_desc_t *T, plasma_desc_t *Q,
                       plasma_workspace_t *work,
                       plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    // Check input arguments
    if (plasma_desc_check(A) != PLASMA_SUCCESS) {
        plasma_error("invalid descriptor A");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (plasma_desc_check(T) != PLASMA_SUCCESS) {
        plasma_error("invalid descriptor T");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (plasma_desc_check(Q) != PLASMA_SUCCESS) {
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
    if (A->nb != A->mb || Q->nb != Q->mb) {
        plasma_error("only square tiles supported");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    // Check sequence status.
    if (sequence->status != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);
        return;
    }

    // Quick return
    if (Q->m <= 0)
        return;

    // set ones to diagonal of Q
    plasma_pzlaset(PlasmaFull, 0., 1., *Q, sequence, request);

    // construct Q
    plasma_pzunglq(*A, *Q, *T, work, sequence, request);
}
