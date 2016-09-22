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
 * @retval PlasmaSuccess successful exit
 * @retval < 0 if -i, the i-th argument had an illegal value
 *
 *******************************************************************************
 *
 * @sa plasma_omp_zungqr
 * @sa PLASMA_cungqr
 * @sa PLASMA_dorgqr
 * @sa PLASMA_sorgqr
 * @sa PLASMA_zgeqrf
 *
 ******************************************************************************/
int PLASMA_zungqr(int m, int n, int k,
                  plasma_complex64_t *A, int lda,
                  plasma_desc_t *descT,
                  plasma_complex64_t *Q, int ldq)
{
    int ib, nb;
    int retval;
    int status;

    plasma_desc_t descA;
    plasma_desc_t descQ;

    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
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
    // Quick return
    if (n <= 0)
        return PlasmaSuccess;

    // Tune NB & IB depending on M & N; Set NB
    //status = plasma_tune(PLASMA_FUNC_ZGELS, M, N, 0);
    //if (status != PlasmaSuccess) {
    //    plasma_error("PLASMA_zungqr", "plasma_tune() failed");
    //    return status;
    //}
    ib = plasma->ib;
    nb = plasma->nb;

    // Create tile matrices.
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        lda, n, 0, 0, m, k, &descA);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        return retval;
    }
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        ldq, n, 0, 0, m, n, &descQ);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        plasma_desc_destroy(&descA);
        return retval;
    }

    // Allocate workspace.
    plasma_workspace_t work;
    size_t lwork = ib*nb;  // unmqr: work
    retval = plasma_workspace_alloc(&work, lwork, PlasmaComplexDouble);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_workspace_alloc() failed");
        return retval;
    }

    // Create sequence.
    plasma_sequence_t *sequence = NULL;
    retval = plasma_sequence_create(&sequence);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_sequence_create() failed");
        return retval;
    }
    // Initialize request.
    plasma_request_t request = PLASMA_REQUEST_INITIALIZER;

    // asynchronous block
    #pragma omp parallel
    #pragma omp master
    {
        // Translate to tile layout.
        PLASMA_zcm2ccrb_Async(A, lda, &descA, sequence, &request);
        PLASMA_zcm2ccrb_Async(Q, ldq, &descQ, sequence, &request);

        // Call the tile async function.
        plasma_omp_zungqr(&descA, descT, &descQ, &work, sequence, &request);

        // Translate Q back to LAPACK layout.
        PLASMA_zccrb2cm_Async(&descQ, Q, ldq, sequence, &request);
    }
    // implicit synchronization

    plasma_workspace_free(&work);

    // Free matrices in tile layout.
    plasma_desc_destroy(&descA);
    plasma_desc_destroy(&descQ);

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
 * @param[in] A
 *          Descriptor of matrix A.
 *          A is stored in the tile layout.
 *
 * @param[in] T
 *          Descriptor of matrix T.
 *          Auxiliary factorization data, computed by PLASMA_zgeqrf.
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
 *          request->status should never be set to PlasmaSuccess (the
 *          initial values) since another async call may be setting a
 *          failure value at the same time.
 *
 *******************************************************************************
 *
 * @sa PLASMA_zungqr
 * @sa plasma_omp_cungqr
 * @sa plasma_omp_dorgqr
 * @sa plasma_omp_sorgqr
 * @sa plasma_omp_zgeqrf
 *
 ******************************************************************************/
void plasma_omp_zungqr(plasma_desc_t *A, plasma_desc_t *T, plasma_desc_t *Q,
                       plasma_workspace_t *work,
                       plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }

    // Check input arguments
    if (plasma_desc_check(A) != PlasmaSuccess) {
        plasma_error("invalid descriptor A");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (A->mb != plasma->nb || A->nb != plasma->nb) {
        plasma_error("wrong tile dimensions of A");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (plasma_desc_check(T) != PlasmaSuccess) {
        plasma_error("invalid descriptor T");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (T->mb != plasma->ib || T->nb != plasma->nb) {
        plasma_error("wrong tile dimensions of T");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (plasma_desc_check(Q) != PlasmaSuccess) {
        plasma_error("invalid descriptor Q");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (Q->mb != plasma->nb || Q->nb != plasma->nb) {
        plasma_error("wrong tile dimensions of Q");
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

    // Quick return
    if (Q->n <= 0)
        return;

    // set ones to diagonal of Q
    plasma_pzlaset(PlasmaGeneral,
                   (plasma_complex64_t)0.0, (plasma_complex64_t)1.0, *Q,
                   sequence, request);

    // construct Q
    plasma_pzungqr(*A, *Q, *T, work, sequence, request);
}
