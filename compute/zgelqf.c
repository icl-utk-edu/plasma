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
 * @ingroup plasma_gelqf
 *
 *  Computes tile LQ factorization of a complex m-by-n matrix A.
 *  The factorization has the form
 *    \f[ A = L \times Q \f],
 *  where L is a lower trapezoidal with positive diagonal and Q is a matrix with
 *  orthonormal rows.
 *
 *******************************************************************************
 *
 * @param[in] m
 *          The number of rows of the matrix A. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix A. n >= 0.
 *
 * @param[in,out] A
 *          On entry, the m-by-n matrix A.
 *          On exit, the elements on and below the diagonal of the array
 *          contain the m-by-min(m,n) lower trapezoidal matrix L (L is lower
 *          triangular if M <= N); the elements above the diagonal represent
 *          the unitary matrix Q as a product of elementary reflectors, stored
 *          by tiles.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,m).
 *
 * @param[out] descT
 *          On exit, auxiliary factorization data, required by PLASMA_zgelqs
 *          to solve the system of equations.
 *
 *******************************************************************************
 *
 * @retval PLASMA_SUCCESS successful exit
 * @retval < 0 if -i, the i-th argument had an illegal value
 *
 *******************************************************************************
 *
 * @sa plasma_omp_zgelqf
 * @sa PLASMA_cgelqf
 * @sa PLASMA_dgelqf
 * @sa PLASMA_sgelqf
 * @sa PLASMA_zgelqs
 *
 ******************************************************************************/
int PLASMA_zgelqf(int m, int n,
                  plasma_complex64_t *A, int lda,
                  plasma_desc_t *descT)
{
    int ib, nb;
    int retval;
    int status;

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
    if (n < 0) {
        plasma_error("illegal value of n");
        return -2;
    }
    if (lda < imax(1, m)) {
        plasma_error("illegal value of lda");
        return -4;
    }
    // Quick return
    if (imin(m, n) == 0)
        return PLASMA_SUCCESS;

    // Tune NB & IB depending on M, N & NRHS; Set NBNBSIZE
    //status = plasma_tune(PLASMA_FUNC_ZGELS, M, N, 0);
    //if (status != PLASMA_SUCCESS) {
    //    plasma_error("PLASMA_zgelqf", "plasma_tune() failed");
    //    return status;
    //}
    ib = plasma->ib;
    nb = plasma->nb;

    // Initialize tile matrix descriptor.
    plasma_desc_t descA;
    descA = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, m, n, 0, 0, m, n);

    // Allocate matrices in tile layout.
    retval = plasma_desc_mat_alloc(&descA);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_desc_mat_alloc() failed");
        return retval;
    }

    // Allocate workspace.
    plasma_workspace_t work;
    size_t lwork = nb + ib*nb;  // gelqt: tau + work
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
        // The Async functions are submitted here.  If an error occurs
        // (at submission time or at run time) the sequence->status
        // will be marked with an error.  After an error, the next
        // Async will not _insert_ more tasks into the runtime.  The
        // sequence->status can be checked after each call to _Async
        // or at the end of the parallel region.

        // Translate to tile layout.
        PLASMA_zcm2ccrb_Async(A, lda, &descA, sequence, &request);

        // Call the tile async function.
        plasma_omp_zgelqf(&descA, descT, &work, sequence, &request);

        // Translate back to LAPACK layout.
        PLASMA_zccrb2cm_Async(&descA, A, lda, sequence, &request);
    }
    // implicit synchronization

    plasma_workspace_free(&work);

    // Free matrix A in tile layout.
    plasma_desc_mat_free(&descA);

    // Return status.
    status = sequence->status;
    plasma_sequence_destroy(sequence);
    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_gelqf
 *
 *  Computes the tile LQ factorization of a matrix.
 *  Non-blocking tile version of PLASMA_zgelqf().
 *  May return before the computation is finished.
 *  Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in,out] A
 *          Descriptor of matrix A.
 *          A is stored in the tile layout.
 *
 * @param[out] T
 *          Descriptor of matrix T.
 *          On exit, auxiliary factorization data, required by PLASMA_zgelqs to
 *          solve the system of equations.
 *
 * @param[in] work
 *          Workspace for the auxiliary arrays needed by some coreblas kernels.
 *          For LQ factorization, contains preallocated space for TAU and WORK
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
 * @sa PLASMA_zgelqf
 * @sa plasma_omp_cgelqf
 * @sa plasma_omp_dgelqf
 * @sa plasma_omp_sgelqf
 * @sa plasma_omp_zgelqs
 *
 ******************************************************************************/
void plasma_omp_zgelqf(plasma_desc_t *A, plasma_desc_t *T,
                       plasma_workspace_t *work,
                       plasma_sequence_t *sequence,
                       plasma_request_t *request)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    // Check input arguments.
    if (plasma_desc_check(A) != PLASMA_SUCCESS) {
        plasma_error("invalid A");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (plasma_desc_check(T) != PLASMA_SUCCESS) {
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
    if (A->nb != A->mb) {
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
    if (imin(A->m, A->n) == 0)
        return;

    // Call the parallel function.
    plasma_pzgelqf(*A, *T, work, sequence, request);
}
