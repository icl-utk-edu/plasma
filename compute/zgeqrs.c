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
 *  Computes a minimum-norm solution min || A*X - B || using the
 *  QR factorization A = Q*R computed by PLASMA_zgeqrf.
 *
 *******************************************************************************
 *
 * @param[in] m
 *          The number of rows of the matrix A. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix A. m >= n >= 0.
 *
 * @param[in] nrhs
 *          The number of columns of B. nrhs >= 0.
 *
 * @param[in] A
 *          Details of the QR factorization of the original matrix A as returned
 *          by PLASMA_zgeqrf.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= m.
 *
 * @param[in] descT
 *          Auxiliary factorization data, computed by PLASMA_zgeqrf.
 *
 * @param[in,out] B
 *          On entry, the m-by-nrhs right hand side matrix B.
 *          On exit, the n-by-nrhs solution matrix X.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1,n).
 *
 *******************************************************************************
 *
 * @retval PLASMA_SUCCESS successful exit
 * @retval < 0 if -i, the i-th argument had an illegal value
 *
 *******************************************************************************
 *
 * @sa PLASMA_zgeqrs_Tile_Async
 * @sa PLASMA_cgeqrs
 * @sa PLASMA_dgeqrs
 * @sa PLASMA_sgeqrs
 * @sa PLASMA_zgeqrf
 * @sa PLASMA_zgels
 *
 *******************************************************************************/
int PLASMA_zgeqrs(int m, int n, int nrhs,
                  PLASMA_Complex64_t *A, int lda,
                  PLASMA_desc *descT,
                  PLASMA_Complex64_t *B, int ldb)
{
    int nb;
    int retval;
    int status;

    PLASMA_desc descA, descB;

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
    if (nrhs < 0) {
        plasma_error("illegal value of nrhs");
        return -3;
    }
    if (lda < imax(1, m)) {
        plasma_error("illegal value of lda");
        return -5;
    }
    if (ldb < imax(1, imax(1, m))) {
        plasma_error("illegal value of ldb");
        return -8;
    }
    // Quick return
    if (imin(m, imin(n, nrhs)) == 0) {
        return PLASMA_SUCCESS;
    }

    // Tune NB & IB depending on M, N & NRHS; Set NBNBSIZE
    //status = plasma_tune(PLASMA_FUNC_ZGELS, M, N, NRHS);
    //if (status != PLASMA_SUCCESS) {
    //    plasma_error("plasma_tune() failed");
    //    return status;
    //}

    nb = plasma->nb;

    // Initialize tile matrix descriptors.
    descA = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, m, n, 0, 0, m, n);

    descB = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, m, nrhs, 0, 0, m, nrhs);

    // Allocate matrices in tile layout.
    retval = plasma_desc_mat_alloc(&descA);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_desc_mat_alloc() failed");
        return retval;
    }

    retval = plasma_desc_mat_alloc(&descB);
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
        if (sequence->status == PLASMA_SUCCESS)
            PLASMA_zcm2ccrb_Async(B, ldb, &descB, sequence, &request);

        // Call the tile async function.
        if (sequence->status == PLASMA_SUCCESS) {
            PLASMA_zgeqrs_Tile_Async(&descA, descT, &descB, sequence, &request);
        }

        // Translate back to LAPACK layout.
        // Jakub S.: It does not seem needed to translate the descriptor back
        //           for out-of-place storage
        //if (sequence->status == PLASMA_SUCCESS)
        //    PLASMA_zccrb2cm_Async(&descA, A, lda, sequence, &request);
        if (sequence->status == PLASMA_SUCCESS)
            PLASMA_zccrb2cm_Async(&descB, B, ldb, sequence, &request);
    } // pragma omp parallel block closed

    // Free matrices in tile layout.
    plasma_desc_mat_free(&descA);
    plasma_desc_mat_free(&descB);

    // Return status.
    status = sequence->status;
    plasma_sequence_destroy(sequence);
    return status;
}

/***************************************************************************//**
 *
 * @ingroup PLASMA_Complex64_t_Tile_Async
 *
 *  Computes a minimum-norm solution using the tile QR factorization.
 *  Non-blocking equivalent of PLASMA_zgeqrs().
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
 * @param[in,out] descB
 *          Descriptor of matrix B.
 *          On entry, right-hand side matrix B in the tile layout.
 *          On exit, solution matrix X in the tile layout.
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
 * @sa PLASMA_zgeqrs
 * @sa PLASMA_zgeqrs_Tile
 * @sa PLASMA_cgeqrs_Tile_Async
 * @sa PLASMA_dgeqrs_Tile_Async
 * @sa PLASMA_sgeqrs_Tile_Async
 * @sa PLASMA_zgeqrf_Tile_Async
 * @sa PLASMA_zgels_Tile_Async
 *
 ******************************************************************************/
void PLASMA_zgeqrs_Tile_Async(PLASMA_desc *descA, PLASMA_desc *descT, 
                              PLASMA_desc *descB,
                              PLASMA_sequence *sequence, PLASMA_request *request)
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
        plasma_error("invalid descriptor A");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (plasma_desc_check(descT) != PLASMA_SUCCESS) {
        plasma_error("invalid descriptor T");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (plasma_desc_check(descB) != PLASMA_SUCCESS) {
        plasma_error("invalid descriptor B");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (descA->nb != descA->mb || descB->nb != descB->mb) {
        plasma_error("only square tiles supported");
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

    // Check sequence status.
    if (sequence->status != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);
        return;
    }

    // Quick return
    // Jakub S.: Why was it commented out in version 2.8.0 ?
    // I leave it like that till explained.
    //if (min(M, min(N, NRHS)) == 0) {
    //    return PLASMA_SUCCESS;
    //}

    // Call the parallel function for Q^T * B (in real case)
    //                             or Q^H * B (in complex case)
    // Plasma_ConjTrans will be converted to PlasmaTrans by the
    // automatic datatype conversion, which is what we want here.
    // Note that PlasmaConjTrans is protected from this conversion.
    plasma_pzunmqr(PlasmaLeft, Plasma_ConjTrans,
                   *descA, *descB, *descT,
                   sequence, request);

    // Call the parallel function for solving R * X = Q^T * B (in real case)
    //                                     or R * X = Q^H * B (in complex case)
    plasma_pztrsm(PlasmaLeft, PlasmaUpper,
                  PlasmaNoTrans, PlasmaNonUnit,
                  1.0, plasma_desc_submatrix(*descA, 0, 0, descA->n, descA->n),
                  plasma_desc_submatrix(*descB, 0, 0, descA->n, descB->n),
                  sequence, request);

    return;
}
