/**
 *
 * @file zlange.c
 *
 *  PLASMA computational routines
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 3.0.0
 * @author  Mathieu Faverge
 * @author  Maksims Abalenkovs
 * @date    2016-07-22
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
 * @ingroup plasma_lange
 *
 *  PLASMA_zlange returns the value
 *
 *     zlange = ( max(abs(A(i,j))), NORM = PlasmaMaxNorm
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
 * @param[in] m
 *          The number of rows of the matrix A. m >= 0. When m = 0,
 *          the returned value is set to zero.
 *
 * @param[in] n
 *          The number of columns of the matrix A. n >= 0. When n = 0,
 *          the returned value is set to zero.
 *
 * @param[in] A
 *          The m-by-n matrix A.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,m).
 *
 *******************************************************************************
 *
 * @retval the norm described above
 *
 *******************************************************************************
 *
 * @sa PLASMA_zlange_Tile_Async
 * @sa PLASMA_clange
 * @sa PLASMA_dlange
 * @sa PLASMA_slange
 *
 ******************************************************************************/
double PLASMA_zlange(PLASMA_enum norm, int m, int n,
                     PLASMA_Complex64_t *A, int lda)
{
    int nb;
    int status;
    double value;
    plasma_context_t *plasma;
    PLASMA_sequence *sequence = NULL;
    PLASMA_request request = PLASMA_REQUEST_INITIALIZER;
    PLASMA_desc descA;

    // Get PLASMA context
    plasma = plasma_context_self();

    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }

    // Check input arguments
    if ((norm != PlasmaMaxNorm) && (norm != PlasmaOneNorm) &&
        (norm != PlasmaInfNorm) && (norm != PlasmaFrobeniusNorm)) {
        plasma_error("illegal value of norm");
        return -1;
    }
    if (m < 0) {
        plasma_error("illegal value of m");
        return -2;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -3;
    }
    if (lda < imax(1, m)) {
        plasma_error("illegal value of lda");
        return -5;
    }

    // Quick return
    if (imin(n, m) == 0)
      return (double)0.0;

    // Tune nb depending on m, n & nrhs, set nbnb
    /*
    status = plasma_tune(PLASMA_FUNC_ZGEMM, m, n, 0);

    if (status != PLASMA_SUCCESS) {
        plasma_error("plasma_tune() failed");
        return status;
    }
    */
    nb = plasma->nb;

    // Initialise matrix descriptor
    descA = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, lda, n, 0, 0, m, n);

    // Allocate matrix in tile layout
    retval = plasma_desc_mat_alloc(&descA);

    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_desc_mat_alloc() failed");
        return retval;
    }

    // Create sequence
    retval = plasma_sequence_create(&sequence);

    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_sequence_create() failed");
        return retval;
    }

#pragma omp parallel
#pragma omp master
    {
        /*
         * the Async functions are submitted here.  If an error occurs
         * (at submission time or at run time) the sequence->status
         * will be marked with an error.  After an error, the next
         * Async will not _insert_ more tasks into the runtime.  The
         * sequence->status can be checked after each call to _Async
         * or at the end of the parallel region.
         */

        // Translate matrix to tile layout
        PLASMA_zcm2ccrb_Async(A, lda, &descA, sequence, &request);

        // Call tile async interface
        if (sequence->status == PLASMA_SUCCESS) {

            PLASMA_zlange_Tile_Async(norm, &descA, &value, sequence, &request);

        }

        // Revert matrix to LAPACK layout
        if (sequence->status == PLASMA_SUCCESS)
            PLASMA_zccrb2cm_Async(&descA, A, lda, sequence, &request);

    } // pragma omp parallel block closed

    // Check for errors in async execution
    if (sequence->status != PLASMA_SUCCESS)
        return sequence->status;

    // Free matrix in tile layout
    plasma_desc_mat_free(&descA);

    // Destroy sequence
    plasma_sequence_destroy(sequence);

    return value;
}

/***************************************************************************//**
 *
 * @ingroup plasma_lange
 *
 *  Calculates the specified norm of a given matrix A. Non-blocking tile
 *  version of PLASMA_zlange(). May return before the computation is finished.
 *  Operates on matrices stored by tiles. All matrices are passed through
 *  descriptors. All dimensions are taken from the descriptors. Allows for
 *  pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] sequence
 *          Identifies the sequence of function calls that this call belongs to
 *          (for completion checks and exception handling purposes).
 *
 * @param[out] request
 *          Identifies this function call (for exception handling purposes).
 *
 *******************************************************************************
 *
 * @sa PLASMA_zlange
 * @sa PLASMA_clange_Tile_Async
 * @sa PLASMA_dlange_Tile_Async
 * @sa PLASMA_slange_Tile_Async
 *
 ******************************************************************************/
void PLASMA_zlange_Tile_Async(PLASMA_enum norm, PLASMA_desc *A, double *value,
                              PLASMA_sequence *sequence, PLASMA_request *request)
{
    PLASMA_desc descA;
    double *work = NULL;
    plasma_context_t *plasma;

    // Get PLASMA context
    plasma = plasma_context_self();

    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        plasma_request_fail(sequence, request, PLASMA_ERR_NOT_INITIALIZED);
        return;
    }

    if (sequence == NULL) {
        plasma_error("NULL sequence");
        plasma_request_fail(sequence, request, PLASMA_ERR_UNALLOCATED);
        return;
    }

    if (request == NULL) {
        plasma_error("NULL request");
        plasma_request_fail(sequence, request, PLASMA_ERR_UNALLOCATED);
        return;
    }

    // Check sequence status
    if (sequence->status != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);
        return;
    }

    // Check descriptor for correctness
    if (plasma_desc_check(A) != PLASMA_SUCCESS) {
        plasma_error("invalid descriptor");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    else {
        descA = *A;
    }

    // Check input arguments
    if (descA.nb != descA.mb) {
        plasma_error("only square tiles supported");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    if ((norm != PlasmaMaxNorm) && (norm != PlasmaOneNorm) &&
        (norm != PlasmaInfNorm) && (norm != PlasmaFrobeniusNorm)) {
        plasma_error("illegal value of norm");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    // Quick return
    if (imin(descA.m, descA.n) == 0) {
        *value = 0.0;
        return;
    }

    /*
    if (PLASMA_SCHEDULING == PLASMA_STATIC_SCHEDULING) {
        if (norm == PlasmaFrobeniusNorm) {
            work = plasma_shared_alloc(plasma, 2*PLASMA_SIZE, PlasmaRealDouble );
        } else {
            work = plasma_shared_alloc(plasma,   PLASMA_SIZE, PlasmaRealDouble );
        }
    }
    */

    // Call parallel function
    plasma_pzlange(norm, descA, work, value, sequence, request);

    /*
    if (work != NULL)
        plasma_shared_free( plasma, work );
        */

    return;
}
