/**
 *
 * @file zgbtrf.c
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
 * @ingroup plasma_gbtrf
 *
 *  Performs the LU factorization of an m-by-n band matrix A with partial pivoting
 *  with row interchanges. The factorization has the form
 *
 *    \f[ A = P \times L \times U, \f]
 *
 *  where P is a permutation matrix, L is lower triangular with unit diagonal 
 *  elements, and U is upper triangular.
 *
 *******************************************************************************
 *
 * @param[in] m
 *          The number of rows of the matrix A. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix A. n >= 0.
 *
 * @param[in] kl
 *          The number of subdiagonals within the band of A. kl >= 0.
 *
 * @param[in] ku
 *          The number of suuperdiagonals within the band of A. ku >= 0.
 *
 * @param[in,out] AB
 *          An ldab-by-n matrix.
 *          On entry, the matrix A in band storage, in rows kl+1 to
 *          2*kl+ku+1; rows 1 to kl of the array need not be set.
 *          The j-th column of A is stored in the j-th column of the
 *          array AB as follows:
 *          AB(kl+ku+1+i-j,j) = A(i,j) for max(1,j-ku)<=i<=min(m,j+kl)
 * \n
 *          On exit, details of the factorization: U is stored as an
 *          upper triangular band matrix with kl+ku superdiagonals in
 *          rows 1 to kl+ku+1, and the multipliers used during the
 *          factorization are stored in rows kl+ku+2 to 2*kl+ku+1.
 *
 * @param[in] ldab
 *          The leading dimension of the array AB. ldab >= 2*kl+ku+1.
 *
 * @param[out] ipiv
 *          The pivot indices; for 1 <= i <= min(m,n), row i of the
 *          matrix was interchanged with row ipiv(i).
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
 * @sa PLASMA_zgbtrf_Tile_Async
 * @sa PLASMA_cgbtrf
 * @sa PLASMA_dgbtrf
 * @sa PLASMA_sgbtrf
 *
 ******************************************************************************/
int PLASMA_zgbtrf(int m, int n, int kl, int ku,
                  PLASMA_Complex64_t *AB, int ldab,
                  int *ipiv, int *fill)
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
    if (m < 0) {
        plasma_error("illegal value of m");
        return -1;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -2;
    }
    if (kl < 0) {
        plasma_error("illegal value of kl");
        return -3;
    }
    if (ku < 0) {
        plasma_error("illegal value of ku");
        return -4;
    }
    if (ldab < 2*kl+ku+1) {
        plasma_error("illegal value of ldab");
        return -6;
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
    int tku = (ku+kl+nb-1)/nb; // number of tiles in upper band (not including diagonal)
    int tkl = (kl+nb-1)/nb;    // number of tiles in lower band (not including diagonal)
    int lda = (tku+tkl+1)*nb;  // since we use zgetrf on panel, we pivot back within panel.
                               // this could fill the last tile of the panel,
                               // and we need extra NB space on the bottom
    descAB = plasma_desc_band_init(PlasmaComplexDouble, nb, nb,
                                   nb*nb, lda, n, 0, 0, m, n, kl, ku);

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
        PLASMA_zcm2ccrb_band_Async(AB, ldab, &descAB, sequence, &request);
    } // pragma omp parallel block closed
    int *fake = (int*)malloc(descAB.nt * sizeof(int));
    #pragma omp parallel
    #pragma omp master
    {
        // Call the tile async function.
        if (sequence->status == PLASMA_SUCCESS) {
            PLASMA_zgbtrf_Tile_Async(&descAB, ipiv, fill, fake, sequence, &request);
        }
    } // pragma omp parallel block closed
    free(fake);
    #pragma omp parallel
    #pragma omp master
    {
        // Translate back to LAPACK layout.
        if (sequence->status == PLASMA_SUCCESS)
            PLASMA_zccrb2cm_band_Async(&descAB, AB, ldab, sequence, &request);
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
 * @ingroup plasma_gbtrf
 *
 *  Performs the LU factorization of a band matrix.
 *  Non-blocking tile version of PLASMA_zgbtrf().
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
 * @param[out] ipiv
 *          The pivot indices; for 1 <= i <= min(m,n), row i of the
 *          matrix was interchanged with row ipiv(i).
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
 * @sa PLASMA_zgbtrf
 * @sa PLASMA_zgbtrf_Tile_Async
 * @sa PLASMA_cgbtrf_Tile_Async
 * @sa PLASMA_dgbtrf_Tile_Async
 * @sa PLASMA_sgbtrf_Tile_Async
 *
 ******************************************************************************/
void PLASMA_zgbtrf_Tile_Async(PLASMA_desc *AB, int *ipiv, int *fill, int *fake,
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
    if (plasma_desc_band_check(AB) != PLASMA_SUCCESS) {
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
    plasma_pzgbtrf(*AB, ipiv, fill, fake, sequence, request);

    return;
}
