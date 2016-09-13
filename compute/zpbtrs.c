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
 * @ingroup plasma_pbtrs
 *
 *  Solves a system of linear equations A * X = B with a Hermitian positive definite
 *  matrix A using the Cholesky factorization of A (i.e., A = L*L^T or A = U^T*U)
 *  computed by PLASMA_zpbtrf.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of A is stored;
 *          - PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in] kd
 *          The number of suuperdiagonals within the band of A if uplo is upper,
 *          or the number of suuperdiagonals if uplo is lower. kd >= 0.
 *
 * @param[in] nrhs
 *          The number of right hand sides, i.e., the number of
 *          columns of the matrix B. nrhs >= 0.
 *
 * @param[in,out] AB
 *          The triangular factor U or L from the Cholesky
 *          factorization A = U^H*U or A = L*L^H, computed by
 *          PLASMA_zpotrf.
 *          Remark: If out-of-place layout translation is used, the
 *          matrix A can be considered as input, however if inplace
 *          layout translation is enabled, the content of A will be
 *          reordered for computation and restored before exiting the
 *          function.
 *
 * @param[in] ldab
 *          The leading dimension of the array AB.
 *
 * @param[in,out] B
 *          On entry, the n-by-nrhs right hand side matrix B.
 *          On exit, if return value = 0, the n-by-nrhs solution matrix X.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1,n).
 *
 *******************************************************************************
 *
 * @retval PLASMA_SUCCESS successful exit
 * @retval  < 0 if -i, the i-th argument had an illegal value
 *
 *******************************************************************************
 *
 * @sa PLASMA_zpbtrs_Tile_Async
 * @sa PLASMA_cpbtrs
 * @sa PLASMA_dpbtrs
 * @sa PLASMA_spbtrs
 * @sa PLASMA_zpbtrf
 *
 ******************************************************************************/
int PLASMA_zpbtrs(PLASMA_enum uplo, int n, int kd, int nrhs,
                  PLASMA_Complex64_t *AB, int ldab,
                  PLASMA_Complex64_t *B, int ldb)
{
    int nb;
    int retval;
    int status;

    PLASMA_desc descAB, descB;

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
        return -4;
    }
    if (nrhs < 0) {
        plasma_error("illegal value of nrhs");
        return -5;
    }
    if (ldab < imax(1, 1+kd))
    {
        plasma_error("illegal value of ldab");
        return -7;
    }
    if (ldb < imax(1, n)) {
        plasma_error("illegal value of ldb");
        return -10;
    }

    // quick return
    if (imax(n, nrhs) == 0)
        return PLASMA_SUCCESS;

    // Tune
    // status = plasma_tune(PLASMA_FUNC_ZPOSV, N, N, NHRS);
    // if (status != PLASMA_SUCCESS) {
    //     plasma_error("plasma_tune() failed");
    //     return status;
    // }

    // Set NT & NHRS
    nb = plasma->nb;
    // Initialize tile matrix descriptors.
    int tku  = (kd+kd+nb-1)/nb; // number of tiles in upper band (not including diagonal)
    int tkl  = (kd+nb-1)/nb;    // number of tiles in lower band (not including diagonal)
    int ldab_desc = (tku+tkl+1)*nb;  // since we use zgetrf on panel, we pivot back within panel.
                                     // this could fill the last tile of the panel,
                                     // and we need extra NB space on the bottom
    descAB = plasma_desc_band_init(PlasmaComplexDouble, uplo, nb, nb,
                                   nb*nb, ldab_desc, n, 0, 0, n, n, kd, kd);
    descB = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, n, nrhs, 0, 0, n, nrhs);

    // Allocate matrices in tile layout.
    retval = plasma_desc_mat_alloc(&descAB);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_desc_mat_alloc() failed");
        return retval;
    }
    retval = plasma_desc_mat_alloc(&descB);
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
    #pragma omp parallel
    #pragma omp master
    {
        // Translate to tile layout.
        PLASMA_zcm2ccrb_band_Async(uplo, AB, ldab, &descAB, sequence, &request);
        if (sequence->status == PLASMA_SUCCESS) {
            PLASMA_zcm2ccrb_Async(B, ldb, &descB, sequence, &request);
        }

        // Call the tile async function.
        if (sequence->status == PLASMA_SUCCESS) {
            PLASMA_zpbtrs_Tile_Async(uplo, &descAB, &descB, sequence, &request);
        }

        // Translate back to LAPACK layout.
        if (sequence->status == PLASMA_SUCCESS)
            PLASMA_zccrb2cm_Async(&descB, B, ldb, sequence, &request);
    }

    // Free matrix A in tile layout.
    plasma_desc_mat_free(&descAB);
    plasma_desc_mat_free(&descB);

    // Return status.
    status = sequence->status;
    plasma_sequence_destroy(sequence);
    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_pbtrs
 *
 *  Solves a system of linear equations using previously
 *  computed Cholesky factorization.
 *  Non-blocking tile version of PLASMA_zpbtrs().
 *  May return before the computation is finished.
 *  Operates on matrices stored by tiles.
 *  All matrices are passed through descriptors.
 *  All dimensions are taken from the descriptors.
 *  Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of A is stored;
 *          - PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] AB
 *          The triangular factor U or L from the Cholesky factorization
 *          A = U^H*U or A = L*L^H, computed by PLASMA_zpotrf.
 *
 * @param[in,out] B
 *          On entry, the n-by-nrhs right hand side matrix B.
 *          On exit, if return value = 0, the n-by-nrhs solution matrix X.
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
 * @sa PLASMA_zpbtrs
 * @sa PLASMA_zpbtrs_Tile_Async
 * @sa PLASMA_cpbtrs_Tile_Async
 * @sa PLASMA_dpbtrs_Tile_Async
 * @sa PLASMA_spbtrs_Tile_Async
 * @sa PLASMA_zpbtrf_Tile_Async
 *
 ******************************************************************************/
void PLASMA_zpbtrs_Tile_Async(PLASMA_enum uplo,
                              PLASMA_desc *AB,
                              PLASMA_desc *B,
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
    if ((uplo != PlasmaUpper) &&
        (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo");
        return;
    }
    if (plasma_desc_band_check(uplo, AB) != PLASMA_SUCCESS) {
        plasma_error("invalid A");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (plasma_desc_check(B) != PLASMA_SUCCESS) {
        plasma_error("invalid B");
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

    if (AB->mb != AB->nb) {
        plasma_error("only square tiles supported");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (AB->m != AB->n) {
        plasma_error("only square matrix A is supported");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (B->mb != B->nb) {
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
    /*
    if (min(n, nrhs) == 0)
        return;
    */

    // Call the parallel functions.
    plasma_pztbsm(
        PlasmaLeft,
        uplo,
        uplo == PlasmaUpper ? PlasmaConjTrans : PlasmaNoTrans,
        PlasmaNonUnit,
        1.0,
        *AB,
        *B,
        NULL,
        sequence,
        request);

    plasma_pztbsm(
        PlasmaLeft,
        uplo,
        uplo == PlasmaUpper ? PlasmaNoTrans : PlasmaConjTrans,
        PlasmaNonUnit,
        1.0,
        *AB,
        *B,
        NULL,
        sequence,
        request);

    return;
}
