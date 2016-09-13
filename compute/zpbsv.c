/**
 *
 * @file zpbsv.c
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
 * @ingroup plasma_pbsv
 *
 *  Computes the solution to a system of linear equations A * X = B,
 *  where A is an n-by-n Hermitian positive definite band matrix, and X and B
 *  are n-by-nrhs matrices. The Cholesky decomposition is used to factor A as
 *
 *    \f[ A =  L\times L^H, \f] if uplo = PlasmaLower,
 *    or
 *    \f[ A =  U^H\times U, \f] if uplo = PlasmaUpper,
 *
 *  where U is an upper triangular matrix and  L is a lower triangular matrix.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of A is stored;
 *          - PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] n
 *          The number of linear equations, i.e., the order of the matrix A.
 *          n >= 0.
 *
 * @param[in] kd
 *          The number of subdiagonals within the band of A if uplo=upper.
 *          The number of suuperdiagonals within the band of A. ku >= 0.
 *
 * @param[in] nrhs
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B.  nrhs >= 0.
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
 *          The leading dimension of the array AB. ldab >= max(1,n).
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
 * @retval  > 0 if i, the leading minor of order i of A is not
 *          positive definite, so the factorization could not
 *          be completed, and the solution has not been computed.
 *
 *******************************************************************************
 *
 * @sa PLASMA_zpbsv_Tile_Async
 * @sa PLASMA_cpbsv
 * @sa PLASMA_dpbsv
 * @sa PLASMA_spbsv
 *
 ******************************************************************************/
int PLASMA_zpbsv(PLASMA_enum uplo, int n, int kd, int nrhs,
                 PLASMA_Complex64_t *AB, int ldab,
                 PLASMA_Complex64_t *B, int ldb)
{
    int nb;
    int status;
    int retval;

    PLASMA_desc descAB;
    PLASMA_desc descB;

    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
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
    if (nrhs < 0) {
        plasma_error("illegal value of nrhs");
        return -4;
    }
    if (ldab < kd+1) {
        plasma_error("illegal value of ldab");
        return -6;
    }
    if (ldb < imax(1, n)) {
        plasma_error("illegal value of ldb");
        return -8;
    }
    // Quick return - currently NOT equivalent to LAPACK's
    //LAPACK does not have such check for DPOSV
    //
    //if (min(n, nrhs) == 0)
    //    return PLASMA_SUCCESS;

    // Tune.
    //status = plasma_tune(PLASMA_FUNC_ZPOSV, N, N, nrhs);
    //if (status != PLASMA_SUCCESS) {
    //   plasma_error("PLASMA_zposv", "plasma_tune() failed");
    //    return status;
    // }
    nb    = plasma->nb;

    // Initialize tile matrix descriptors.
    int lda = nb*(1+(kd+nb-1)/nb);
    descAB = plasma_desc_band_init(PlasmaComplexDouble, uplo, nb, nb,
                                   nb*nb, lda, n, 0, 0, n, n, kd, kd);
    descB = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, ldb, nrhs, 0, 0, n, nrhs);

    // Allocate matrices in tile layout.
    retval = plasma_desc_mat_alloc(&descAB);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_desc_mat_alloc() failed");
        return retval;
    }
    retval = plasma_desc_mat_alloc(&descB);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_desc_mat_alloc() failed");
        plasma_desc_mat_free(&descAB);
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
        // Translate to tile layout.
        PLASMA_zcm2ccrb_band_Async(uplo, AB, ldab, &descAB, sequence, &request);
        PLASMA_zcm2ccrb_Async(B, ldb, &descB, sequence, &request);

        // Call the tile async function.
        PLASMA_zpbsv_Tile_Async(uplo,
                                &descAB,
                                &descB,
                                sequence, &request);

        // Translate back to LAPACK layout.
        PLASMA_zccrb2cm_Async(&descB, B, ldb, sequence, &request);
        PLASMA_zccrb2cm_band_Async(uplo, &descAB, AB, ldab, sequence, &request);
    }
    // implicit synchronization

    // Free matrices in tile layout.
    plasma_desc_mat_free(&descAB);
    plasma_desc_mat_free(&descB);

    // Return status.
    status = sequence->status;
    plasma_sequence_destroy(sequence);
    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_pbsv
 *
 *  Solves a Hermitian positive definite band system of linear equations
 *  using Cholesky factorization.
 *  Non-blocking tile version of PLASMA_zpbsv().
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
 * @param[in,out] AB
 *          Descriptor of matrix A.
 *
 * @param[in,out] B
 *          Descriptor of right-hand-sides B.
 *
 * @param[in] sequence
 *          Identifies the sequence of function calls that this call belongs to
 *          (for completion checks and exception handling purposes).  Check
 *          the sequence->status for errors.

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
 * @sa PLASMA_zpbsv
 * @sa PLASMA_cpbsv_Tile_Async
 * @sa PLASMA_dpbsv_Tile_Async
 * @sa PLASMA_spbsv_Tile_Async
 *
 ******************************************************************************/
void PLASMA_zpbsv_Tile_Async(PLASMA_enum uplo,
                             PLASMA_desc *AB,
                             PLASMA_desc *B,
                             PLASMA_sequence *sequence, 
                             PLASMA_request *request)
{
    PLASMA_Complex64_t zone = 1.0;

    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
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
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        plasma_error("invalid A");
        return;
    }
    if (plasma_desc_check(B) != PLASMA_SUCCESS) {
        plasma_error("invalid B");
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
/*
    // quick return
    if (min(n, nrhs == 0)
        return PLASMA_SUCCESS;
*/
    // Call the parallel functions.
    // Do factorization
    plasma_pzpbtrf(uplo, *AB, sequence, request);

    // Do forward-substitution
    plasma_pztbsm(PlasmaLeft, 
                  uplo,
                  uplo == PlasmaUpper ? PlasmaConjTrans : PlasmaNoTrans, 
                  PlasmaNonUnit,
                  zone,
                  *AB,
                  *B,
                  NULL,
                  sequence, request);

    // Do backward-substitution
    plasma_pztbsm(PlasmaLeft, 
                  uplo,
                  uplo == PlasmaUpper ? PlasmaNoTrans : PlasmaConjTrans, 
                  PlasmaNonUnit,
                  zone,
                  *AB,
                  *B,
                  NULL,
                  sequence, request);
}