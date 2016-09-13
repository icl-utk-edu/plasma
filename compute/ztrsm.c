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
 * @ingroup plasma_trsm
 *
 *  Solves one of the matrix equations
 *
 *    \f[ op( A ) \times X = \alpha B, \f] or
 *    \f[ X \times op( A ) = \alpha B, \f]
 *
 *  where op( A ) is one of:
 *    \f[ op( A ) = A,   \f]
 *    \f[ op( A ) = A^T, \f]
 *    \f[ op( A ) = A^H, \f]
 *
 *  alpha is a scalar, X and B are m-by-n matrices, and
 *  A is a unit or non-unit, upper or lower triangular matrix.
 *  The matrix X overwrites B.
 *
 *******************************************************************************
 *
 * @param[in] side
 *          - PlasmaLeft:  op(A)*X = B,
 *          - PlasmaRight: X*op(A) = B.
 *
 * @param[in] uplo
 *          - PlasmaUpper: A is upper triangular,
 *          - PlasmaLower: A is lower triangular.
 *
 * @param[in] transA
 *          - PlasmaNoTrans:   A is not transposed,
 *          - PlasmaTrans:     A is transposed,
 *          - PlasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] diag
 *          - PlasmaNonUnit: A has non-unit diagonal,
 *          - PlasmaUnit:    A has unit diagonal.
 *
 * @param[in] m
 *          The number of rows of the matrix B. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix B. n >= 0.
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          The k-by-k triangular matrix,
 *          where k = m if side = PlasmaLeft,
 *            and k = n if side = PlasmaRight.
 *          If uplo = PlasmaUpper, the leading k-by-k upper triangular part
 *          of the array A contains the upper triangular matrix, and the
 *          strictly lower triangular part of A is not referenced.
 *          If uplo = PlasmaLower, the leading k-by-k lower triangular part
 *          of the array A contains the lower triangular matrix, and the
 *          strictly upper triangular part of A is not referenced.
 *          If diag = PlasmaUnit, the diagonal elements of A are also not
 *          referenced and are assumed to be 1.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,k).
 *
 * @param[in,out] B
 *          On entry, the m-by-n right hand side matrix B.
 *          On exit, if return value = 0, the m-by-n solution matrix X.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1,m).
 *
 *
 *******************************************************************************
 *
 * @retval PlasmaSuccess successful exit
 *
 *******************************************************************************
 *
 * @sa plasma_omp_ztrsm
 * @sa PLASMA_ctrsm
 * @sa PLASMA_dtrsm
 * @sa PLASMA_strsm
 *
 ******************************************************************************/
int PLASMA_ztrsm(plasma_enum_t side, plasma_enum_t uplo,
                 plasma_enum_t transA, plasma_enum_t diag,
                 int m, int n,
                 plasma_complex64_t alpha, plasma_complex64_t *A, int lda,
                                           plasma_complex64_t *B, int ldb)
{
    int An;
    int nb;
    int retval;
    int status;

    plasma_desc_t descA;
    plasma_desc_t  descB;

    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    if (side == PlasmaLeft) {
        An = m;
    }
    else {
        An = n;
    }

    // Check input arguments
    if ((side != PlasmaLeft) &&
        (side != PlasmaRight)) {
        plasma_error("illegal value of side");
        return -1;
    }
    if ((uplo != PlasmaUpper) &&
            (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo");
        return -2;
    }
    if ((transA != PlasmaConjTrans) &&
        (transA != PlasmaNoTrans) &&
        (transA != PlasmaTrans )) {
        plasma_error("illegal value of transA");
        return -3;
    }
    if ((diag != PlasmaUnit) &&
            (diag != PlasmaNonUnit)) {
        plasma_error("illegal value of diag");
        return -4;
    }
    if (m < 0) {
        plasma_error("illegal value of m");
        return -5;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -6;
    }
    if (lda < imax(1, An)) {
        plasma_error("illegal value of lda");
        return -8;
    }
    if (ldb < imax(1, m)) {
        plasma_error("illegal value of ldb");
        return -10;
    }

    // quick return
    if ((m == 0) || (n == 0))
        return PlasmaSuccess;

    // Tune.
    // if (plasma_tune(PLASMA_FUNC_ZTRSM, m, n, 0) != PlasmaSuccess) {
    //     plasma_error("plasma_tune() failed");
    //     return status;
    // }
    nb = plasma->nb;

    // Initialize tile matrix descriptors.
    descA = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, An, An, 0, 0, An, An);

    descB = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, m, n, 0, 0, m, n);

    // Allocate matrices in tile layout.
    retval = plasma_desc_mat_alloc(&descA);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_mat_alloc() failed");
        return retval;
    }
    retval = plasma_desc_mat_alloc(&descB);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_mat_alloc() failed");
        plasma_desc_mat_free(&descA);
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
        PLASMA_zcm2ccrb_Async(B, ldb, &descB, sequence, &request);

        // Call the tile async function.
        plasma_omp_ztrsm(side, uplo, transA, diag,
                         alpha, &descA,
                                &descB,
                         sequence, &request);

        // Translate back to LAPACK layout.
        PLASMA_zccrb2cm_Async(&descB, B, ldb, sequence, &request);
    }
    // implicit synchronization

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
 * @ingroup plasma_trsm
 *
 *  Computes triangular solve.
 *  Non-blocking tile version of PLASMA_ztrsm().
 *  Operates on matrices stored by tiles.
 *  All matrices are passed through descriptors.
 *  All dimensions are taken from the descriptors.
 *  Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] side
 *          - PlasmaLeft:  op(A)*X = B,
 *          - PlasmaRight: X*op(A) = B.
 *
 * @param[in] uplo
 *          - PlasmaUpper: A is upper triangular,
 *          - PlasmaLower: A is lower triangular.
 *
 * @param[in] transA
 *          - PlasmaNoTrans:   A is not transposed,
 *          - PlasmaTrans:     A is transposed,
 *          - PlasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] diag
 *          - PlasmaNonUnit: A has non-unit diagonal,
 *          - PlasmaUnit:    A has unit diagonal.
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          Descriptor of matrix A.
 *
 * @param[in] B
 *          Descriptor of matrix B.
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
 *          request->status should never be set to PlasmaSuccess (the
 *          initial values) since another async call may be setting a
 *          failure value at the same time.
 *
 *******************************************************************************
 *
 * @sa PLASMA_ztrsm
 * @sa plasma_omp_ctrsm
 * @sa plasma_omp_dtrsm
 * @sa plasma_omp_strsm
 *
 ******************************************************************************/
void plasma_omp_ztrsm(plasma_enum_t side, plasma_enum_t uplo,
                      plasma_enum_t transA, plasma_enum_t diag,
                      plasma_complex64_t alpha, plasma_desc_t *A,
                                                plasma_desc_t *B,
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
    if ((side != PlasmaLeft) &&
        (side != PlasmaRight)) {
        plasma_error("illegal value of side");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if ((uplo != PlasmaUpper) &&
        (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if ((transA != PlasmaConjTrans) &&
        (transA != PlasmaNoTrans) &&
        (transA != PlasmaTrans)) {
        plasma_error("illegal value of transA");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if ((diag != PlasmaUnit) &&
        (diag != PlasmaNonUnit)) {
        plasma_error("illegal value of diag");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (plasma_desc_check(A) != PlasmaSuccess) {
        plasma_error("invalid A");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (plasma_desc_check(B) != PlasmaSuccess) {
        plasma_error("invalid B");
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

    // quick return
    if ((B->m == 0) || (B->n == 0))
      return;

    // Call the parallel function.
    plasma_pztrsm(side, uplo,
                  transA, diag,
                  alpha, *A,
                         *B,
                  sequence, request);
}
