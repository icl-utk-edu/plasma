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
 * @ingroup plasma_trmm
 *
 *  Performs a triangular matrix-matrix multiply of the form
 *
 *          \f[B = \alpha [op(A) \times B] \f], if side = PlasmaLeft  or
 *          \f[B = \alpha [B \times op(A)] \f], if side = PlasmaRight
 *
 *  where op( X ) is one of:
 *
 *          - op(A) = A   or
 *          - op(A) = A^T or
 *          - op(A) = A^H
 *
 *  alpha is a scalar, B is an m-by-n matrix and A is a unit or non-unit, upper
 *  or lower triangular matrix.
 *
 *******************************************************************************
 *
 * @param[in] side
 *          Specifies whether op( A ) appears on the left or on the right of B:
 *          - PlasmaLeft:  alpha*op( A )*B
 *          - PlasmaRight: alpha*B*op( A )
 *
 * @param[in] uplo
 *          Specifies whether the matrix A is upper triangular or lower
 *          triangular:
 *          - PlasmaUpper: Upper triangle of A is stored;
 *          - PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] transA
 *          Specifies whether the matrix A is transposed, not transposed or
 *          conjugate transposed:
 *          - PlasmaNoTrans:   A is transposed;
 *          - PlasmaTrans:     A is not transposed;
 *          - PlasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] diag
 *          Specifies whether or not A is unit triangular:
 *          - PlasmaNonUnit: A is non-unit triangular;
 *          - PlasmaUnit:    A is unit triangular.
 *
 * @param[in] m
 *          The number of rows of matrix B.
 *          m >= 0.
 *
 * @param[in] n
 *          The number of columns of matrix B.
 *          n >= 0.
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          The triangular matrix A of dimension lda-by-k, where k is m when
 *          side='L' or 'l' and k is n when when side='R' or 'r'. If uplo =
 *          PlasmaUpper, the leading k-by-k upper triangular part of the array
 *          A contains the upper triangular matrix, and the strictly lower
 *          triangular part of A is not referenced. If uplo = PlasmaLower, the
 *          leading k-by-k lower triangular part of the array A contains the
 *          lower triangular matrix, and the strictly upper triangular part of
 *          A is not referenced. If diag = PlasmaUnit, the diagonal elements of
 *          A are also not referenced and are assumed to be 1.
 *
 * @param[in] lda
 *          The leading dimension of the array A. When side='L' or 'l',
 *          lda >= max(1,m), when side='R' or 'r' then lda >= max(1,n).
 *
 * @param[in,out] B
 *          On entry, the matrix B of dimension ldb-by-n.
 *          On exit, the result of a triangular matrix-matrix multiply
 *          ( alpha*op(A)*B ) or ( alpha*B*op(A) ).
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1,m).
 *
 *******************************************************************************
 *
 * @retval PLASMA_SUCCESS successful exit
 *
 *******************************************************************************
 *
 * @sa plasma_omp_ztrmm
 * @sa PLASMA_ctrmm
 * @sa PLASMA_dtrmm
 * @sa PLASMA_strmm
 *
 ******************************************************************************/
int PLASMA_ztrmm(plasma_enum_t side, plasma_enum_t uplo,
                 plasma_enum_t transA, plasma_enum_t diag,
                 int m, int n, plasma_complex64_t alpha,
                 plasma_complex64_t *A, int lda,
                 plasma_complex64_t *B, int ldb)
{
    int retval;
    int nb, na;
    int status;
    plasma_context_t *plasma;
    plasma_sequence_t  *sequence = NULL;
    plasma_request_t    request = PLASMA_REQUEST_INITIALIZER;
    plasma_desc_t descA, descB;

    // Get PLASMA context
    plasma = plasma_context_self();

    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }

    if (side == PlasmaLeft) {
      na = m;
    }
    else {
      na = n;
    }

    // Check input arguments
    if (side != PlasmaLeft && side != PlasmaRight) {
        plasma_error("illegal value of side");
        return -1;
    }
    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        plasma_error("illegal value of uplo");
        return -2;
    }
    if (transA != PlasmaConjTrans &&
        transA != PlasmaNoTrans   &&
        transA != PlasmaTrans )
    {
        plasma_error("illegal value of transA");
        return -3;
    }
    if (diag != PlasmaUnit && diag != PlasmaNonUnit) {
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
    if (lda < imax(1, na)) {
        plasma_error("illegal value of lda");
        return -8;
    }
    if (ldb < imax(1, m)) {
        plasma_error("illegal value of ldb");
        return -10;
    }

    // Quick return
    if (imin(m, n) == 0)
        return PLASMA_SUCCESS;

    // Tune nb depending on m, n
    // if (plasma_tune(PLASMA_FUNC_ZTRMM, m, n, 0) != PLASMA_SUCCESS) {
    //     plasma_error("plasma_tune() failed");
    //     return status;
    // }
    nb = plasma->nb;

    // Initialise matrix descriptors
    descA = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, na, na, 0, 0, na, na);

    descB = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, m,  n,  0, 0, m,  n);

    // Allocate matrices in tile layout
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

    // Create sequence
    retval = plasma_sequence_create(&sequence);

    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_sequence_create() failed");
        return retval;
    }

    // Asynchronous block
    #pragma omp parallel
    #pragma omp master
    {
        // Translate matrices to tile layout
        PLASMA_zcm2ccrb_Async(A, lda, &descA, sequence, &request);

        if (sequence->status == PLASMA_SUCCESS)
            PLASMA_zcm2ccrb_Async(B, ldb, &descB, sequence, &request);

        // Call tile async interface
        if (sequence->status == PLASMA_SUCCESS) {
            plasma_omp_ztrmm(side, uplo, transA, diag,
                             alpha, &descA,
                                    &descB,
                             sequence, &request);
        }

        // Revert matrices to LAPACK layout
        if (sequence->status == PLASMA_SUCCESS)
            PLASMA_zccrb2cm_Async(&descA, A, lda, sequence, &request);

        if (sequence->status == PLASMA_SUCCESS)
            PLASMA_zccrb2cm_Async(&descB, B, ldb, sequence, &request);
    } // pragma omp parallel block closed
    // implicit synchronization

    // Free matrices in tile layout
    plasma_desc_mat_free(&descA);
    plasma_desc_mat_free(&descB);

    // Destroy sequence
    plasma_sequence_destroy(sequence);

    // Return status
    status = sequence->status;

    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_trmm
 *
 *  Performs triangular matrix multiplication. Non-blocking tile version of
 *  PLASMA_ztrmm(). May return before the computation is finished. Operates on
 *  matrices stored by tiles. All matrices are passed through descriptors. All
 *  dimensions are taken from the descriptors. Allows for pipelining of
 *  operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] A
 *          Descriptor of the triangular matrix A.
 *
 * @param[in,out] B
 *          Descriptor of matrix B.
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
 * @sa PLASMA_ztrmm
 * @sa plasma_omp_ctrmm
 * @sa plasma_omp_dtrmm
 * @sa plasma_omp_strmm
 *
 ******************************************************************************/
void plasma_omp_ztrmm(plasma_enum_t side, plasma_enum_t uplo,
                            plasma_enum_t transA, plasma_enum_t diag,
                            plasma_complex64_t alpha, plasma_desc_t *A,
                                                      plasma_desc_t *B,
                            plasma_sequence_t *sequence, plasma_request_t  *request)
{
    plasma_desc_t descA;
    plasma_desc_t descB;
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

    // Check descriptors for correctness
    if (plasma_desc_check(A) != PLASMA_SUCCESS) {
        plasma_error("invalid first descriptor");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    else {
        descA = *A;
    }

    if (plasma_desc_check(B) != PLASMA_SUCCESS) {
        plasma_error("invalid second descriptor");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    else {
        descB = *B;
    }

    // Check input arguments
    if (descA.nb != descA.mb || descB.nb != descB.mb) {
        plasma_error("only square tiles supported");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    if (side != PlasmaLeft && side != PlasmaRight) {
        plasma_error("illegal value of side");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        plasma_error("illegal value of uplo");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    if (transA != PlasmaConjTrans &&
        transA != PlasmaNoTrans &&
        transA != PlasmaTrans) {

        plasma_error("illegal value of transA");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    if (diag != PlasmaUnit && diag != PlasmaNonUnit) {
        plasma_error("illegal value of diag");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    // Quick return
    if (A->m == 0 || A->n == 0 || alpha == 0.0 || B->m == 0 || B->n == 0)
        return;

    // Call parallel function
    plasma_pztrmm(side, uplo, transA, diag, alpha,
                  descA, descB, sequence, request);

    return;
}
