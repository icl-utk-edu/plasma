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
 * @ingroup plasma_gemm
 *
 *  Performs one of the matrix-matrix operations
 *
 *          \f[ C = \alpha [op( A )\times op( B )] + \beta C, \f]
 *
 *  where op( X ) is one of:
 *    \f[ op( X ) = X,   \f]
 *    \f[ op( X ) = X^T, \f]
 *    \f[ op( X ) = X^H, \f]
 *
 *  alpha and beta are scalars, and A, B and C are matrices, with op( A )
 *  an m-by-k matrix, op( B ) a k-by-n matrix and C an m-by-n matrix.
 *
 *******************************************************************************
 *
 * @param[in] transA
 *          - PlasmaNoTrans:   A is not transposed,
 *          - PlasmaTrans:     A is transposed,
 *          - PlasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] transB
 *          - PlasmaNoTrans:   B is not transposed,
 *          - PlasmaTrans:     B is transposed,
 *          - PlasmaConjTrans: B is conjugate transposed.
 *
 * @param[in] m
 *          The number of rows of the matrix op( A ) and of the matrix C.
 *          m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix op( B ) and of the matrix C.
 *          n >= 0.
 *
 * @param[in] k
 *          The number of columns of the matrix op( A ) and the number of rows
 *          of the matrix op( B ). k >= 0.
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          An lda-by-ka matrix, where ka is k when transA = PlasmaNoTrans,
 *          and is m otherwise.
 *
 * @param[in] lda
 *          The leading dimension of the array A.
 *          When transA = PlasmaNoTrans, lda >= max(1,m),
 *          otherwise, lda >= max(1,k).
 *
 * @param[in] B
 *          An ldb-by-kb matrix, where kb is n when transB = PlasmaNoTrans,
 *          and is k otherwise.
 *
 * @param[in] ldb
 *          The leading dimension of the array B.
 *          When transB = PlasmaNoTrans, ldb >= max(1,k),
 *          otherwise, ldb >= max(1,n).
 *
 * @param[in] beta
 *          The scalar beta.
 *
 * @param[in,out] C
 *          An ldc-by-n matrix. On exit, the array is overwritten by the m-by-n
 *          matrix ( alpha*op( A )*op( B ) + beta*C ).
 *
 * @param[in] ldc
 *          The leading dimension of the array C. ldc >= max(1,m).
 *
 *******************************************************************************
 *
 * @retval PlasmaSuccess successful exit
 *
 *******************************************************************************
 *
 * @sa plasma_omp_zgemm
 * @sa PLASMA_cgemm
 * @sa PLASMA_dgemm
 * @sa PLASMA_sgemm
 *
 ******************************************************************************/
int PLASMA_zgemm(plasma_enum_t transA, plasma_enum_t transB,
                 int m, int n, int k,
                 plasma_complex64_t alpha, plasma_complex64_t *A, int lda,
                                           plasma_complex64_t *B, int ldb,
                 plasma_complex64_t beta,  plasma_complex64_t *C, int ldc)
{
    int Am, An;
    int Bm, Bn;
    int nb;
    int retval;
    int status;

    plasma_desc_t descA;
    plasma_desc_t descB;
    plasma_desc_t descC;

    plasma_complex64_t zzero = 0.0;
    plasma_complex64_t zone  = 1.0;

    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    // Check input arguments.
    if ((transA != PlasmaNoTrans) &&
        (transA != PlasmaTrans) &&
        (transA != PlasmaConjTrans)) {
        plasma_error("illegal value of transA");
        return -1;
    }
    if ((transB != PlasmaNoTrans) &&
        (transB != PlasmaTrans) &&
        (transB != PlasmaConjTrans)) {
        plasma_error("illegal value of transB");
        return -2;
    }
    if (m < 0) {
        plasma_error("illegal value of m");
        return -3;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -4;
    }
    if (k < 0) {
        plasma_error("illegal value of k");
        return -5;
    }
    if (A == NULL) {
        plasma_error("NULL A");
        return -7;
    }

    if (transA == PlasmaNoTrans) {
        Am = m;
        An = k;
    }
    else {
        Am = k;
        An = m;
    }
    if (transB == PlasmaNoTrans) {
        Bm = k;
        Bn = n;
    }
    else {
        Bm = n;
        Bn = k;
    }

    if (lda < imax(1, Am)) {
        plasma_error("illegal value of lda");
        return -8;
    }
    if (B == NULL) {
        plasma_error("NULL B");
        return -9;
    }
    if (ldb < imax(1, Bm)) {
        plasma_error("illegal value of ldb");
        return -10;
    }
    if (C == NULL) {
        plasma_error("NULL C");
        return -12;
    }
    if (ldc < imax(1, m)) {
        plasma_error("illegal value of ldc");
        return -13;
    }

    // quick return
    if (m == 0 || n == 0 || ((alpha == zzero || k == 0) && beta == zone))
        return PlasmaSuccess;

    // Tune.
    // if (plasma_tune(PLASMA_FUNC_ZGEMM, m, n, 0) != PlasmaSuccess) {
    //     plasma_error("plasma_tune() failed");
    //     return status;
    // }
    nb = plasma->nb;

    // Initialize tile matrix descriptors.
    descA = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, Am, An, 0, 0, Am, An);

    descB = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, Bm, Bn, 0, 0, Bm, Bn);

    descC = plasma_desc_init(PlasmaComplexDouble, nb, nb,
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
    retval = plasma_desc_mat_alloc(&descC);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_mat_alloc() failed");
        plasma_desc_mat_free(&descA);
        plasma_desc_mat_free(&descB);
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
        PLASMA_zcm2ccrb_Async(C, ldc, &descC, sequence, &request);

        // Call the tile async function.
        if (sequence->status == PlasmaSuccess) {
            plasma_omp_zgemm(transA, transB,
                             alpha, &descA,
                             &descB,
                             beta, &descC,
                             sequence, &request);
        }

        // Translate back to LAPACK layout.
        PLASMA_zccrb2cm_Async(&descC, C, ldc, sequence, &request);
    }
    // implicit synchronization

    // Free matrices in tile layout.
    plasma_desc_mat_free(&descA);
    plasma_desc_mat_free(&descB);
    plasma_desc_mat_free(&descC);

    // Return status.
    status = sequence->status;
    plasma_sequence_destroy(sequence);
    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_gemm
 *
 *  Performs matrix multiplication.
 *  Non-blocking tile version of PLASMA_zgemm().
 *  May return before the computation is finished.
 *  Operates on matrices stored by tiles.
 *  All matrices are passed through descriptors.
 *  All dimensions are taken from the descriptors.
 *  Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] transA
 *          - PlasmaNoTrans:   A is not transposed,
 *          - PlasmaTrans:     A is transposed,
 *          - PlasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] transB
 *          - PlasmaNoTrans:   B is not transposed,
 *          - PlasmaTrans:     B is transposed,
 *          - PlasmaConjTrans: B is conjugate transposed.
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
 * @param[in] beta
 *          The scalar beta.
 *
 * @param[in,out] C
 *          Descriptor of matrix C.
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
 * @sa PLASMA_zgemm
 * @sa plasma_omp_cgemm
 * @sa plasma_omp_dgemm
 * @sa plasma_omp_sgemm
 *
 ******************************************************************************/
void plasma_omp_zgemm(plasma_enum_t transA, plasma_enum_t transB,
                      plasma_complex64_t alpha, plasma_desc_t *A,
                                                plasma_desc_t *B,
                      plasma_complex64_t beta,  plasma_desc_t *C,
                      plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }

    // Check input arguments.
    if ((transA != PlasmaNoTrans) &&
        (transA != PlasmaTrans) &&
        (transA != PlasmaConjTrans)) {
        plasma_error("illegal value of transA");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if ((transB != PlasmaNoTrans) &&
        (transB != PlasmaTrans) &&
        (transB != PlasmaConjTrans)) {
        plasma_error("illegal value of transB");
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
    if (plasma_desc_check(C) != PlasmaSuccess) {
        plasma_error("invalid C");
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
    int k = transA == PlasmaNoTrans ? A->n : A->m;
    plasma_complex64_t zzero = (plasma_complex64_t)0.0;

    if (C->m == 0 || C->n == 0 || ((alpha == zzero || k == 0) && beta == 1.0))
        return;

    // Call the parallel function.
    plasma_pzgemm(transA, transB,
                  alpha, *A,
                         *B,
                   beta, *C,
                  sequence, request);
}
