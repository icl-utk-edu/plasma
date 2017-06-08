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

#include "plasma.h"
#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_tuning.h"
#include "plasma_types.h"
#include "plasma_workspace.h"

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
 * @param[in] transa
 *          - PlasmaNoTrans:   A is not transposed,
 *          - PlasmaTrans:     A is transposed,
 *          - PlasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] transb
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
 * @param[in] pA
 *          An lda-by-ka matrix, where ka is k when transa = PlasmaNoTrans,
 *          and is m otherwise.
 *
 * @param[in] lda
 *          The leading dimension of the array A.
 *          When transa = PlasmaNoTrans, lda >= max(1,m),
 *          otherwise, lda >= max(1,k).
 *
 * @param[in] pB
 *          An ldb-by-kb matrix, where kb is n when transb = PlasmaNoTrans,
 *          and is k otherwise.
 *
 * @param[in] ldb
 *          The leading dimension of the array B.
 *          When transb = PlasmaNoTrans, ldb >= max(1,k),
 *          otherwise, ldb >= max(1,n).
 *
 * @param[in] beta
 *          The scalar beta.
 *
 * @param[in,out] pC
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
 * @sa plasma_cgemm
 * @sa plasma_dgemm
 * @sa plasma_sgemm
 *
 ******************************************************************************/
int plasma_zgemm(plasma_enum_t transa, plasma_enum_t transb,
                 int m, int n, int k,
                 plasma_complex64_t alpha, plasma_complex64_t *pA, int lda,
                                           plasma_complex64_t *pB, int ldb,
                 plasma_complex64_t beta,  plasma_complex64_t *pC, int ldc)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    // Check input arguments.
    if ((transa != PlasmaNoTrans) &&
        (transa != PlasmaTrans) &&
        (transa != PlasmaConjTrans)) {
        plasma_error("illegal value of transa");
        return -1;
    }
    if ((transb != PlasmaNoTrans) &&
        (transb != PlasmaTrans) &&
        (transb != PlasmaConjTrans)) {
        plasma_error("illegal value of transb");
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

    int am, an;
    int bm, bn;
    if (transa == PlasmaNoTrans) {
        am = m;
        an = k;
    }
    else {
        am = k;
        an = m;
    }
    if (transb == PlasmaNoTrans) {
        bm = k;
        bn = n;
    }
    else {
        bm = n;
        bn = k;
    }

    if (lda < imax(1, am)) {
        plasma_error("illegal value of lda");
        return -8;
    }
    if (ldb < imax(1, bm)) {
        plasma_error("illegal value of ldb");
        return -10;
    }
    if (ldc < imax(1, m)) {
        plasma_error("illegal value of ldc");
        return -13;
    }

    // quick return
    if (m == 0 || n == 0 || ((alpha == 0.0 || k == 0) && beta == 1.0))
        return PlasmaSuccess;

    // Tune parameters.
    if (plasma->tuning)
        plasma_tune_gemm(plasma, PlasmaComplexDouble, m, n, k);

    // Set tiling parameters.
    int nb = plasma->nb;

    // Create tile matrices.
    plasma_desc_t A;
    plasma_desc_t B;
    plasma_desc_t C;
    int retval;
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        am, an, 0, 0, am, an, &A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        return retval;
    }
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        bm, bn, 0, 0, bm, bn, &B);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        plasma_desc_destroy(&A);
        return retval;
    }
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        m, n, 0, 0, m, n, &C);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        plasma_desc_destroy(&A);
        plasma_desc_destroy(&B);
        return retval;
    }

    // Initialize sequence.
    plasma_sequence_t sequence;
    retval = plasma_sequence_init(&sequence);

    // Initialize request.
    plasma_request_t request;
    retval = plasma_request_init(&request);

    // asynchronous block
    #pragma omp parallel
    #pragma omp master
    {
        // Translate to tile layout.
        plasma_omp_zge2desc(pA, lda, A, &sequence, &request);
        plasma_omp_zge2desc(pB, ldb, B, &sequence, &request);
        plasma_omp_zge2desc(pC, ldc, C, &sequence, &request);

        // Call the tile async function.
        plasma_omp_zgemm(transa, transb,
                         alpha, A,
                                B,
                         beta,  C,
                         &sequence, &request);

        // Translate back to LAPACK layout.
        plasma_omp_zdesc2ge(C, pC, ldc, &sequence, &request);
    }
    // implicit synchronization

    // Free matrices in tile layout.
    plasma_desc_destroy(&A);
    plasma_desc_destroy(&B);
    plasma_desc_destroy(&C);

    // Return status.
    int status = sequence.status;
    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_gemm
 *
 *  Performs matrix multiplication.
 *  Non-blocking tile version of plasma_zgemm().
 *  May return before the computation is finished.
 *  Operates on matrices stored by tiles.
 *  All matrices are passed through descriptors.
 *  All dimensions are taken from the descriptors.
 *  Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] transa
 *          - PlasmaNoTrans:   A is not transposed,
 *          - PlasmaTrans:     A is transposed,
 *          - PlasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] transb
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
 * @sa plasma_zgemm
 * @sa plasma_omp_cgemm
 * @sa plasma_omp_dgemm
 * @sa plasma_omp_sgemm
 *
 ******************************************************************************/
void plasma_omp_zgemm(plasma_enum_t transa, plasma_enum_t transb,
                      plasma_complex64_t alpha, plasma_desc_t A,
                                                plasma_desc_t B,
                      plasma_complex64_t beta,  plasma_desc_t C,
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
    if ((transa != PlasmaNoTrans) &&
        (transa != PlasmaTrans) &&
        (transa != PlasmaConjTrans)) {
        plasma_error("illegal value of transa");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if ((transb != PlasmaNoTrans) &&
        (transb != PlasmaTrans) &&
        (transb != PlasmaConjTrans)) {
        plasma_error("illegal value of transb");
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
    int k = transa == PlasmaNoTrans ? A.n : A.m;
    if (C.m == 0 || C.n == 0 || ((alpha == 0.0 || k == 0) && beta == 1.0))
        return;

    // Call the parallel function.
    plasma_pzgemm(transa, transb,
                  alpha, A,
                         B,
                  beta,  C,
                  sequence, request);
}
