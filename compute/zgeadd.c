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
 * @ingroup plasma_geadd
 *
 *  Performs an addition of two general rectangular matrices similarly to the
 *  pzgeadd() function from the PBLAS library:
 *
 *    \f[ B = \alpha * op( A ) + \beta * B, \f]
 *
 *  where op( X ) is one of:
 *    \f[ op( X ) = X,   \f]
 *    \f[ op( X ) = X^T, \f]
 *    \f[ op( X ) = X^H, \f]
 *
 *  alpha and beta are scalars and A, B are matrices with op( A ) an m-by-n or
 *  n-by-m matrix depending on the value of transa and B an m-by-n matrix.
 *
 *******************************************************************************
 *
 * @param[in] transa
 *          Specifies whether the matrix A is non-transposed, transposed, or
 *          conjugate transposed
 *          - PlasmaNoTrans:   op( A ) = A
 *          - PlasmaTrans:     op( A ) = A^T
 *          - PlasmaConjTrans: op( A ) = A^H
 *
 * @param[in] m
 *          Number of rows of the matrices op( A ) and B.
 *          m >= 0.
 *
 * @param[in] n
 *          Number of columns of the matrices op( A ) and B.
 *          n >= 0.
 *
 * @param[in] alpha
 *          Scalar factor of A.
 *
 * @param[in] pA
 *          Matrix of size lda-by-k, where k is n when transa == PlasmaNoTrans
 *          and m otherwise.
 *
 * @param[in] lda
 *          Leading dimension of the array A. lda >= max(1,l), where l is m
 *          when transa = PlasmaNoTrans and n otherwise.
 *
 * @param[in] beta
 *          Scalar factor of B.
 *
 * @param[in,out] pB
 *          Matrix of size ldb-by-n.
 *          On exit, B = alpha * op( A ) + beta * B
 *
 * @param[in] ldb
 *          Leading dimension of the array B.
 *          ldb >= max(1,m).
 *
 *******************************************************************************
 *
 * @retval PlasmaSuccess successful exit
 *
 *******************************************************************************
 *
 * @sa plasma_omp_zgeadd
 * @sa plasma_cgeadd
 * @sa plasma_dgeadd
 * @sa plasma_sgeadd
 *
 ******************************************************************************/
int plasma_zgeadd(plasma_enum_t transa,
                  int m, int n,
                  plasma_complex64_t alpha, plasma_complex64_t *pA, int lda,
                  plasma_complex64_t beta,  plasma_complex64_t *pB, int ldb)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    // Check input arguments.
    if ((transa != PlasmaNoTrans) &&
        (transa != PlasmaTrans)   &&
        (transa != PlasmaConjTrans)) {
        plasma_error("illegal value of transa");
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
    if (pA == NULL) {
        plasma_error("NULL A");
        return -5;
    }

    int am, an;
    if (transa == PlasmaNoTrans) {
        am = m;
        an = n;
    }
    else {
        am = n;
        an = m;
    }
    int bm = m;
    int bn = n;

    if (lda < imax(1, am)) {
        plasma_error("illegal value of lda");
        return -6;
    }
    if (pB == NULL) {
        plasma_error("NULL B");
        return -8;
    }
    if (ldb < imax(1, bm)) {
        plasma_error("illegal value of ldb");
        return -9;
    }

    // quick return
    if (m == 0 || n == 0 || (alpha == 0.0 && beta == 1.0))
        return PlasmaSuccess;

    // Tune parameters.
    if (plasma->tuning)
        plasma_tune_geadd(plasma, PlasmaComplexDouble, m, n);

    // Set tiling parameters.
    int nb = plasma->nb;

    // Create tile matrices.
    plasma_desc_t A;
    plasma_desc_t B;
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

        // Call tile async function.
        plasma_omp_zgeadd(transa,
                          alpha,     A,
                          beta,      B,
                          &sequence, &request);

        // Translate back to LAPACK layout.
        plasma_omp_zdesc2ge(A, pA, lda, &sequence, &request);
        plasma_omp_zdesc2ge(B, pB, ldb, &sequence, &request);
    }
    // implicit synchronization

    // Free matrices in tile layout.
    plasma_desc_destroy(&A);
    plasma_desc_destroy(&B);

    // Return status.
    int status = sequence.status;
    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_geadd
 *
 *  Performs an addition of two general rectangular matrices similarly to the
 *  pzgeadd() function from the PBLAS library. Non-blocking tile version of
 *  plasma_zgeadd(). May return before the computation is finished. Operates on
 *  matrices stored by tiles. All matrices are passed through descriptors. All
 *  dimensions are taken from the descriptors. Allows for pipelining of
 *  operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] transa
 *          Specifies whether the matrix A is non-transposed, transposed, or
 *          conjugate transposed
 *          - PlasmaNoTrans:   op( A ) = A
 *          - PlasmaTrans:     op( A ) = A^T
 *          - PlasmaConjTrans: op( A ) = A^H
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          Descriptor of matrix A.
 *
 * @param[in] beta
 *          The scalar beta.
 *
 * @param[in,out] B
 *          Descriptor of matrix B.
 *
 * @param[in] sequence
 *          Identifies the sequence of function calls that this call belongs to
 *         (for completion checks and exception handling purposes). Check the
 *          sequence->status for errors.
 *
 * @param[out] request
 *          Identifies this function call (for exception handling purposes).
 *
 * @retval void
 *          Errors are returned by setting sequence->status and
 *          request->status to error values. The sequence->status and
 *          request->status should never be set to PlasmaSuccess (the
 *          initial values) since another async call may be setting a
 *          failure value at the same time.
 *
 *******************************************************************************
 *
 * @sa plasma_zgeadd
 * @sa plasma_omp_cgeadd
 * @sa plasma_omp_dgeadd
 * @sa plasma_omp_sgeadd
 *
 ******************************************************************************/
void plasma_omp_zgeadd(plasma_enum_t transa,
                       plasma_complex64_t alpha, plasma_desc_t A,
                       plasma_complex64_t beta,  plasma_desc_t B,
                       plasma_sequence_t *sequence, plasma_request_t  *request)
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
        (transa != PlasmaTrans)   &&
        (transa != PlasmaConjTrans)) {
        plasma_error("illegal value of transa");
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
    int am = transa == PlasmaNoTrans ? A.m : A.n;
    if ((alpha == 0.0 || am == 0) && beta == 1.0)
        return;

    // Call the parallel function.
    plasma_pzgeadd(transa,
                   alpha, A,
                   beta,  B,
                   sequence, request);
}
