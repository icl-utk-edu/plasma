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
 * @ingroup plasma_syrk
 *
 *  Performs one of the symmetric rank k operations
 *
 *    \f[ C = \alpha A \times A^T + \beta C, \f]
 *    or
 *    \f[ C = \alpha A^T \times A + \beta C, \f]
 *
 *  where alpha and beta are scalars, C is an n-by-n symmetric
 *  matrix, and A is an n-by-k matrix in the first case and a k-by-n
 *  matrix in the second case.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of C is stored;
 *          - PlasmaLower: Lower triangle of C is stored.
 *
 * @param[in] trans
 *          - PlasmaNoTrans: \f[ C = \alpha A \times A^T + \beta C; \f]
 *          - PlasmaTrans:   \f[ C = \alpha A^T \times A + \beta C. \f]
 *
 * @param[in] n
 *          The order of the matrix C. n >= 0.
 *
 * @param[in] k
 *          If trans = PlasmaNoTrans, number of columns of the A matrix;
 *          if trans = PlasmaTrans, number of rows of the A matrix.
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] pA
 *          A is an lda-by-ka matrix.
 *          If trans = PlasmaNoTrans, ka = k;
 *          if trans = PlasmaTrans,   ka = n.
 *
 * @param[in] lda
 *          The leading dimension of the array A.
 *          If trans = PlasmaNoTrans, lda >= max(1, n);
 *          if trans = PlasmaTrans,   lda >= max(1, k).
 *
 * @param[in] beta
 *          The scalar beta.
 *
 * @param[in,out] pC
 *          C is an ldc-by-n matrix.
 *          On exit, the uplo part of the matrix is overwritten
 *          by the uplo part of the updated matrix.
 *
 * @param[in] ldc
 *          The leading dimension of the array C. ldc >= max(1, n).
 *
 *******************************************************************************
 *
 * @retval PlasmaSuccess successful exit
 *
 *******************************************************************************
 *
 * @sa plasma_omp_zsyrk
 * @sa plasma_csyrk
 * @sa plasma_dsyrk
 * @sa plasma_ssyrk
 *
 ******************************************************************************/
int plasma_zsyrk(plasma_enum_t uplo, plasma_enum_t trans,
                 int n, int k,
                 plasma_complex64_t alpha, plasma_complex64_t *pA, int lda,
                 plasma_complex64_t beta,  plasma_complex64_t *pC, int ldc)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    // Check input arguments.
    if ((uplo != PlasmaUpper) &&
        (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo");
        return -1;
    }
    if ((trans != PlasmaNoTrans) &&
        (trans != PlasmaTrans)) {
        plasma_error("illegal value of trans");
        return -2;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -3;
    }
    if (k < 0) {
        plasma_error("illegal value of k");
        return -4;
    }

    int am, an;
    if (trans == PlasmaNoTrans) {
        am = n;
        an = k;
    }
    else {
        am = k;
        an = n;
    }

    if (lda < imax(1, am)) {
        plasma_error("illegal value of lda");
        return -7;
    }
    if (ldc < imax(1, n)) {
        plasma_error("illegal value of ldc");
        return -10;
    }

    // quick return
    if (n == 0 || ((alpha == 0.0 || k == 0) && beta == 1.0))
        return PlasmaSuccess;

    // Tune parameters.
    if (plasma->tuning)
        plasma_tune_syrk(plasma, PlasmaComplexDouble, n, k);

    // Set tiling parameters.
    int nb = plasma->nb;

    // Create tile matrices.
    plasma_desc_t A;
    plasma_desc_t C;
    int retval;
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        am, an, 0, 0, am, an, &A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        return retval;
    }
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        n, n, 0, 0, n, n, &C);
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
        plasma_omp_zge2desc(pC, ldc, C, &sequence, &request);

        // Call the tile async function.
        plasma_omp_zsyrk(uplo, trans,
                         alpha, A,
                         beta,  C,
                         &sequence, &request);

        // Translate back to LAPACK layout.
        plasma_omp_zdesc2ge(C, pC, ldc, &sequence, &request);
    }
    // implicit synchronization

    // Free matrices in tile layout.
    plasma_desc_destroy(&A);
    plasma_desc_destroy(&C);

    // Return status.
    int status = sequence.status;
    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_syrk
 *
 *  Performs rank k update.
 *  Non-blocking tile version of plasma_zsyrk().
 *  May return before the computation is finished.
 *  Operates on matrices stored by tiles.
 *  All matrices are passed through descriptors.
 *  All dimensions are taken from the descriptors.
 *  Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of C is stored;
 *          - PlasmaLower: Lower triangle of C is stored.
 *
 * @param[in] trans
 *          - PlasmaNoTrans: \f[ C = \alpha A \times A^T + \beta C; \f]
 *          - PlasmaTrans:   \f[ C = \alpha A^T \times A + \beta C. \f]
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
 * @sa plasma_zsyrk
 * @sa plasma_omp_zsyrk
 * @sa plasma_omp_csyrk
 * @sa plasma_omp_dsyrk
 * @sa plasma_omp_ssyrk
 *
 ******************************************************************************/
void plasma_omp_zsyrk(plasma_enum_t uplo, plasma_enum_t trans,
                      plasma_complex64_t alpha, plasma_desc_t A,
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
    if ((uplo != PlasmaUpper) &&
        (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if ((trans != PlasmaNoTrans) &&
        (trans != PlasmaTrans)) {
        plasma_error("illegal value of trans");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (plasma_desc_check(A) != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        plasma_error("invalid A");
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
    int k = trans == PlasmaNoTrans ? A.n : A.m;
    if (C.m == 0 || ((alpha == 0.0 || k == 0) && beta == 1.0))
        return;

    // Call the parallel function.
    plasma_pzsyrk(uplo, trans,
                  alpha, A,
                  beta,  C,
                  sequence, request);
}
