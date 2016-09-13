/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> c
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
 * @ingroup plasma_herk
 *
 *  Performs one of the Hermitian rank k operations
 *
 *    \f[ C = \alpha A \times A^H + \beta C, \f]
 *    or
 *    \f[ C = \alpha A^H \times A + \beta C, \f]
 *
 *  where alpha and beta are real scalars, C is an n-by-n Hermitian
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
 *          - PlasmaNoTrans:   \f[ C = \alpha A \times A^H + \beta C; \f]
 *          - PlasmaConjTrans: \f[ C = \alpha A^H \times A + \beta C. \f]
 *
 * @param[in] n
 *          The order of the matrix C. n >= 0.
 *
 * @param[in] k
 *          If trans = PlasmaNoTrans, number of columns of the A matrix;
 *          if trans = PlasmaConjTrans, number of rows of the A matrix.
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          A is a lda-by-ka matrix.
 *          If trans = PlasmaNoTrans,   ka = k;
 *          if trans = PlasmaConjTrans, ka = n.
 *
 * @param[in] lda
 *          The leading dimension of the array A.
 *          If trans = PlasmaNoTrans,   lda >= max(1, n);
 *          if trans = PlasmaConjTrans, lda >= max(1, k).
 *
 * @param[in] beta
 *          The scalar beta.
 *
 * @param[in,out] C
 *          C is a ldc-by-n matrix.
 *          On exit, the uplo part of the matrix is overwritten
 *          by the uplo part of the updated matrix.
 *
 * @param[in] ldc
 *          The leading dimension of the array C. ldc >= max(1, n).
 *
 *******************************************************************************
 *
 * @retval PLASMA_SUCCESS successful exit
 *
 *******************************************************************************
 *
 * @sa plasma_omp_zherk
 * @sa PLASMA_cherk
 *
 ******************************************************************************/
int PLASMA_zherk(PLASMA_enum uplo, PLASMA_enum trans,
                 int n, int k,
                 double alpha, PLASMA_Complex64_t *A, int lda,
                 double beta,  PLASMA_Complex64_t *C, int ldc)
{
    int Am, An;
    int nb;
    int retval;
    int status;

    plasma_desc_t descA;
    plasma_desc_t descC;

    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }

    // Check input arguments.
    if ((uplo != PlasmaUpper) &&
        (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo");
        return -1;
    }
    if ((trans != PlasmaNoTrans) &&
        (trans != PlasmaConjTrans)) {
        plasma_error("illegal value of trans");
        return -2;
    }
    if (trans == PlasmaNoTrans) {
        Am = n; An = k;
    }
    else {
        Am = k; An = n;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -3;
    }
    if (k < 0) {
        plasma_error("illegal value of k");
        return -4;
    }
    if (lda < imax(1, Am)) {
        plasma_error("illegal value of lda");
        return -7;
    }
    if (ldc < imax(1, n)) {
        plasma_error("illegal value of ldc");
        return -10;
    }

    // quick return
    if (n == 0 || ((alpha == 0.0 || k == 0) && beta == 1.0))
        return PLASMA_SUCCESS;

    // Tune
    // status = plasma_tune(PLASMA_FUNC_ZSYRK, n, k, 0);
    // if (status != PLASMA_SUCCESS) {
    //     plasma_error("PLASMA_zherk", "plasma_tune() failed");
    //     return status;
    // }
    nb = plasma->nb;

    // Initialize tile matrix descriptors.
    descA = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, Am, An, 0, 0, Am, An);

    descC = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, n, n, 0, 0, n, n);

    // Allocate matrices in tile layout.
    retval = plasma_desc_mat_alloc(&descA);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_desc_mat_alloc() failed");
        return retval;
    }
    retval = plasma_desc_mat_alloc(&descC);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_desc_mat_alloc() failed");
        plasma_desc_mat_free(&descA);
        return retval;
    }

    // Create sequence.
    plasma_sequence_t *sequence = NULL;
    retval = plasma_sequence_create(&sequence);
    if (retval != PLASMA_SUCCESS) {
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
        PLASMA_zcm2ccrb_Async(C, ldc, &descC, sequence, &request);

        // Call the tile async function.
        plasma_omp_zherk(uplo, trans,
                         alpha, &descA,
                         beta,  &descC,
                         sequence, &request);

        // Translate back to LAPACK layout.
        PLASMA_zccrb2cm_Async(&descC, C, ldc, sequence, &request);
    }
    // implicit synchronization

    // Free matrices in tile layout.
    plasma_desc_mat_free(&descA);
    plasma_desc_mat_free(&descC);

    // Return status.
    status = sequence->status;
    plasma_sequence_destroy(sequence);
    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_herk
 *
 *  Performs rank k update.
 *  Non-blocking tile version of PLASMA_zherk().
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
 *          - PlasmaNoTrans:   \f[ C = \alpha A \times A^H + \beta C; \f]
 *          - PlasmaConjTrans: \f[ C = \alpha A^H \times A + \beta C. \f]
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
 *          request->status should never be set to PLASMA_SUCCESS (the
 *          initial values) since another async call may be setting a
 *          failure value at the same time.
 *
 *******************************************************************************
 *
 * @sa PLASMA_zherk
 * @sa plasma_omp_zherk
 * @sa plasma_omp_cherk
 * @sa plasma_omp_dherk
 * @sa plasma_omp_sherk
 *
 ******************************************************************************/
void plasma_omp_zherk(PLASMA_enum uplo, PLASMA_enum trans,
                      double alpha, plasma_desc_t *A,
                      double beta,  plasma_desc_t *C,
                      plasma_sequence_t *sequence, plasma_request_t *request)
{
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
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if ((trans != PlasmaNoTrans) &&
        (trans != PlasmaConjTrans)) {
        plasma_error("illegal value of trans");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (plasma_desc_check(A) != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        plasma_error("invalid A");
        return;
    }
    if (plasma_desc_check(C) != PLASMA_SUCCESS) {
        plasma_error("invalid C");
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

    // quick return
    int k = trans == PlasmaNoTrans ? A->n : A->m;
    PLASMA_Complex64_t zzero = (PLASMA_Complex64_t)0.0;
    PLASMA_Complex64_t zone  = (PLASMA_Complex64_t)1.0;

    if (C->m == 0 || ((alpha == zzero || k == 0) && beta == zone))
        return;

    // Call the parallel function.
    plasma_pzherk(uplo, trans,
                  alpha, *A,
                  beta, *C,
                  sequence, request);
}
