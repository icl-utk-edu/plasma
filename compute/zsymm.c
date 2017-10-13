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
 * @ingroup plasma_symm
 *
 *  Performs one of the matrix-matrix operations
 *
 *     \f[ C = \alpha \times A \times B + \beta \times C \f]
 *  or
 *     \f[ C = \alpha \times B \times A + \beta \times C \f]
 *
 *  where alpha and beta are scalars, A is a symmetric matrix and B and
 *  C are m-by-n matrices.
 *
 *******************************************************************************
 *
 * @param[in] side
 *          Specifies whether the symmetric matrix A appears on the
 *          left or right in the operation as follows:
 *          - PlasmaLeft:  \f[ C = \alpha \times A \times B + \beta \times C \f]
 *          - PlasmaRight: \f[ C = \alpha \times B \times A + \beta \times C \f]
 *
 * @param[in] uplo
 *          Specifies whether the upper or lower triangular part of
 *          the symmetric matrix A is to be referenced as follows:
 *          - PlasmaLower:     Only the lower triangular part of the
 *                             symmetric matrix A is to be referenced.
 *          - PlasmaUpper:     Only the upper triangular part of the
 *                             symmetric matrix A is to be referenced.
 *
 * @param[in] m
 *          The number of rows of the matrix C. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix C. n >= 0.
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] pA
 *          A is an lda-by-ka matrix, where ka is m when side = PlasmaLeft,
 *          and is n otherwise. Only the uplo triangular part is referenced.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,ka).
 *
 * @param[in] pB
 *          B is an ldb-by-n matrix, where the leading m-by-n part of
 *          the array B must contain the matrix B.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1,m).
 *
 * @param[in] beta
 *          The scalar beta.
 *
 * @param[in,out] pC
 *          C is an ldc-by-n matrix.
 *          On exit, the array is overwritten by the m-by-n updated matrix.
 *
 * @param[in] ldc
 *          The leading dimension of the array C. ldc >= max(1,m).
 *
 *******************************************************************************
 *
 * @retval  PlasmaSuccess successful exit
 *
 *******************************************************************************
 *
 * @sa plasma_omp_zsymm
 * @sa plasma_csymm
 * @sa plasma_dsymm
 * @sa plasma_ssymm
 *
 ******************************************************************************/
int plasma_zsymm(plasma_enum_t side, plasma_enum_t uplo,
                 int m, int n,
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
    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        plasma_error("illegal value of side");
        return -1;
    }
    if ((uplo != PlasmaLower) && (uplo != PlasmaUpper)) {
        plasma_error("illegal value of uplo");
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

    int am;
    if (side == PlasmaLeft)
        am = m;
    else
        am = n;

    if (lda < imax(1, am)) {
        plasma_error("illegal value of lda");
        return -7;
    }
    if (ldb < imax(1, m)) {
        plasma_error("illegal value of ldb");
        return -9;
    }
    if (ldc < imax(1, m)) {
        plasma_error("illegal value of ldc");
        return -12;
    }

    // quick return
    if (m == 0 || n == 0 || (alpha == 0.0 && beta == 1.0))
        return PlasmaSuccess;

    // Tune parameters.
    if (plasma->tuning)
        plasma_tune_symm(plasma, PlasmaComplexDouble, m, n);

    // Set tiling parameters.
    int nb = plasma->nb;

    // Create tile matrices.
    plasma_desc_t A;
    plasma_desc_t B;
    plasma_desc_t C;
    int retval;
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        am, am, 0, 0, am, am, &A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        return retval;
    }
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        m, n, 0, 0, m, n, &B);
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
        plasma_omp_zsymm(side, uplo,
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
 * @ingroup plasma_symm
 *
 *  Performs symmetric matrix multiplication.
 *  Non-blocking tile version of plasma_zsymm().
 *  May return before the computation is finished.
 *  Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] side
 *          Specifies whether the symmetric matrix A appears on the
 *          left or right in the operation as follows:
 *          - PlasmaLeft:  \f[ C = \alpha \times A \times B + \beta \times C \f]
 *          - PlasmaRight: \f[ C = \alpha \times B \times A + \beta \times C \f]
 *
 * @param[in] uplo
 *          Specifies whether the upper or lower triangular part of
 *          the symmetric matrix A is to be referenced as follows:
 *          - PlasmaLower:     Only the lower triangular part of the
 *                             symmetric matrix A is to be referenced.
 *          - PlasmaUpper:     Only the upper triangular part of the
 *                             symmetric matrix A is to be referenced.
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
 *          (for completion checks and exception handling purposes).
 *
 * @param[out] request
 *          Identifies this function call (for exception handling purposes).
 *
 *******************************************************************************
 *
 * @sa plasma_zsymm
 * @sa plasma_omp_csymm
 * @sa plasma_omp_dsymm
 * @sa plasma_omp_ssymm
 *
 ******************************************************************************/
void plasma_omp_zsymm(plasma_enum_t side, plasma_enum_t uplo,
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
    if ((side != PlasmaLeft) &&
        (side != PlasmaRight)) {
        plasma_error("illegal value of side");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if ((uplo != PlasmaLower) &&
        (uplo != PlasmaUpper)) {
        plasma_error("illegal value of uplo");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (plasma_desc_check(A) != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        plasma_error("invalid A");
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
    if (C.m == 0 || C.n == 0 || ((alpha == 0.0 || A.n == 0) && beta == 1.0))
        return;

    // Call the parallel function.
    plasma_pzsymm(side, uplo,
                  alpha, A,
                         B,
                  beta,  C,
                  sequence, request);
}
