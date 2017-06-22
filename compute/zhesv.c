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
#include <string.h>

/***************************************************************************//**
 *
 * @ingroup plasma_hesv
 *
 *  Solves a system of linear equations A * X = B with LTLt factorization.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of A is stored;
 *          - PlasmaLower: Lower triangle of A is stored.
 *            TODO: only support Lower for now
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in] nrhs
 *          The number of right hand sides, i.e., the number of
 *          columns of the matrix B. nrhs >= 0.
 *
 * @param[in,out] A
 *          Details of the LTL factorization of the Hermitian matrix A,
 *          as computed by plasma_zhetrf.
 *
 * @param[in] lda
 *          The leading dimension of the array A.
 *
 * @param[in,out] T
 *          Details of the LU factorization of the band matrix A, as
 *          computed by plasma_zgbtrf.
 *
 * @param[in] ldt
 *          The leading dimension of the array T.
 *
 * @param[in] ipiv
 *          The pivot indices used for zhetrf; for 1 <= i <= min(m,n),
 *          row i of the matrix was interchanged with row ipiv(i).
 *
 * @param[in] ipiv2
 *          The pivot indices used for zgbtrf; for 1 <= i <= min(m,n),
 *          row i of the matrix was interchanged with row ipiv(i).
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
 * @retval PlasmaSuccess successful exit
 * @retval  < 0 if -i, the i-th argument had an illegal value
 *
 *******************************************************************************
 *
 * @sa plasma_omp_zhesv
 * @sa plasma_chesv
 * @sa plasma_dsysv
 * @sa plasma_ssysv
 * @sa plasma_zhetrf
 * @sa plasma_zhetrs
 *
 ******************************************************************************/
int plasma_zhesv(plasma_enum_t uplo, int n, int nrhs,
                 plasma_complex64_t *pA, int lda,
                 int *ipiv,
                 plasma_complex64_t *pT, int ldt,
                 int *ipiv2,
                 plasma_complex64_t *pB,  int ldb)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    // Check input arguments.
    if (//(uplo != PlasmaUpper) &&
        (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo (Upper not supported, yet)");
        return -1;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -2;
    }
    if (nrhs < 0) {
        plasma_error("illegal value of nrhs");
        return -5;
    }
    if (lda < imax(1, n)) {
        plasma_error("illegal value of lda");
        return -7;
    }
    if (ldb < imax(1, n)) {
        plasma_error("illegal value of ldb");
        return -10;
    }

    // quick return
    if (imax(n, nrhs) == 0)
        return PlasmaSuccess;

    // Tune parameters.
    if (plasma->tuning)
        plasma_tune_hetrf(plasma, PlasmaComplexDouble, n);

    // Set tiling parameters.
    int nb = plasma->nb;

    // Adjust max number of panel threads
    int max_panel_threads_gbtrf = 1;
    int max_panel_threads_hetrf = 1;
    if (plasma->max_panel_threads > 3) {
        max_panel_threads_gbtrf = 2;
    }
    max_panel_threads_hetrf = imax(1, plasma->max_panel_threads - max_panel_threads_gbtrf);
    plasma->max_panel_threads  = max_panel_threads_hetrf;

    // Initialize barrier.
    plasma_barrier_init(&plasma->barrier);

    // Initialize tile matrix descriptors.
    plasma_desc_t A;
    plasma_desc_t T;
    plasma_desc_t B;
    int tku = (nb+nb+nb-1)/nb; // number of tiles in upper band (not including diagonal)
    int tkl = (nb+nb-1)/nb;    // number of tiles in lower band (not including diagonal)
    int lm  = (tku+tkl+1)*nb;  // since we use zgetrf on panel, we pivot back within panel.
                               // this could fill the last tile of the panel,
                               // and we need extra NB space on the bottom
    int retval;
    retval = plasma_desc_triangular_create(PlasmaComplexDouble, uplo, nb, nb,
                                           n, n, 0, 0, n, n, &A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        return retval;
    }
    retval = plasma_desc_general_band_create(PlasmaComplexDouble, PlasmaGeneral,
                                             nb, nb, lm, n, 0, 0, n, n, nb, nb,
                                             &T);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_band_create() failed");
        return retval;
    }
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        n, nrhs, 0, 0, n, nrhs, &B);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        plasma_desc_destroy(&A);
        return retval;
    }

    // Create workspace.
    plasma_desc_t W;
    int tot = 3;
    int ldw = (1+(4+tot)*A.mt)*nb; // block column
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        ldw, nb, 0, 0, ldw, nb, &W);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        return retval;
    }

    // Initialize sequence.
    plasma_sequence_t sequence;
    retval = plasma_sequence_init(&sequence);

    // Initialize request.
    plasma_request_t request;
    retval = plasma_request_init(&request);

    // Initialize data.
    memset(T.matrix, 0, ldt*n*sizeof(plasma_complex64_t));
    memset(W.matrix, 0, ldw*nb*sizeof(plasma_complex64_t));
    for (int i = 0; i < nb; i++) ipiv[i] = 1+i;

    // asynchronous block
    #pragma omp parallel
    #pragma omp master
    {
        // Translate to tile layout.
        plasma_omp_ztr2desc(pA, lda, A, &sequence, &request);
        plasma_omp_zpb2desc(pT, ldt, T, &sequence, &request);
        plasma_omp_zge2desc(pB, ldb, B, &sequence, &request);
    }
    // implicit synchronization

    #pragma omp parallel
    #pragma omp master
    {
        // Call the tile async function.
        plasma_omp_zhesv(uplo, A, ipiv, T, ipiv2, B, W, &sequence, &request);
    }
    // implicit synchronization

    #pragma omp parallel
    #pragma omp master
    {
        // Translate back to LAPACK layout.
        plasma_omp_zdesc2ge(B, pB, ldb, &sequence, &request);
    }
    // implicit synchronization

    // Free matrix A in tile layout.
    plasma_desc_destroy(&A);
    plasma_desc_destroy(&T);
    plasma_desc_destroy(&B);
    plasma_desc_destroy(&W);

    // Return status.
    int status = sequence.status;
    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_hesv
 *
 *  Solves a system of linear equations using previously
 *  computed factorization.
 *  Non-blocking tile version of plasma_zhesv().
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
 * @param[in] A
 *          The triangular factor U or L from the Cholesky factorization
 *          A = U^H*U or A = L*L^H, computed by plasma_zpotrf.
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
 *          request->status should never be set to PlasmaSuccess (the
 *          initial values) since another async call may be setting a
 *          failure value at the same time.
 *
 *******************************************************************************
 *
 * @sa plasma_zhesv
 * @sa plasma_omp_zhesv
 * @sa plasma_omp_chesv
 * @sa plasma_omp_dsysv
 * @sa plasma_omp_ssysv
 * @sa plasma_omp_zhetrf
 * @sa plasma_omp_zhetrs
 *
 ******************************************************************************/
void plasma_omp_zhesv(plasma_enum_t uplo,
                      plasma_desc_t A, int *ipiv,
                      plasma_desc_t T, int *ipiv2,
                      plasma_desc_t B,
                      plasma_desc_t W,
                      plasma_sequence_t *sequence,
                      plasma_request_t *request)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }

    // Check input arguments.
    if (//(uplo != PlasmaUpper) &&
        (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo (Upper not supported, yet)");
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
        plasma_fatal_error("NULL sequence");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (request == NULL) {
        plasma_fatal_error("NULL request");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }

    // quick return
    if (A.n == 0 || B.n == 0)
        return;

    // Call the parallel functions.
    plasma_pzhetrf_aasen(uplo, A, ipiv, T, W, sequence, request);
    plasma_pzgbtrf(T, ipiv2, sequence, request);
    // dependency on ipiv
    #pragma omp taskwait
    if (uplo == PlasmaLower) {
        plasma_desc_t vA;
        plasma_desc_t vB;
        // forward-substitution with L
        if (A.m > A.nb) {
            vA = plasma_desc_view(A,
                                  A.nb, 0,
                                  A.m-A.nb, A.n-A.nb);
            vB = plasma_desc_view(B,
                                  B.nb, 0,
                                  B.m-B.nb, B.n);

            plasma_pzgeswp(PlasmaRowwise, B, ipiv, 1, sequence, request);
            #pragma omp taskwait
            plasma_pztrsm(PlasmaLeft, PlasmaLower, PlasmaNoTrans, PlasmaUnit,
                          1.0, vA,
                               vB,
                          sequence, request);
        }
        // solve with band matrix T
        #pragma omp taskwait
        plasma_pztbsm(PlasmaLeft, PlasmaLower, PlasmaNoTrans,
                      PlasmaUnit,
                      1.0, T,
                           B,
                      ipiv2,
                      sequence, request);
        plasma_pztbsm(PlasmaLeft, PlasmaUpper, PlasmaNoTrans,
                      PlasmaNonUnit,
                      1.0, T,
                           B,
                      ipiv2,
                      sequence, request);
        // backward-substitution with L^H
        if (A.m > A.nb) {
            plasma_pztrsm(PlasmaLeft, PlasmaLower, PlasmaConjTrans, PlasmaUnit,
                          1.0, vA,
                               vB,
                          sequence, request);
            #pragma omp taskwait
            plasma_pzgeswp(PlasmaRowwise, B, ipiv, -1, sequence, request);
        }
    }
    else {
        // TODO: upper
    }
}
