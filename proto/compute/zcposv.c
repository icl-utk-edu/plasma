/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 *
 *
 **/

// @precisions mixed zc -> ds

#include "plasma_types.h"
#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_z.h"

#ifdef PLASMA_WITH_MKL
    #include <mkl_lapacke.h>
#else
    #include <lapacke.h>
#endif

/***************************************************************************//**
 *
 * @ingroup plasma_zcposv
 *
 *  PLASMA_zcposv - Computes the solution to a system of linear equations
 *  A * X = B, where A is an n-by-n symmetric positive definite (or Hermitian
 *  positive definite in the complex case) matrix and X and B are n-by-nrhs
 *  matrices. The Cholesky decomposition is used to factor A as
 *
 *    A = U^H * U, if uplo = PlasmaUpper, or
 *    A = L * L^H, if uplo = PlasmaLower,
 *
 *  where U is an upper triangular matrix and L is a lower triangular matrix.
 *  The factored form of A is then used to solve the system of equations
 *  A * X = B.
 *
 *  PLASMA_zcposv first attempts to factorize the matrix in COMPLEX and use
 *  this factorization within an iterative refinement procedure to produce a
 *  solution with COMPLEX*16 normwise backward error quality (see below). If
 *  the approach fails the method switches to a COMPLEX*16 factorization and
 *  solve.
 *
 *  The iterative refinement is not going to be a winning strategy if
 *  the ratio COMPLEX performance over COMPLEX*16 performance is too
 *  small. A reasonable strategy should take the number of right-hand
 *  sides and the size of the matrix into account. This might be done
 *  with a call to ILAENV in the future. Up to now, we always try
 *  iterative refinement.
 *
 *  The iterative refinement process is stopped if iter > itermax or
 *  for all the RHS we have: Rnorm < n*Xnorm*Anorm*eps*BWDmax
 *  where:
 *
 *  - iter is the number of the current iteration in the iterative refinement
 *  process
 *  - Rnorm is the infinity-norm of the residual
 *  - Xnorm is the infinity-norm of the solution
 *  - Anorm is the infinity-operator-norm of the matrix A
 *  - eps is the machine epsilon returned by DLAMCH('Epsilon').
 *
 *  Actually, in its current state (PLASMA 3.0.0), the test is slightly
 *  relaxed.
 *
 *  The values itermax and BWDmax are fixed to 30 and 1.0D+00 respectively.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          Specifies whether the matrix A is upper triangular or lower
 *          triangular:
 *          - PlasmaUpper: Upper triangle of A is stored;
 *          - PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] n
 *          The number of linear equations, i.e., the order of the matrix A.
 *          n >= 0.
 *
 * @param[in] nrhs
 *          The number of right hand sides, i.e., the number of columns of the
 *          matrix B. nrhs >= 0.
 *
 * @param[in] A
 *          The n-by-n symmetric positive definite (or Hermitian) coefficient
 *          matrix A. If uplo = PlasmaUpper, the leading n-by-n upper
 *          triangular part of A contains the upper triangular part of the
 *          matrix A, and the strictly lower triangular part of A is not
 *          referenced. If uplo = 'L', the leading n-by-n lower triangular
 *          part of A contains the lower triangular part of the matrix A, and
 *          the strictly upper triangular part of A is not referenced. This
 *          matrix is not modified.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,n).
 *
 * @param[in] B
 *          The n-by-nrhs matrix of right hand side matrix B.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1,n).
 *
 * @param[out] X
 *          If return value = 0, the n-by-nrhs solution matrix X.
 *
 * @param[in] ldx
 *          The leading dimension of the array X. ldx >= max(1,n).
 *
 * @param[out] iter
 *          The number of the current iteration in the iterative refinement
 *          process.
 *
 *******************************************************************************
 *
 * @retval PLASMA_SUCCESS successful exit
 * @retval <0 if -i, the i-th argument had an illegal value
 * @retval >0 if i, the leading minor of order i of A is not positive definite,
 *            so the factorization could not be completed, and the solution has
 *            not been computed.
 *
 *******************************************************************************
 *
 * @sa PLASMA_zcposv_Tile_Async
 * @sa PLASMA_dsposv
 * @sa PLASMA_zposv
 *
 ******************************************************************************/
int PLASMA_zcposv(PLASMA_enum uplo, int n, int nrhs,
                  PLASMA_Complex64_t *A, int lda,
                  PLASMA_Complex64_t *B, int ldb,
                  PLASMA_Complex64_t *X, int ldx, int *iter)
{
    int retval;
    int nb;
    int status;
    PLASMA_desc descA;
    PLASMA_desc descB;
    PLASMA_desc descX;
    plasma_context_t *plasma;
    PLASMA_sequence  *sequence = NULL;
    PLASMA_request    request = PLASMA_REQUEST_INITIALIZER;

    // Get PLASMA context
    plasma = plasma_context_self();

    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }

    *iter = 0;

    // Check input arguments
    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        plasma_error("illegal value of uplo");
        return -1;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -2;
    }
    if (nrhs < 0) {
        plasma_error("illegal value of nrhs");
        return -3;
    }
    if (lda < imax(1, n)) {
        plasma_error("illegal value of lda");
        return -5;
    }
    if (ldb < imax(1, n)) {
        plasma_error("illegal value of ldb");
        return -7;
    }
    if (ldx < imax(1, n)) {
        plasma_error("illegal value of ldx");
        return -9;
    }

    // Quick return - currently NOT equivalent to LAPACK's
    // LAPACK does not have such check for ZCPOSV
    if (imin(n, nrhs) == 0)
        return PLASMA_SUCCESS;

    // Tune nb depending on m, n & nrhs; Set nbnbsize
    /*
    if (plasma_tune(PLASMA_FUNC_ZCPOSV, n, n, nrhs) != PLASMA_SUCCESS) {
        plasma_error("plasma_tune() failed");
        return status;
    }
    */
    nb = plasma->nb;

    // Initialise matrix descriptors
    descA = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, lda, n, 0, 0, lda, n);

    descB = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, ldb, nrhs, 0, 0, ldb, nrhs);

    descX = plasma_desc_init(PlasmaComplexDouble, nb, nb,
                             nb*nb, ldx, nrhs, 0, 0, ldx, nrhs);

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

    retval = plasma_desc_mat_alloc(&descX);

    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_desc_mat_alloc() failed");
        plasma_desc_mat_free(&descA);
        plasma_desc_mat_free(&descB);
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
        PLASMA_zcm2ccrb_Async(B, ldb, &descB, sequence, &request);
        PLASMA_zcm2ccrb_Async(X, ldx, &descX, sequence, &request);

        // Call the tile async interface
        PLASMA_zcposv_Tile_Async(uplo, &descA, &descB, &descX,
                                 iter, sequence, &request);
    
        // Revert matrices to LAPACK layout
        PLASMA_zccrb2cm_Async(&descA, A, lda, sequence, &request);
        PLASMA_zccrb2cm_Async(&descB, B, ldb, sequence, &request);
        PLASMA_zccrb2cm_Async(&descX, X, ldx, sequence, &request);

    } // pragma omp parallel block closed
    // implicit synchronization

    // Free matrices in tile layout
    plasma_desc_mat_free(&descA);
    plasma_desc_mat_free(&descB);
    plasma_desc_mat_free(&descX);

    // Destroy sequence
    plasma_sequence_destroy(sequence);

    // Return status
    status = sequence->status;
    return status;
}

/***************************************************************************//**
 *
 * @ingroup PLASMA_zcposv
 *
 *  Solves a symmetric positive definite or Hermitian positive definite system
 *  of linear equations using the Cholesky factorization and mixed-precision
 *  iterative refinement. Non-blocking equivalent of PLASMA_zcposv(). May
 *  return before the computation is finished. Allows for pipelining of
 *  operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] A
 *          Descriptor of matrix A.
 *
 * @param[in,out] X
 *          Descriptor of matrix X.
 *
 * @param[in] B
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
 * @sa PLASMA_zcposv
 * @sa PLASMA_zcposv_Tile
 * @sa PLASMA_dsposv_Tile_Async
 * @sa PLASMA_zposv_Tile_Async
 *
 ******************************************************************************/
void PLASMA_zcposv_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B,
                              PLASMA_desc *X, int *iter,
                              PLASMA_sequence *sequence,
                              PLASMA_request  *request)
{
    int n, nb;
    PLASMA_desc descA;
    PLASMA_desc descB;
    PLASMA_desc descX;
    plasma_context_t *plasma;
    // @todo Use workspace once LAPACKE functions are removed
    // PLASMA_workspace work;
    PLASMA_desc descR, descAs, descXs;
    PLASMA_enum transA;

    const int itermax = 30;
    const double bwdmax = 1.0;
    const PLASMA_Complex64_t negone = -1.0;
    const PLASMA_Complex64_t one = 1.0;
    int iiter, retval;
    double Anorm = 0.0, Rnorm = 0.0, Xnorm = 0.0;
    double cte, eps;
    *iter = 0;

    PLASMA_enum mtrxLayout = LAPACK_COL_MAJOR;
    char        mtrxNorm   = 'I';

    // Get PLASMA context
    plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
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

    // Check descriptors for correctness
    if (plasma_desc_check(A) != PLASMA_SUCCESS) {
        plasma_error("invalid descriptor of matrix A");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    else {
        descA = *A;
    }

    if (plasma_desc_check(B) != PLASMA_SUCCESS) {
        plasma_error("invalid descriptor of matrix B");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    else {
        descB = *B;
    }

    if (plasma_desc_check(X) != PLASMA_SUCCESS) {
        plasma_error("invalid descriptor of matrix X");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    else {
        descX = *X;
    }

    // Check input arguments
    if (descA.nb != descA.mb || descB.nb != descB.mb || descX.nb != descX.mb) {
        plasma_error("only square tiles supported");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        plasma_error("illegal value of uplo");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    // Quick return - currently NOT equivalent to LAPACK's
    // LAPACK does not have such check for DPOSV

    if (A->m == 0 || A->n == 0 || B->m == 0 || B->n == 0 || X->m == 0 || X->n == 0)
        return;

    // Set n, nb
    n  = descA.m;
    nb = descA.nb;

    // @todo Use workspace once LAPACKE functions are removed
    // Allocate workspace
    /*
    size_t lwork = imax(1,n);
    retval = plasma_workspace_alloc(&work, lwork, PlasmaComplexDouble);

    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_workspace_alloc() failed");
        return retval;
    }
    */

    // Initialise additional matrix descriptors
    descR  = plasma_desc_init(PlasmaComplexDouble, nb, nb, nb*nb,
                              descB.m, descB.n, 0, 0, descB.m, descB.n);

    descAs = plasma_desc_init(PlasmaComplexFloat,  nb, nb, nb*nb,
                              descA.m, descA.n, 0, 0, descA.m, descA.n);

    descXs = plasma_desc_init(PlasmaComplexFloat, nb, nb, nb*nb,
                              descX.m, descX.n, 0, 0, descX.m, descX.n);

    // Allocate additional matrices in tile layout
    retval = plasma_desc_mat_alloc(&descR);

    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_desc_mat_alloc() failed");
        plasma_request_fail(sequence, request, PLASMA_ERR_OUT_OF_RESOURCES);
        return;
    }

    retval = plasma_desc_mat_alloc(&descAs);

    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_desc_mat_alloc() failed");
        plasma_request_fail(sequence, request, PLASMA_ERR_OUT_OF_RESOURCES);
        plasma_desc_mat_free(&descR);
        return;
    }

    retval = plasma_desc_mat_alloc(&descXs);

    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_desc_mat_alloc() failed");
        plasma_request_fail(sequence, request, PLASMA_ERR_OUT_OF_RESOURCES);
        plasma_desc_mat_free(&descR);
        plasma_desc_mat_free(&descAs);
        return;
    }

    // Compute constants
    // @todo Convert to OpenMP
    // plasma_pzlanhe(PlasmaInfNorm, uplo, descA, work, Anorm, sequence, request);
    // @todo Verify if infinity norm of matrix A will remain the same for a matrix in tile layout
    Anorm = LAPACKE_zlanhe(mtrxLayout, mtrxNorm, uplo, n, descA.mat, descA.lm);
    eps = LAPACKE_dlamch_work('e');

    // Convert B from double to single precision, store result in Xs
    // @todo Convert to OpenMP
    // plasma_pzlag2c(descB, descXs);
    LAPACKE_zlag2c(mtrxLayout, descB.lm, descB.ln, descB.mat, descB.lm,
                   descXs.mat, descXs.lm);

    // @todo Remove once above LAPACKE function is removed
    if (sequence->status != PLASMA_SUCCESS) {
        plasma_error("unable to convert matrix B from double to single precision");
        plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);
        return;
    }

    // Convert A from double to single precision, store result in As
    // @todo Convert to OpenMP
    // plasma_pzlag2c(descA, descAs);
    LAPACKE_zlag2c(mtrxLayout, descA.lm, descA.ln, descA.mat, descA.lm, 
                   descAs.mat, descAs.lm);

    // @todo Remove once above LAPACKE function is removed
    if (sequence->status != PLASMA_SUCCESS) {
        plasma_error("unable to convert matrix A from double to single precision");
        plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);
        return;
    }

    // Compute Cholesky factorization of As
    plasma_pcpotrf(uplo, descAs, sequence, request);

    // Solve system As*Xs = Bs
    // Forward substitution
    transA = (uplo == PlasmaUpper ? PlasmaConjTrans : PlasmaNoTrans);

    plasma_pctrsm(PlasmaLeft, uplo, transA, PlasmaNonUnit,
                 (PLASMA_Complex32_t) 1.0, descAs, descXs, sequence, request);

    // Backward substitution
    transA = (uplo == PlasmaUpper ? PlasmaNoTrans : PlasmaConjTrans);

    plasma_pctrsm(PlasmaLeft, uplo, transA, PlasmaNonUnit,
                 (PLASMA_Complex32_t) 1.0, descAs, descXs, sequence, request);

    // Convert Xs to double precision
    // @todo Convert to OpenMP
    // plasma_pclag2z(descXs, descX);
    LAPACKE_clag2z(mtrxLayout, descXs.lm, descXs.ln, descXs.mat, descXs.lm,
                   descX.mat, descX.lm);

    // Compute residual R = B-A*X
    // @todo Convert to OpenMP
    // plasma_pzlacpy(descB, descR);
    LAPACKE_zlacpy(mtrxLayout, uplo, descB.lm, descB.ln, descB.mat, descB.lm,
                   descR.mat, descR.lm);

    plasma_pzhemm(PlasmaLeft, uplo, negone, descA, descX, one, descR,
                  sequence, request);

    // Check, whether nrhs normwise backward error satisfies the
    // stopping criterion. If yes, return. Note, that iter = 0 is already set
    // @todo Convert to OpenMP
    // plasma_pzlange(PlasmaInfNorm, descX, Xnorm, wrk);
    // plasma_pzlange(PlasmaInfNorm, descR, Rnorm, wrk);
    // @todo Verify if infinity norm of matrices X, R will remain the same for matrices in tile layout
    Xnorm = LAPACKE_zlange(mtrxLayout, mtrxNorm, descX.lm, descX.ln,
                           descX.mat, descX.lm);

    Rnorm = LAPACKE_zlange(mtrxLayout, mtrxNorm, descR.lm, descR.ln,
                           descR.mat, descR.lm);

    // Wait for end of Anorm, Xnorm and Bnorm computations
    // plasma_dynamic_sync();

    cte = Anorm * eps * ((double)n) * bwdmax;

    // The nrhs normwise backward error satisfies the stopping criterion => Exit
    if (Rnorm < Xnorm * cte) {
        // @todo Enable once work array is in use
        // plasma_workspace_free(work);

        // Deallocate memory for additional matrices As, Xs, R
        plasma_desc_mat_free(&descAs);
        plasma_desc_mat_free(&descXs);
        plasma_desc_mat_free(&descR);

        return;
    }

    double *Xmtrx = (double *) malloc(sizeof(double)*descX.lm*descX.ln);
    double *Rmtrx = (double *) malloc(sizeof(double)*descR.lm*descR.ln);

    // Iterative refinement
    for (iiter = 0; iiter < itermax; iiter++) {

        // Convert R from double to single precision, store result in Xs
        // @todo Convert to OpenMP
        // plasma_pzlag2c(descR, descXs);
        LAPACKE_zlag2c(mtrxLayout, descR.lm, descR.ln, descR.mat, descR.lm,
                       descXs.mat, descXs.lm);

        // Solve system As*Xs = Rs
        // Forward substitution
        transA = (uplo == PlasmaUpper ? PlasmaConjTrans : PlasmaNoTrans);

        plasma_pctrsm(PlasmaLeft, uplo, transA, PlasmaNonUnit,
                     (PLASMA_Complex32_t) 1.0, descAs, descXs, sequence, request);

        // Backward substitution
        transA = (uplo == PlasmaUpper ? PlasmaNoTrans : PlasmaConjTrans);

        plasma_pctrsm(PlasmaLeft, uplo, transA, PlasmaNonUnit,
                     (PLASMA_Complex32_t) 1.0, descAs, descXs, sequence, request);

        // Revert Xs to double precision, store result in R
        // Update current iteration
        // @todo Convert to OpenMP
        // plasma_pclag2z(descXs, descR);
        LAPACKE_clag2z(mtrxLayout, descXs.lm, descXs.ln, descXs.mat, descXs.lm,
                       descR.mat, descR.lm);

        // Update solution matrix X:=X+R
        // @todo Convert to OpenMP
        // plasma_pztradd(PlasmaFull, PlasmaNoTrans, (PLASMA_Complex64_t) one,
        //                descR, (PLASMA_Complex64_t) 1.0, descX, sequence, request);
        Xmtrx = descX.mat;
        Rmtrx = descR.mat;

        for (int i = 0; i < descX.lm*descX.ln; i++) {
            Xmtrx[i] = Xmtrx[i] + Rmtrx[i];
        }

        descX.mat = Xmtrx;

        // Compute residual R = B-A*X
        // @todo Convert to OpenMP
        // plasma_pzlacpy(descB, descR);
        LAPACKE_zlacpy(mtrxLayout, uplo, descB.lm, descB.ln, descB.mat, descB.lm,
                       descR.mat, descR.lm);

        plasma_pzhemm(PlasmaLeft, uplo, negone, descA, descX,
                      one, descR, sequence, request);

        // Check, whether nrhs normwise backward error satisfies the
        // stopping criterion. If yes, set iter = iiter > 0 and return
        // @todo Convert to OpenMP
        // plasma_pzlange(PlasmaInfNorm, descX, Xnorm, wrk);
        // plasma_pzlange(PlasmaInfNorm, descR, Rnorm, wrk);
        // @todo Verify if infinity norm of matrices X, R will remain the same
        // for matrices in tile layout
        Xnorm = LAPACKE_zlange(mtrxLayout, mtrxNorm, descX.lm, descX.ln,
                               descX.mat, descX.lm);
    
        Rnorm = LAPACKE_zlange(mtrxLayout, mtrxNorm, descR.lm, descR.ln,
                               descR.mat, descR.lm);

        // Wait for the end of Xnorm and Rnorm computations
        // plasma_dynamic_sync();

        // The nrhs normwise backward error satisfies the stopping criterion =>
        // Exit
        if (Rnorm < Xnorm * cte) {
            *iter = iiter;

            // @todo Enable once work array is in use
            // plasma_workspace_free(work);

            // Deallocate memory for additional matrices As, Xs, R
            plasma_desc_mat_free(&descAs);
            plasma_desc_mat_free(&descXs);
            plasma_desc_mat_free(&descR);

            return;
        }

        Xnorm = 0.0; Rnorm = 0.0;
    }

    // Itermax iterations have been performed, the stopping criterion has not
    // been satisfied => Update iter counter and execute standard double
    // precision routine
    *iter = -itermax - 1;

    // @todo Enable once work array is in use
    // plasma_workspace_free(work);

    // Deallocate memory for additional matrices As, Xs, R
    plasma_desc_mat_free(&descAs);
    plasma_desc_mat_free(&descXs);
    plasma_desc_mat_free(&descR);

    // Solve linear system A*X=B in double precision
    // Factorize matrix A using Cholesky decomposition
    plasma_pzpotrf(uplo, descA, sequence, request);

    // @todo Convert to OpenMP
    // plasma_pzlacpy(descB, descX);
    LAPACKE_zlacpy(mtrxLayout, uplo, descB.lm, descB.ln, descB.mat, descB.lm,
                   descX.mat, descX.lm);

    transA = (uplo == PlasmaUpper ? PlasmaConjTrans : PlasmaNoTrans);

    // Perform forward substitution L*d=b
    plasma_pztrsm(PlasmaLeft, uplo, transA, PlasmaNonUnit, (PLASMA_Complex64_t) 1.0,
                  descA, descX, sequence, request);

    transA = (uplo == PlasmaUpper ? PlasmaNoTrans : PlasmaConjTrans);

    // Perform backward substitution U*x=d
    plasma_pztrsm(PlasmaLeft, uplo, transA, PlasmaNonUnit, (PLASMA_Complex64_t) 1.0,
                  descA, descX, sequence, request);

    return;
}
