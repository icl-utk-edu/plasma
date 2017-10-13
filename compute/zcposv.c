/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee,  US,
 *  University of Manchester, UK.
 *
 * @precisions mixed zc -> ds
 *
 **/

#include "plasma.h"
#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_tuning.h"
#include "plasma_types.h"
#include "core_lapack.h"

#include <math.h>
#include <omp.h>
#include <stdbool.h>

/***************************************************************************//**
 *
 * @ingroup plasma_posv
 *
 *  Computes the solution to a system of linear equations A * X = B, where A is
 *  an n-by-n Hermitian positive definite matrix and X and B are n-by-nrhs matrices.
 *
 *  plasma_zcposv first factorizes the matrix using plasma_cpotrf and uses
 *  this factorization within an iterative refinement procedure to produce a
 *  solution with COMPLEX*16 normwise backward error quality (see below). If
 *  the approach fails the method falls back to a COMPLEX*16 factorization and
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
 *  for all the RHS we have: Rnorm < sqrt(n)*Xnorm*Anorm*eps, where:
 *
 *  - iter is the number of the current iteration in the iterative refinement
 *     process
 *  - Rnorm is the Infinity-norm of the residual
 *  - Xnorm is the Infinity-norm of the solution
 *  - Anorm is the Infinity-operator-norm of the matrix A
 *  - eps is the machine epsilon returned by DLAMCH('Epsilon').
 *  The values itermax is fixed to 30.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          Specifies whether the matrix A is upper or lower triangular:
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
 * @param[in,out] pA
 *          The n-by-n Hermitian positive definite coefficient matrix A.
 *          If uplo = PlasmaUpper, the leading n-by-n upper triangular part of
 *          A contains the upper triangular part of the matrix A, and the
 *          strictly lower triangular part of A is not referenced.
 *          If uplo = PlasmaLower, the leading n-by-n lower triangular part of
 *          A contains the lower triangular part of the matrix A, and the
 *          strictly upper triangular part of A is not referenced.
 *          On exit, contains the lower Cholesky factor matrix L,
 *          if uplo == PlasmaLower and upper Cholesky factor conj(L^T),
 *          otherwise.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,n).
 *
 * @param[in] pB
 *          The n-by-nrhs matrix of right hand side matrix B.
 *          This matrix remains unchanged.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1,n).
 *
 * @param[out] pX
 *          If return value = 0, the n-by-nrhs solution matrix X.
 *
 * @param[in] ldx
 *          The leading dimension of the array X. ldx >= max(1,n).
 *
 * @param[out] iter
 *          The number of the iterations in the iterative refinement
 *          process, needed for the convergence. If failed, it is set
 *          to be -(1+itermax), where itermax = 30.
 *
 *******************************************************************************
 *
 * @retval PlasmaSuccess successful exit
 *
 *******************************************************************************
 *
 * @sa plasma_omp_zcposv
 * @sa plasma_dsposv
 * @sa plasma_zposv
 *
 ******************************************************************************/
int plasma_zcposv(plasma_enum_t uplo, int n, int nrhs,
                  plasma_complex64_t *pA, int lda,
                  plasma_complex64_t *pB, int ldb,
                  plasma_complex64_t *pX, int ldx, int *iter)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    // Check input arguments.
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

    // quick return
    *iter = 0;
    if (imin(n, nrhs) == 0)
        return PlasmaSuccess;

    // Tune parameters.
    if (plasma->tuning)
        plasma_tune_potrf(plasma, PlasmaComplexFloat, n);

    // Set tiling parameters.
    int nb = plasma->nb;

    // Create tile matrices.
    plasma_desc_t A;
    plasma_desc_t B;
    plasma_desc_t X;
    int retval;
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        n, n, 0, 0, n, n, &A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        return retval;
    }
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        n, nrhs, 0, 0, n, nrhs, &B);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        plasma_desc_destroy(&A);
        return retval;
    }
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        n, nrhs, 0, 0, n, nrhs, &X);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        plasma_desc_destroy(&A);
        plasma_desc_destroy(&B);
        return retval;
    }

    // Create additional tile matrices.
    plasma_desc_t R, As, Xs;
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        B.m, B.n, 0, 0, B.m, B.n, &R);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        plasma_desc_destroy(&A);
        plasma_desc_destroy(&B);
        plasma_desc_destroy(&X);
        return retval;
    }
    retval = plasma_desc_general_create(PlasmaComplexFloat, nb, nb,
                                        A.m, A.n, 0, 0, A.m, A.n, &As);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        plasma_desc_destroy(&A);
        plasma_desc_destroy(&B);
        plasma_desc_destroy(&X);
        plasma_desc_destroy(&R);
        return retval;
    }
    retval = plasma_desc_general_create(PlasmaComplexFloat, nb, nb,
                                        X.m, X.n, 0, 0, X.m, X.n, &Xs);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        plasma_desc_destroy(&A);
        plasma_desc_destroy(&B);
        plasma_desc_destroy(&X);
        plasma_desc_destroy(&R);
        plasma_desc_destroy(&As);
        return retval;
    }

    // Allocate tiled workspace for Infinity norm calculations.
    size_t lwork = imax((size_t)A.nt*A.n+A.n, (size_t)X.mt*X.n+(size_t)R.mt*R.n);
    double *work  = (double*)malloc(((size_t)lwork)*sizeof(double));
    double *Rnorm = (double*)malloc(((size_t)R.n)*sizeof(double));
    double *Xnorm = (double*)malloc(((size_t)X.n)*sizeof(double));

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
        // Translate matrices to tile layout.
        plasma_omp_zge2desc(pA, lda, A, &sequence, &request);
        plasma_omp_zge2desc(pB, ldb, B, &sequence, &request);

        // Call tile async function.
        plasma_omp_zcposv(uplo, A, B, X, As, Xs, R, work, Rnorm, Xnorm,
                          iter, &sequence, &request);

        // Translate back to LAPACK layout.
        plasma_omp_zdesc2ge(X, pX, ldx, &sequence, &request);
    }
    // implicit synchronization

    // Free matrices in tile layout.
    plasma_desc_destroy(&A);
    plasma_desc_destroy(&B);
    plasma_desc_destroy(&X);
    plasma_desc_destroy(&R);
    plasma_desc_destroy(&As);
    plasma_desc_destroy(&Xs);
    free(work);
    free(Rnorm);
    free(Xnorm);

    // Return status.
    int status = sequence.status;
    return status;
}


// Checks, that convergence criterion is true for all columns of R and X
static bool conv(double *Rnorm, double *Xnorm, int n, double cte)
{
    bool value = true;

    for (int i = 0; i < n; i++) {
        if (Rnorm[i] > Xnorm[i] * cte) {
            value = false;
            break;
        }
    }

    return value;
}


/***************************************************************************//**
 *
 * @ingroup plasma_posv
 *
 *  Solves a Hermitian positive definite system using iterative refinement
 *  with the Cholesky factor computed using plasma_cpotrf.
 *  Non-blocking tile version of plasma_zcposv().
 *  Operates on matrices stored by tiles.
 *  All matrices are passed through descriptors.
 *  All dimensions are taken from the descriptors.
 *  Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          Specifies whether the matrix A is upper or lower triangular:
 *          - PlasmaUpper: Upper triangle of A is stored;
 *          - PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] A
 *          Descriptor of matrix A.
 *
 * @param[in] B
 *          Descriptor of matrix B.
 *
 * @param[in,out] X
 *          Descriptor of matrix X.
 *
 * @param[out] As
 *          Descriptor of auxiliary matrix A in single complex precision.
 *
 * @param[out] Xs
 *          Descriptor of auxiliary matrix X in single complex precision.
 *
 * @param[out] R
 *          Descriptor of auxiliary remainder matrix R.
 *
 * @param[out] work
 *          Workspace needed to compute infinity norm of the matrix A.
 *
 * @param[out] Rnorm
 *          Workspace needed to store the max value in each of resudual vectors.
 *
 * @param[out] Xnorm
 *          Workspace needed to store the max value in each of currenct solution
 *          vectors.
 *
 * @param[out] iter
 *          The number of the iterations in the iterative refinement
 *          process, needed for the convergence. If failed, it is set
 *          to be -(1+itermax), where itermax = 30.
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
 * @sa plasma_zcposv
 * @sa plasma_omp_dsposv
 * @sa plasma_omp_zposv
 *
 ******************************************************************************/
void plasma_omp_zcposv(plasma_enum_t uplo,
                       plasma_desc_t A,  plasma_desc_t B,  plasma_desc_t X,
                       plasma_desc_t As, plasma_desc_t Xs, plasma_desc_t R,
                       double *work, double *Rnorm, double *Xnorm, int *iter,
                       plasma_sequence_t *sequence,
                       plasma_request_t  *request)
{
    const int itermax = 30;
    const plasma_complex64_t zmone = -1.0;
    const plasma_complex64_t zone  =  1.0;
    *iter = 0;

    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }

    // Check input arguments.
    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        plasma_error("illegal value of uplo");
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
    if (plasma_desc_check(X) != PlasmaSuccess) {
        plasma_error("invalid X");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (plasma_desc_check(As) != PlasmaSuccess) {
        plasma_error("invalid As");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (plasma_desc_check(Xs) != PlasmaSuccess) {
        plasma_error("invalid Xs");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (plasma_desc_check(R) != PlasmaSuccess) {
        plasma_error("invalid R");
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
    if (A.n == 0 || B.n == 0)
        return;

    // workspace for dzamax
    double *workX = work;
    double *workR = &work[X.mt*X.n];

    // Compute some constants.
    double cte;
    double eps = LAPACKE_dlamch_work('E');
    double Anorm;
    plasma_pzlanhe(PlasmaInfNorm, uplo, A, work, &Anorm, sequence, request);

    // Convert B from double to single precision, store result in Xs.
    plasma_pzlag2c(B, Xs, sequence, request);

    // Convert A from double to single precision, store result in As.
    // TODO: need zlat2c
    plasma_pzlag2c(A, As, sequence, request);

    // Compute the Cholesky factorization of As.
    plasma_pcpotrf(uplo, As, sequence, request);

    // Solve the system As * Xs = Bs.
    plasma_pctrsm(PlasmaLeft, uplo,
                  uplo == PlasmaUpper ? PlasmaConjTrans : PlasmaNoTrans,
                  PlasmaNonUnit, 1.0, As, Xs, sequence, request);
    plasma_pctrsm(PlasmaLeft, uplo,
                  uplo == PlasmaUpper ? PlasmaNoTrans : PlasmaConjTrans,
                  PlasmaNonUnit, 1.0, As, Xs, sequence, request);

    // Convert Xs to double precision.
    plasma_pclag2z(Xs, X, sequence, request);

    // Compute R = B - A * X.
    plasma_pzlacpy(PlasmaGeneral, PlasmaNoTrans, B, R, sequence, request);
    plasma_pzhemm(PlasmaLeft, uplo, zmone, A, X, zone, R, sequence, request);

    // Check whether the nrhs normwise backward error satisfies the
    // stopping criterion. If yes, set iter=0 and return.
    plasma_pdzamax(PlasmaColumnwise, X, workX, Xnorm, sequence, request);
    plasma_pdzamax(PlasmaColumnwise, R, workR, Rnorm, sequence, request);

    #pragma omp taskwait
    {
        cte = Anorm * eps * sqrt((double)A.n);

        if (conv(Rnorm, Xnorm, R.n, cte)) {
           *iter = 0;
            return;
        }
    }

    // iterative refinement
    for (int iiter = 0; iiter < itermax; iiter++) {
        // Convert R from double to single precision, store result in Xs.
        plasma_pzlag2c(R, Xs, sequence, request);

        // Solve the system As * Xs = Rs.
        plasma_pctrsm(PlasmaLeft, uplo,
                      uplo == PlasmaUpper ? PlasmaConjTrans : PlasmaNoTrans,
                      PlasmaNonUnit, 1.0, As, Xs, sequence, request);
        plasma_pctrsm(PlasmaLeft, uplo,
                      uplo == PlasmaUpper ? PlasmaNoTrans : PlasmaConjTrans,
                      PlasmaNonUnit, 1.0, As, Xs, sequence, request);

        // Convert Xs back to double precision and update the current iterate.
        plasma_pclag2z(Xs, R, sequence, request);
        plasma_pzgeadd(PlasmaNoTrans, zone, R, zone, X, sequence, request);

        // Compute R = B - A * X.
        plasma_pzlacpy(PlasmaGeneral, PlasmaNoTrans, B, R, sequence, request);
        plasma_pzhemm(PlasmaLeft, uplo, zmone, A, X, zone, R,
                      sequence, request);

        // Check whether nrhs normwise backward error satisfies the
        // stopping criterion. If yes, set iter = iiter > 0 and return.
        plasma_pdzamax(PlasmaColumnwise, X, workX, Xnorm, sequence, request);
        plasma_pdzamax(PlasmaColumnwise, R, workR, Rnorm, sequence, request);

        #pragma omp taskwait
        {
            if (conv(Rnorm, Xnorm, R.n, cte)) {
               *iter = iiter+1;
                return;
            }
        }
    }

    // If we are at this place of the code, this is because we have performed
    // iter = itermax iterations and never satisfied the stopping criterion,
    // set up the iter flag accordingly and follow up with double precision
    // routine.
    *iter = -itermax - 1;

    // Compute Cholesky factorization of A.
    plasma_pzpotrf(uplo, A, sequence, request);

    // Solve the system A * X = B.
    plasma_pzlacpy(PlasmaGeneral, PlasmaNoTrans, B, X, sequence, request);

    plasma_pztrsm(PlasmaLeft, uplo,
                  uplo == PlasmaUpper ? PlasmaConjTrans : PlasmaNoTrans,
                  PlasmaNonUnit, 1.0, A, X, sequence, request);

    plasma_pztrsm(PlasmaLeft, uplo,
                  uplo == PlasmaUpper ? PlasmaNoTrans : PlasmaConjTrans,
                  PlasmaNonUnit, 1.0, A, X, sequence, request);
}
