/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
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

/***************************************************************************//**
 * TODO: adjust the documents for band matrix A.
 * @ingroup plasma_gbsv
 *
 *  Computes the solution to a system of linear equations A * X = B, where A is
 *  an n-by-n matrix and X and B are n-by-nrhs matrices.
 *
 *  plasma_zcgesv first factorizes the matrix using plasma_cgetrf and uses
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
 *  for all the RHS we have: Rnorm < sqrt(n)*Xnorm*Anorm*eps*BWDmax
 *  where:
 *
 *  - iter is the number of the current iteration in the iterative refinement
 *     process
 *  - Rnorm is the Infinity-norm of the residual
 *  - Xnorm is the Infinity-norm of the solution
 *  - Anorm is the Infinity-operator-norm of the matrix A
 *  - eps is the machine epsilon returned by DLAMCH('Epsilon').
 *  The values itermax and BWDmax are fixed to 30 and 1.0D+00 respectively.
 *
 *******************************************************************************
 *
 * @param[in] n
 *          The number of linear equations, i.e., the order of the matrix A.
 *          n >= 0.
 *
 * @param[in] kl
 *          The number of subdiagonals within the band of A. kl >= 0.
 *
 * @param[in] ku
 *          The number of superdiagonals within the band of A. ku >= 0.

 * @param[in] nrhs
 *          The number of right hand sides, i.e., the number of columns of the
 *          matrix B. nrhs >= 0.
 *
 * @param[in] pAB
 *          The band matrix AB in LAPACK band matrix format.
 *
 * @param[in] ldab
 *          The leading dimension of the array AB.
 *
 * @param[out] ipiv
 *          The pivot indices; for 1 <= i <= min(m,n), row i of the
 *          matrix was interchanged with row ipiv(i).
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
 * @sa plasma_omp_zcgbsv
 * @sa plasma_dsgbsv
 * @sa plasma_zgbsv
 *
 ******************************************************************************/
int plasma_zcgbsv(int n, int kl, int ku, int nrhs,
                  plasma_complex64_t *pAB, int ldab, int *ipiv,
                  plasma_complex64_t *pB, int ldb,
                  plasma_complex64_t *pX, int ldx, int *iter)
{
    // Get PLASMA context
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    // Check input arguments.
    if (n < 0) {
        plasma_error("illegal value of n");
        return -1;
    }
    if (nrhs < 0) {
        plasma_error("illegal value of nrhs");
        return -2;
    }
    if (ldab < imax(1, 1+kl+ku)) {
        plasma_error("illegal value of lda");
        return -4;
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
        plasma_tune_gbtrf(plasma, PlasmaComplexDouble, n, kl+ku+1);

    // Set tiling parameters.
    int nb = plasma->nb;

    // Initialize barrier
    plasma_barrier_init(&plasma->barrier);

    // Create tile matrices.
    plasma_desc_t AB;
    plasma_desc_t B;
    plasma_desc_t X;
    int tku = (ku+kl+nb-1)/nb; // number of tiles in upper band (not including diagonal)
    int tkl = (kl+nb-1)/nb;    // number of tiles in lower band (not including diagonal)
    int lm = (tku+tkl+1)*nb;   // since we use zgetrf on panel, we pivot back within panel.
                               // this could fill the last tile of the panel,
                               // and we need extra NB space on the bottom
    int retval;
    retval = plasma_desc_general_band_create(PlasmaComplexDouble, PlasmaGeneral,
                                             nb, nb, lm, n, 0, 0, n, n, kl, ku, &AB);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        return retval;
    }
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        n, nrhs, 0, 0, n, nrhs, &B);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        plasma_desc_destroy(&AB);
        return retval;
    }
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        n, nrhs, 0, 0, n, nrhs, &X);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        plasma_desc_destroy(&AB);
        plasma_desc_destroy(&B);
        return retval;
    }

    // Create additional tile matrices.
    plasma_desc_t R, ABs, Xs;
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        B.m, B.n, 0, 0, B.m, B.n, &R);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        plasma_desc_destroy(&AB);
        plasma_desc_destroy(&B);
        plasma_desc_destroy(&X);
        return retval;
    }

    retval = plasma_desc_general_band_create(PlasmaComplexFloat, PlasmaGeneral,
                                             nb, nb, lm, n, 0, 0, n, n, kl, ku, &ABs);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        plasma_desc_destroy(&AB);
        plasma_desc_destroy(&B);
        plasma_desc_destroy(&X);
        plasma_desc_destroy(&R);
        return retval;
    }

    retval = plasma_desc_general_create(PlasmaComplexFloat, nb, nb,
                                        X.m, X.n, 0, 0, X.m, X.n, &Xs);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        plasma_desc_destroy(&AB);
        plasma_desc_destroy(&B);
        plasma_desc_destroy(&X);
        plasma_desc_destroy(&R);
        plasma_desc_destroy(&ABs);
        return retval;
    }



    // Allocate tiled workspace for Infinity norm calculations.
    size_t lwork = imax(((size_t)AB.nt*AB.mt*AB.mb+AB.mb*AB.mt),
                        (size_t)X.mt*X.n+(size_t)R.mt*R.n);
    double *work  = (double*)calloc((lwork),sizeof(double));
    double *Rnorm = (double*)malloc(((size_t)R.n)*sizeof(double));
    double *Xnorm = (double*)malloc(((size_t)X.n)*sizeof(double));

    // Initialize sequence.
    plasma_sequence_t sequence;
    retval = plasma_sequence_init(&sequence);

    // Initialize request.
    plasma_request_t request;
    retval = plasma_request_init(&request);

    // Initialize barrier.
    plasma_barrier_init(&plasma->barrier);

    // asynchronous block
    #pragma omp parallel
    #pragma omp master
    {
        // Translate matrices to tile layout.
        plasma_omp_zpb2desc(pAB, ldab, AB, &sequence, &request);
        plasma_omp_zge2desc(pB, ldb, B, &sequence, &request);

        // Call tile async function.
        plasma_omp_zcgbsv(AB, ipiv, B, X, ABs, Xs, R, work, Rnorm, Xnorm, iter,
                          &sequence, &request);

        // Translate back to LAPACK layout.
        plasma_omp_zdesc2ge(X, pX, ldx, &sequence, &request);
    }
    // implicit synchronization

    // Free matrices in tile layout.
    plasma_desc_destroy(&AB);
    plasma_desc_destroy(&B);
    plasma_desc_destroy(&X);
    plasma_desc_destroy(&R);
    plasma_desc_destroy(&ABs);
    plasma_desc_destroy(&Xs);

    free(work);
    free(Rnorm);
    free(Xnorm);

    // Return status.
    int status = sequence.status;
    return status;
}

/***************************************************************************//**
 * @ingroup plasma_gbsv
 *
 *  Solves a general band linear system of equations using iterative
 *  refinement with the LU factor computed using plasma_cgbtrf.
 *  Non-blocking tile version of plasma_zcgbsv().  Operates on
 *  matrices stored by tiles.  All matrices are passed through
 *  descriptors.  All dimensions are taken from the descriptors.
 *  Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] A
 *          Descriptor of matrix A.
 *
 * @param[out] ipiv
 *          The pivot indices; for 1 <= i <= min(m,n), row i of the
 *          matrix was interchanged with row ipiv(i).
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
 * @sa plasma_zcgbsv
 * @sa plasma_omp_dsgbsv
 * @sa plasma_omp_zgbsv
 *
 ******************************************************************************/
void plasma_omp_zcgbsv(plasma_desc_t A,  int *ipiv,
                       plasma_desc_t B,  plasma_desc_t X,
                       plasma_desc_t As, plasma_desc_t Xs, plasma_desc_t R,
                       double *work, double *Rnorm, double *Xnorm, int *iter,
                       plasma_sequence_t *sequence,
                       plasma_request_t  *request)
{
    const int    itermax = 30;
    const double bwdmax  = 1.0;
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

    // workspaces for dzamax
    double *workX = work;
    double *workR = &work[X.mt*X.n];

    // Compute some constants.
    double cte;
    double eps = LAPACKE_dlamch_work('E');
    double Anorm;
    plasma_pzlangb(PlasmaInfNorm, A, work, &Anorm, sequence, request);

    // Convert B from double to single precision, store result in Xs.
    plasma_pzlag2c(B, Xs, sequence, request);

    // Convert A from double to single precision, store result in As.
    plasma_pzlag2c(A, As, sequence, request);

    // Compute the LU factorization of As.

    plasma_pcgbtrf(As, ipiv, sequence, request);
    // Solve the system As * Xs = Bs.
    plasma_pctbsm(PlasmaLeft, PlasmaLower, PlasmaNoTrans, PlasmaUnit,
                  1.0, As, Xs, ipiv, sequence, request);
    plasma_pctbsm(PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit,
                  1.0, As, Xs, ipiv, sequence, request);
    // Convert Xs to double precision
    plasma_pclag2z(Xs, X, sequence, request);
    // Compute R = B - A * X.

    plasma_pzlacpy(PlasmaGeneral, PlasmaNoTrans, B, R, sequence, request);
    plasma_pzgemm(PlasmaNoTrans, PlasmaNoTrans,
                  zmone, A, X, zone, R, sequence, request);

    // Check whether the nrhs normwise backward error satisfies the
    // stopping criterion. If yes, set iter=0 and return.
    plasma_pdzamax(PlasmaColumnwise, X, workX, Xnorm, sequence, request);
    plasma_pdzamax(PlasmaColumnwise, R, workR, Rnorm, sequence, request);

    #pragma omp taskwait
    {
        cte = Anorm * eps * sqrt((double)A.n) * bwdmax;
        int flag = 1;
        for (int n = 0; n < R.n && flag == 1; n++) {
            if (Rnorm[n] > Xnorm[n] * cte) {
                flag = 0;
            }
        }
        if (flag == 1) {
            *iter = 0;
            return;
        }
    }

    // iterative refinement
    for (int iiter = 0; iiter < itermax; iiter++) {
        // Convert R from double to single precision, store result in Xs.
        plasma_pzlag2c(R, Xs, sequence, request);

        // Solve the system As * Xs = Rs.
        plasma_pctbsm(PlasmaLeft, PlasmaLower, PlasmaNoTrans, PlasmaUnit,
                      1.0, As, Xs, ipiv, sequence, request);

        plasma_pctbsm(PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit,
                      1.0, As, Xs, ipiv, sequence, request);

        // Convert Xs back to double precision and update the current iterate.
        plasma_pclag2z(Xs, R, sequence, request);
        plasma_pzgeadd(PlasmaNoTrans, zone, R, zone, X, sequence, request);

        // Compute R = B - A * X.
        plasma_pzlacpy(PlasmaGeneral, PlasmaNoTrans, B, R, sequence, request);
        plasma_pzgemm(PlasmaNoTrans, PlasmaNoTrans, zmone, A, X, zone, R,
                      sequence, request);

        // Check whether nrhs normwise backward error satisfies the
        // stopping criterion. If yes, set iter = iiter > 0 and return.
        plasma_pdzamax(PlasmaColumnwise, X, workX, Xnorm, sequence, request);
        plasma_pdzamax(PlasmaColumnwise, R, workR, Rnorm, sequence, request);
        #pragma omp taskwait
        {
            int flag = 1;
            for (int n = 0; n < R.n && flag == 1; n++) {
                if (Rnorm[n] > Xnorm[n] * cte) {
                    flag = 0;
                }
            }
            if (flag == 1) {
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


//#if !defined(PLASMA_ZCGESV_WORKAROUND)
    // Compute LU factorization of A.
    //#pragma omp taskwait
    plasma_pzgbtrf(A, ipiv, sequence, request);

    // Solve the system A * X = B.
    plasma_pzlacpy(PlasmaGeneral, PlasmaNoTrans, B, X, sequence, request);

    //#pragma omp taskwait
    //plasma_pzgeswp(PlasmaRowwise, X, ipiv, 1, sequence, request);

    plasma_pztbsm(PlasmaLeft, PlasmaLower, PlasmaNoTrans, PlasmaUnit,
                  1.0, A, X, ipiv, sequence, request);

    plasma_pztbsm(PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit,
                  1.0, A, X, ipiv, sequence, request);
}
