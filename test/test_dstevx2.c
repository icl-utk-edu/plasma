/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal d -> s
 *
 **/
#include "test.h"
#include "flops.h"
#include "plasma.h"
#include "core_lapack.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <omp.h>

#define REAL

/******************************************************************************
 * Matrix detailed in Kahan; et al.
 * Matrix Test: diag=[+x,-x,+x,-x,...+x,-x] for any real x, but Kahan chooses
 *                                          a tiny x.
 *              offd=[1,1,...1]
 * Dimension: n.
 * Computed eigenvalues:
 * evalue[k] = [ x*x + 4*cos(k/(n+1))^2 ] ^(1/2),
 * evalue[n+1-k] = -evalue[k], for k=1,[n/2],
 * evalue[(n+1)/2] = 0 if n is odd.
 * Note k is 1-relative in these formulations.
 * The eigenvalues range from (-2,+2).
 * Note: This routine verified to match documentation for n=4,8,12,24.
 *****************************************************************************/

static void testMatrix_Kahan(double* diag, double *offd,
            double* evalue, lapack_int n, double myDiag) {
   lapack_int i,k;
   for (k=1; k<=(n/2); k++) {
      double ev;
      ev = (M_PI*k+0.)/(n+1.0); /* angle in radians.                       */
      ev = cos(ev);             /* cos(angle)                              */
      ev *= 4.*ev;              /* 4*cos^2(angle)                          */
      ev += myDiag*myDiag;      /* x^2 + 4*cos^2(angle)                    */
      ev = sqrt(ev);            /* (x^2 + 4*cos^2(angle))^(0.5)            */
      /* we reverse the -ev and ev here, to get in ascending sorted order. */
      evalue[k-1] = -ev;
      evalue[n+1-k-1] = ev;
   }

   for (i=0; i<n-1; i++) {
      k=(i&1);
      if (k) diag[i]=-myDiag;
      else   diag[i]=myDiag;
      offd[i] = 1.0;
   }

      k=(i&1);
      if (k) diag[i]=-myDiag;
      else   diag[i]=myDiag;
}


/******************************************************************************
 * This tests an eigenvector X for the eigenvalue lambda.
 * We should have A*X = lambda*X. Thus, (A*X)/lambda = X.
 * We perform the matrix multiply for each element X[i], and divide the result
 * by lambda, yielding mmRes[i] which should equal X[i]. We sum the squares of
 * these results, and the squares of X[i], to compute the Frobenious Norm. We
 * return the absolute difference of these norms as the error in the vector.
 *
 * Matrix multiply; A * X = Y.
 * A = [diag[0], offd[0],
 *     [offd[0], diag[1], offd[1]
 *     [      0, offd[1], diag[2], offd[2],
 *     ...
 *     [ 0...0                     offd[n-2], diag[n-1] ]
 *****************************************************************************/

static double testEVec(double *diag, double *offd,
              int n, double *X, double lambda) {
    int i;
    double mmRes, vmRes, error, sumMM=0., sumVec=0., invLambda = 1.0/lambda;

    mmRes = (diag[0]*X[0] + offd[0]*X[1])*invLambda;
    vmRes = X[0];
    sumMM += mmRes*mmRes;
    sumVec += vmRes*vmRes;

    mmRes = (offd[n-2]*X[n-2] + diag[n-1]*X[n-1])*invLambda;
    vmRes = X[n-1];
    sumMM += mmRes*mmRes;
    sumVec += vmRes*vmRes;

    for (i=1; i<(n-1); i++) {
        mmRes = (offd[i-1]*X[i-1] + diag[i]*X[i] + offd[i]*X[i+1])*invLambda;
        vmRes = X[i];
        sumMM += mmRes*mmRes;
        sumVec += vmRes*vmRes;
    }

    sumMM = sqrt(sumMM);
    sumVec = sqrt(sumVec);

    return(fabs(sumVec-sumMM));
}


/***************************************************************************//**
 * @brief Tests DSTEVX2.
 *
 * @param[in,out] param - array of parameters
 * @param[in]     run - whether to run test
 *
 * Sets used flags in param indicating parameters that are used.
 * If run is true, also runs test and stores output parameters.
 ******************************************************************************/
void test_dstevx2(param_value_t param[], bool run)
{
    int i,j;
    /*****************************************************************
     * Mark which parameters are used.
     ****************************************************************/
    param[PARAM_DIM    ].used = PARAM_USE_M;
    if (! run)
        return;

    /*****************************************************************
     * Set parameters.
     ****************************************************************/
    int m = param[PARAM_DIM].dim.m;
    int test = param[PARAM_TEST].c == 'y';
    double eps = LAPACKE_dlamch('E');

    /*****************************************************************
     * Set tuning parameters.
     ****************************************************************/
    plasma_set(PlasmaTuning, PlasmaDisabled);

    /*****************************************************************
     * Allocate and initialize arrays.
     ****************************************************************/
    double *Diag =
        (double*)malloc((size_t)m*sizeof(double));
    assert(Diag != NULL);

    double *Offd =
        (double*)malloc((size_t)(m-1)*sizeof(double));
    assert(Offd != NULL);

    double *eigenvalues =
        (double*)malloc((size_t)m*sizeof(double));
    assert(eigenvalues != NULL);

    double *pVal =
        (double*)malloc((size_t)m*sizeof(double));
    assert(pVal != NULL);

    int *pMul = (int*)malloc((size_t)m*sizeof(int));
    assert(pMul != NULL);

    /**************************************************************************
     * Kahan has eigenvalues from [-2.0 to +2.0]. However, eigenvalues are
     * dense near -2.0 and +2.0, so for large matrices, the density may cause
     * eigenvalues separated by less than machine precision, which causes us
     * multiplicity (eigenvalues are identical at machine precision). We first
     * see this in single precision at m=14734, with a multiplicity of 2.
     *************************************************************************/

    double myDiag=1.e-5;
    testMatrix_Kahan(Diag, Offd, eigenvalues, m, myDiag);
    double minAbsEV=__DBL_MAX__, maxAbsEV=0., Kond;
    for (i=0; i<m; i++) {
        if (fabs(eigenvalues[i]) < minAbsEV) minAbsEV=fabs(eigenvalues[i]);
        if (fabs(eigenvalues[i]) > maxAbsEV) maxAbsEV=fabs(eigenvalues[i]);
    }
    Kond = maxAbsEV / minAbsEV;

    lapack_int nEigVals=0, vectorsFound=0;
    lapack_int il=0, iu=500;
    double vl=1.5, vu=2.01;
    double *pVec = NULL;

    /**************************************************************************
     * Get the number of eigenvalues in a value range. Note these can include
     * multiplicity; the number of unique eigenvectors will be discovered by
     * plasma_dstevx2.
     *************************************************************************/

    lapack_int ret;
    ret=plasma_dstevx2(
            PlasmaCount,    /* Type of call (1)         */
            PlasmaRangeV,   /* Range type (2)           */
            m, 0,           /* N, k (3,4)               */
            Diag, Offd,     /* diag, offd (5,6)         */
            vl, vu,         /* vl, vu (7,8)             */
            il, iu,         /* il, iu (9,10)            */
            &nEigVals,      /* pFound, (11)             */
            pVal,           /* p eigenvals array. (12)  */
            pMul,           /* p eigenMult array  (13)  */
            pVec);          /* p eigenVec  array  (14)  */

    if (nEigVals < 1) {
        plasma_error("plasma_dstevx2() found no eigenvalues for test matrix.");
        param[PARAM_TIME].d    = 0.0;
        param[PARAM_GFLOPS].d  = 0.0;
        param[PARAM_ERROR].d   = 1.0;
        param[PARAM_SUCCESS].i = false;
        return;
    }

    /**************************************************************************
     * We allocate pVec late, we cannot afford to allocate m*m entries
     * (to cover every possibility) when m is huge.
     *************************************************************************/

    pVec = (double*)malloc((size_t)m*nEigVals*sizeof(double));
    assert(pVec != NULL);

    /* Run and time plasma_dstevx2, range based on values. */
    plasma_time_t start = omp_get_wtime();

    ret=plasma_dstevx2(
            PlasmaVec,     /* Type of call (1)          */
            PlasmaRangeV,  /* Range type (2)            */
            m, nEigVals,   /* N, k (3,4)                */
            Diag, Offd,    /* diag, offd (5,6)          */
            vl, vu,        /* vl, vu (7,8)              */
            il, iu,        /* il, iu (9,10)             */
            &vectorsFound, /* pFound, (11)              */
            pVal,          /* p eigenvals array. (12)   */
            pMul,          /* p eigenMult array  (13)   */
            pVec);         /* p eigenVec  array  (14)   */

    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    if (ret != 0) {
        char errstr[128];
        sprintf(errstr, "plasma_dstevx2() failed returned %i", ret);
        plasma_error(errstr);
        param[PARAM_TIME].d    = 0.0;
        param[PARAM_GFLOPS].d  = 0.0;
        param[PARAM_ERROR].d   = 1.0;
        param[PARAM_SUCCESS].i = false;
        return;
    }

    param[PARAM_TIME].d = time;

    /*****************************************************************
     * Test results directly. Check eigenvalues discovered by vl, vu.
     ****************************************************************/

    if (test) {
        /**********************************************************************
         * Worth reporting for debug: m (matrix rows) vectorsFound (columns).
         * eigenvalues in pVal[0..vectorsFound-1], multiplicity in pMul[].
         *********************************************************************/

        /**********************************************************************
         * Find worst eigenvalue error. However, we must worry about
         * multiplicity. In single precision this first occurs at m=14734, with
         * vl=1.5, vu=2.01; mpcity=2. At m=75000, vl=1.5, vu=2.01, mpcity=10.
         * We must also worry about the magnitude of eigenvalues; machine
         * epsilon for large eigenvalues is much greater than for small ones.
         *********************************************************************/

        double worstEigenvalue_error = 0, worstEigenvalue_eps;
        lapack_int worstEigenvalue_index = 0, worstEigenvalue_mpcty = 0, max_mpcty = 0;
        double worstEigenvector_error = 0;
        lapack_int worstEigenvector_index = 0;
        i=0;
        lapack_int evIdx=m-nEigVals;
        while (evIdx < m) {
            if (pMul[i] > max_mpcty) max_mpcty = pMul[i];

            for (j=0; j<pMul[i]; j++) {
                double ev_eps = nexttoward(fabs(eigenvalues[evIdx]), __DBL_MAX__) - fabs(eigenvalues[evIdx]);
                double error = fabs(pVal[i]-eigenvalues[evIdx]) / ev_eps;
                if (error > worstEigenvalue_error) {
                    worstEigenvalue_index = i;
                    worstEigenvalue_error = error;
                    worstEigenvalue_eps = ev_eps;
                    worstEigenvalue_mpcty = pMul[i];
                }

                evIdx++; /* advance known eigenvalue index for a multiplicity. */
                if (evIdx == m) break;
            }

            i++; /* advance to next discovered eigenvalue. */
        }

        /**********************************************************************
         * Worth reporting for debug: worstEigenvalue_index,
         * worstEigenvalue_error, max_mpcty.
         *********************************************************************/

        param[PARAM_ERROR].d = worstEigenvalue_error*worstEigenvalue_eps;
        param[PARAM_SUCCESS].i = (worstEigenvalue_error < 3.);

        if (!param[PARAM_SUCCESS].i) goto TestingDone; /* exit if not successful. */

        /**********************************************************************
         * If we have no eigenvalue errors, We need to test the eigenvectors in
         * pVec; testEVec returns fabs(||(A*pVec)/pVal||_2 - ||pVec||_2) for
         * each eigenvalue and eigenvector, we track the largest value.
         * Empirically; the error grows slowly with m. We divide by epsilon,
         * and 2*ceil(log_2(m)) epsilons seems a reasonable threshold without
         * being too liberal. Obviously this is related to the number of bits
         * of error in the result. The condition number (Kond) of the Kahan
         * matrix also grows nearly linearly with m; Kond is computed above.
         *********************************************************************/

        for (i=0; i<vectorsFound; i++) {
            double vErr;
            vErr=testEVec(Diag, Offd, m, &pVec[m*i], pVal[i]);

            if (vErr > worstEigenvector_error) {
                worstEigenvector_error = vErr;
                worstEigenvector_index = i;
            }
        }

        /* Find ceiling(log_2(m)); double it as allowable eps of err */
        i=1;
        while ((m>>i)) i++;
        param[PARAM_ERROR].d = (worstEigenvector_error);
        param[PARAM_SUCCESS].i = (worstEigenvector_error <= (i<<1)*eps);
    } /* end if (test) */

    /*****************************************************************
     * Free arrays.
     ****************************************************************/
TestingDone:
    if (Diag != NULL) free(Diag);
    if (Offd != NULL) free(Offd);
    if (eigenvalues != NULL) free(eigenvalues);
    if (pVal != NULL) free(pVal);
    if (pMul != NULL) free(pMul);
    if (pVec != NULL) free(pVec);

    if (test) {
        /* free any test specific matrices; currently none. */
    }
}
