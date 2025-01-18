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

#include "plasma.h"
#include "plasma_internal.h"     /* needed for imin, imax. */
#include "plasma_dlaebz2_work.h" /* work areas. */

#include <string.h>
#include <omp.h>
#include <math.h>
#include "core_lapack.h"

/*******************************************************************************
 *
 * @ingroup plasma_stevx2
 * Symmetric Tridiagonal Eigenvalues/pairs by range.
 *
 * Computes a caller-selected range of eigenvalues and, optionally,
 * eigenvectors of a symmetric tridiagonal matrix A.  Eigenvalues and
 * eigenvectors can be selected by specifying either a range of values or a
 * range of indices for the desired eigenvalues.
 *
 * This is similiar to LAPACK dstevx, with more output parameters.
 *
 * Because input matrices are expected to be extremely large and the exact
 * number of eigenvalues is not necessarily known to the caller, this routine
 * provides a way to get the number of eigenvalues in either a value range or
 * an index range; so the caller can allocate the return arrays. There are
 * three; the floating point vector pVal, the integer vector pMul, and the
 * floating point matrix pVec, which is only required and only referenced for
 * jobtype=PLasmaVec.
 *
 * When the jobtype=PlasmaCount; the code returns the maximum number of
 * eigenvalues in the caller-selected range in pFound (an integer pointer).
 *
 * However, upon return from jobtype=PlasmaVec or jobtype=PlasmaNoVec, the code
 * returns the number of unique eigenvalues found in pFound, which may be less
 * due to ULP-multiplicity.  (ULP is the Unit of Least Precision, the magnitude
 * of the smallest change possible to a given real number). To explain: A real
 * symmetric matrix in NxN should have N distinct real eigenvalues; however, if
 * eigenvalues are closely packed either absolutely (their difference is close
 * to zero) or relatively (their ratio is close to 1.0) then in real arithmetic
 * two such eigenvalues may be within ULP of each other, and thus represented
 * by the same real number. Thus we have ULP-multiplicity, two theoretically
 * distinct eigenvalues represented by the same real number.
 *
 * Finding eigenvalues alone is much faster than finding eigenpairs; the
 * majority of the time consumed when eigenvectors are found is in
 * orthogonalizing the eigenvectors; an O(N*K^2) operation.
 *******************************************************************************
 *
 * @param[in] jobtype
 *          enum:
 *          = PlasmaNoVec: computes eigenvalues only;
 *          = PlasmaVec:   computes eigenvalues and eigenvectors.
 *          = PlasmaCount: computes pFound as the max number of eigenvalues/pairs
 *                         in the given range if there is no ULP-multiplicity, so
 *                         the user can allocate pVal[], pMul[], pVec[].
 *
 * @param[in] range
 *          enum:
 *          PlasmaRangeV use vl, vu for range [vl, vu)
 *          PlasmaRangeI use il, iu for range [il, iu]. 1-relative; 1..N.
 *
 * @param[in] n
 *          int. The order of the matrix A. n >= 0.
 *
 * @param[in] k
 *          int. The space the user has allocated for eigenvalues; as reflected
 *          in pVal, pMul, pVec.
 *
 * @param[in] diag double[n]. Vector of [n] diagonal entries of A.
 *
 * @param[in] offd double[n-1]. A vector of [n-1] off-diagonal entries of A.
 *
 * @param[in] vl   double. Lowest eigenvalue in desired range [vl, vu).  if
 * less than Gerschgorin min; we use Gerschgorin min.
 *
 * @param[in] vu double. Highest eigenvalue in desired range, [vl,vu).  if
 * greater than Gerschgorin max, we use Gerschgorin max+ulp.
 *
 * @param[in] il int. Low Index of range. Must be in range [1,n].
 *
 * @param[in] iu int. High index of range. Must be in range [1,n], >=il.
 *
 * @param[out] pFound int*. On exit, the number of distinct eigenvalues (or
 * pairs) found.  Due to ULP-multiplicity, may be less than the maximum number
 * of eigenvalues in the user's range.  For jobtype=PlasmaCount, the maximum
 * number of distinct eigenvalues in the interval selected by range, [vl,vu) or
 * [il,iu].
 *
 * @param[out] pVal double*. expect double Val[k]. The first 'found' elements
 * are the found eigenvalues.
 *
 * @param[out] pMul int*. expect int Mul[k]. The first 'found' elements are the
 * multiplicity values.
 *
 * @param[out] pVec double*. Expect double Vec[n*k]. the first ('n'*'found')
 * elements contain an orthonormal set of 'found' eigenvectors, each of 'n'
 * elements, in column major format. e.g. eigenvector j is found in Vec[n*j+0]
 * ... Vec[n*j+n-1]. It corresponds to eigenvalue Val[j], with multiplicity
 * Mul[j].  if jobtype=PlasmaNoVec, then pVec is not referenced.
 *
 *******************************************************************************
 *
 * @retval PlasmaSuccess successful exit @retval < 0 if -i, the i-th argument
 * had an illegal value
 *
 ******************************************************************************/

/******************************************************************************
 * STELG: Symmetric Tridiagonal Eigenvalue Least Greatest (Min and Max).
 * Finds the least and largest signed eigenvalues (not least magnitude).
 * begins with bounds by Gerschgorin disc. These may be over or under
 * estimated; Gerschgorin only ensures a disk will contain each. Thus we then
 * use those with bisection to find the actual minimum and maximum eigenvalues.
 * Note we could find the least magnitude eigenvalue by bisection between 0 and
 * each extreme value.
 * By Gerschgorin Circle Theorem;
 * All Eigval(A) are \in [\lamda_{min}, \lambda_{max}].
 * \lambda_{min} = min (i=0; i<n) diag[i]-|offd[i]| - |offd[i-1]|,
 * \lambda_{max} = max (i=0; i<n) diag[i]+|offd[i]| + |offd[i-1]|,
 * with offd[-1], offd[n] = 0.
 * Indexes above are 0 relative.
 * Although Gerschgorin is mentioned in ?larr?.f LAPACK files, it is coded
 * inline there.
 *****************************************************************************/

void plasma_dstelg(double *diag,  double *offd, int n,
        double *Min, double *Max) {
    int i;
    double test, testdi, testdim1, min=__DBL_MAX__, max=-__DBL_MAX__;

    for (i=0; i<n; i++) {
        if (i == 0) testdim1=0.;
        else        testdim1=offd[i-1];

        if (i==(n-1)) testdi=0;
        else          testdi=offd[i];

        test=diag[i] - fabs(testdi) - fabs(testdim1);
        if (test < min) {
            min=test;
        }

        test=diag[i] + fabs(testdi) + fabs(testdim1);
        if (test > max) {
            max=test;
        }
    }


    double cp, minLB=min, minUB=max, maxLB=min, maxUB=max;
    /* Within that range, find the actual minimum. */
    for (;;) {
        cp = (minLB+minUB)*0.5;
        if (cp == minLB || cp == minUB) break;
        if (plasma_dlaneg2(diag, offd, n, cp) == n) minLB = cp;
        else                                      minUB = cp;
    }

    /* Within that range, find the actual maximum. */
    for (;;) {
        cp = (maxLB+maxUB)*0.5;
        if (cp == maxLB || cp == maxUB) break;
        if (plasma_dlaneg2(diag, offd, n, cp) == n) {
            maxUB=cp;
        } else {
            maxLB=cp;
        }
    }

    *Min = minLB;
    *Max = maxUB;
}

/******************************************************************************
 * STMVM: Symmetric Tridiagonal Matrix Vector Multiply.
 * Matrix multiply; A * X = Y.
 * A = [diag[0], offd[0],
 *     [offd[0], diag[1], offd[1]
 *     [      0, offd[1], diag[2], offd[2],
 *     ...
 *     [ 0...0                     offd[n-2], diag[n-1] ]
 * LAPACK does not do just Y=A*X for a packed symmetric tridiagonal matrix.
 * This routine is necessary to determine if eigenvectors should be swapped.
 * This could be done by 3 daxpy, but more code and I think more confusing.
 *****************************************************************************/

void plasma_dstmv(double *diag, double *offd, int n,
        double *X, double *Y) {
    int i;
    Y[0] = diag[0]*X[0] + offd[0]*X[1];
    Y[n-1] = offd[n-2]*X[n-2] + diag[n-1]*X[n-1];

    for (i=1; i<(n-1); i++) {
        Y[i] = offd[i-1]*X[i-1] + diag[i]*X[i] + offd[i]*X[i+1];
    }
}


/******************************************************************************
 * STEPE: Symmetric Tridiagonal EigenPair Error.
 * This routine is necessary to determine if eigenvectors should be swapped.
 * eigenpair error: If A*v = u*v, then A*v-u*v should == 0. We compute the
 * L_infinity norm of (A*v-u*v).
 * We return DBL_MAX if the eigenvector (v) is all zeros, or if we fail to
 * allocate memory.
 * If u==0.0, we'll return L_INF of (A*V).
 *****************************************************************************/

double plasma_dstepe(double *diag,
    double *offd, int n, double u,
    double *v) {
    int i, zeros=0;
    double *AV;
    double norm, dtemp;

    AV = (double*) malloc(n * sizeof(double));
    if (AV == NULL) return __DBL_MAX__;

    plasma_dstmv(diag, offd, n, v, AV); /* AV = A*v. */

    norm = -__DBL_MAX__;  /* Trying to find maximum. */
    zeros=0;
    for (i=0; i<n; i++) {
        dtemp = fabs(AV[i] - u*v[i]);    /* This should be zero. */
        if (dtemp > norm) norm=dtemp;
        if (v[i] == 0.) zeros++;
    }

    free(AV);
    if (zeros == n) return __DBL_MAX__;
    return norm;
}


/******************************************************************************
 * This is the main routine; plasma_dstevx2
 * Arguments are described at the top of this source.
 *****************************************************************************/
int plasma_dstevx2(
  /* error report */
  /* args 1 - 4 */ plasma_enum_t jobtype, plasma_enum_t range, int n, int k,
  /* args 5,6   */ double *diag, double *offd,
  /* args 7,8   */ double vl, double vu,
  /* args 9 - 12*/ int il, int iu, int *pFound, double *pVal,
  /* arg 13,14  */ int    *pMul, double *pVec)
{
    int i, max_threads;
    dlaebz2_Stein_Array_t *stein_arrays = NULL;
    /* Get PLASMA context. */
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    /* Check input arguments */
    if (jobtype != PlasmaVec && jobtype != PlasmaNoVec && jobtype != PlasmaCount) {
        plasma_error("illegal value of jobtype");
        return -1;
    }
    if (range != PlasmaRangeV &&
        range != PlasmaRangeI ) {
        plasma_error("illegal value of range");
        return -2;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -3;
    }

    /* Any value of 'k' is legal on entry, we check it later. */

    if (diag == NULL) {
        plasma_error("illegal pointer diag");
        return -5;
    }
    if (offd == NULL) {
        plasma_error("illegal pointer offd");
        return -6;
    }

    if (range == PlasmaRangeV && vu <= vl ) {
        plasma_error("illegal value of vl and vu");
        return -7;
    }

    if (range == PlasmaRangeI) {
        if (il < 1 || il > imax(1,n)) {
             plasma_error("illegal value of il");
             return -9;
        } else if (iu < imin(n,il) || iu > n) {
            plasma_error("illegal value of iu");
            return -10;
        }
    }

    if (pFound == NULL) return -11;

    /* Quick return */
    if (n == 0) {
        pFound[0]=0;
        return PlasmaSuccess;
    }

    max_threads = omp_get_max_threads();

    if (jobtype == PlasmaVec) {
        /* we use calloc because we rely on pointer elements being NULL to single */
        /* a need to allocate.                                                    */
        stein_arrays = (dlaebz2_Stein_Array_t*) calloc(max_threads, sizeof(dlaebz2_Stein_Array_t));
        if (stein_arrays == NULL) {
            return PlasmaErrorOutOfMemory;
        }
    }

    /* Initialize sequence. */
    plasma_sequence_t sequence;
    plasma_sequence_init(&sequence);

    /* Initialize request. */
    plasma_request_t request;
    plasma_request_init(&request);

    double globMinEval, globMaxEval;

    dlaebz2_Control_t Control;
    memset(&Control, 0, sizeof(dlaebz2_Control_t));
    Control.N = n;
    Control.diag = diag;
    Control.offd = offd;
    Control.jobtype = jobtype;
    Control.range = range;
    Control.il = il;
    Control.iu = iu;
    Control.stein_arrays = stein_arrays;

    /* Find actual least and greatest eigenvalues. */
    plasma_dstelg(Control.diag, Control.offd, Control.N, &globMinEval, &globMaxEval);

    int evLessThanVL=0, evLessThanVU=n, nEigVals=0;
    if (range == PlasmaRangeV) {
        /* We don't call Sturm if we already know the answer. */
        if (vl >= globMinEval) evLessThanVL=plasma_dlaneg2(diag, offd, n, vl);
        else vl = globMinEval; /* optimize for computing step size. */

        if (vu <= globMaxEval) evLessThanVU=plasma_dlaneg2(diag, offd, n, vu);
        else vu = nexttoward(globMaxEval, __DBL_MAX__);  /* optimize for computing step size */
        /* Compute the number of eigenvalues in [vl, vu). */
        nEigVals = (evLessThanVU - evLessThanVL);

         Control.baseIdx = evLessThanVL;
    } else {
        /* PlasmaRangeI: iu, il already vetted by code above. */
        nEigVals = iu+1-il; /* The range is inclusive. */
        /* We still bisect by values to discover eigenvalues, though. */
        vl = globMinEval;
        vu = nexttoward(globMaxEval, __DBL_MAX__); /* be sure to include globMaxVal. */
        Control.baseIdx = 0; /* There are zero eigenvalues less than vl. */
    }

    /* if we just need to find the count of eigenvalues in a value range, */
    if (jobtype == PlasmaCount) {
        pFound[0] = nEigVals;
        return PlasmaSuccess;
    }

    /* Now if user's K (arg 4) isn't enough room, we have a problem. */
    if (k < nEigVals) {
        return -4;             /* problem with user's K value. */
    }

    /* We are going into discovery. Make sure we have arrays. */
    if (pVal == NULL) return -12;   /* pointers cannot be null. */
    if (pMul == NULL) return -13;
    if (jobtype == PlasmaVec && pVec == NULL) return -14;   /* If to be used, cannot be NULL. */

    /* handle value range. */
    /* Set up Control. */
    Control.pVal = pVal;
    Control.pMul = pMul;
    Control.pVec = pVec;

    /* We launch the root task: The full range to subdivide. */
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
                plasma_dlaebz2(&Control, vl, vu, -1, -1, nEigVals);
        }
    }

    /* Now, all the eigenvalues should have unit eigenvectors in the array Control.pVec.
     * We don't need to sort that, but we do want to compress it; in case of multiplicity.
     * We compute the final number of eigenvectors in vectorsFound, and mpcity is recorded.
     */
    int vectorsFound = 0;
    for (i=0; i<nEigVals; i++) {
        if (pMul[i] > 0) {
            vectorsFound++;
        }
    }

    /* record for user. */
    pFound[0] = vectorsFound;

    /* compress the array in case vectorsFound < nEigVals (due to multiplicities).    */
    /* Note that pMul[] is initialized to zeros, if still zero, a multiplicity entry. */
    if (vectorsFound < nEigVals) {
        int j=0;
        for (i=0; i<nEigVals; i++) {
            if (pMul[i] > 0) {                          /* If this is NOT a multiplicity, */
                pMul[j] = pMul[i];                      /* copy to next open slot j       */
                pVal[j] = pVal[i];
                if (Control.jobtype == PlasmaVec) {
                    if (j != i) {
                        memcpy(&pVec[j*Control.N], &pVec[i*Control.N], Control.N*sizeof(double));
                    }
                }

                j++;
            } /* end if we found a non-multiplicity eigenvalue */
        }
    } /* end if compression is needed. */

    /* perform QR factorization, remember the descriptor. */
    plasma_desc_t T;
    int retqrf=0, retgqr=0;

    retqrf = plasma_dgeqrf(Control.N, vectorsFound, /* This leaves pVec in compressed state of Q+R */
        pVec, Control.N, &T);

    if (retqrf != 0) {
        plasma_error("plasma_dgeqrf failed.");
    } else {
        /* extract just the Q of the QR, in normal form, in workspace pQ */
        double* pQ = (double*) malloc(Control.N * vectorsFound * sizeof(double));
        retgqr = plasma_dorgqr(Control.N, vectorsFound, vectorsFound,
                      pVec, Control.N, T, pQ, Control.N);

        if (retgqr != 0) {
            plasma_error("plasma_dorgqr failed.");
        }

        /* copy orthonormal vectors from workspace pQ to pVec for user return. */
        memcpy(pVec, pQ, Control.N*vectorsFound*sizeof(double));
        free(pQ);
        pQ = NULL;
    }

    /* skip swaps if anything failed. */
    if (retqrf || retgqr) goto Cleanup;
    /*************************************************************************
     * When eigenvalue are crowded, it is possible that after orthogonalizing
     * vectors, it can be better to swap neighboring eigenvectors. We just
     * test all the pairs; basically ||(A*V-e*V)||_max is the error.  if BOTH
     * vectors in a pair have less error by being swapped, we swap them.
     ************************************************************************/
    int swaps=0;
    if (jobtype == PlasmaVec) {
        int N = Control.N;
        double *Y = malloc(N * sizeof(double));
        double test[4];

        for (i=0; i<vectorsFound-1; i++) {
            if (fabs(pVal[i+1]-pVal[i]) > 1.E-11) continue;

            /* We've tried to parallelize the following four tests
             * as four omp tasks. It works, but takes an average of
             * 8% longer (~3.6 ms) than just serial execution.
             * omp schedule and taskwait overhead, I presume.
             */

            test[0]= plasma_dstepe(Control.diag, Control.offd, N,
                    pVal[i], &pVec[i*N]);
            test[1] = plasma_dstepe(Control.diag, Control.offd, N,
                    pVal[i+1], &pVec[(i+1)*N]);

            test[2] = plasma_dstepe(Control.diag, Control.offd, N,
                    pVal[i], &pVec[(i+1)*N]);
            test[3] = plasma_dstepe(Control.diag, Control.offd, N,
                    pVal[i+1], &pVec[i*N]);

            if ( (test[2] < test[0])         /* val1 with vec2 beats val1 with vec1 */
              && (test[3] < test[1]) ) {     /* val2 with vec1 beats val2 with vec2 */
                memcpy(Y, &pVec[i*N], N*sizeof(double));
                memcpy(&pVec[i*N], &pVec[(i+1)*N], N*sizeof(double));
                memcpy(&pVec[(i+1)*N], Y, N*sizeof(double));
                swaps++;
            }
        } /* end swapping. */

        free(Y);
    } /* end if we found eigenvectors. */

    /* Free all the blocks that got used. */
Cleanup:
    for (i=0; i<max_threads; i++) {
       if (stein_arrays[i].IBLOCK) free(stein_arrays[i].IBLOCK);
       if (stein_arrays[i].ISPLIT) free(stein_arrays[i].ISPLIT);
       if (stein_arrays[i].WORK  ) free(stein_arrays[i].WORK  );
       if (stein_arrays[i].IWORK ) free(stein_arrays[i].IWORK );
       if (stein_arrays[i].IFAIL ) free(stein_arrays[i].IFAIL );
    }

    if (stein_arrays) free(stein_arrays);
    if (retqrf || retgqr) /* if we failed orthogonalization */
        plasma_request_fail(&sequence, &request, PlasmaErrorIllegalValue);

    /* Return status. */
    return sequence.status;
}
