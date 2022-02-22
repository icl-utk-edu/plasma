/**
 *
 * @file 
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> s d 
 *
 **/

/*
 * This file is a z-template to generate s and d code.
 * Only s and d are compiled; not c or z. 
 */
 
#include "plasma.h"
#include "plasma_internal.h"     /* needed for imin, imax. */
#include "plasma_zlaebz2_work.h" /* work areas. */

#include <string.h>
#include <omp.h>
#include <math.h>
#include <float.h>
#include "mkl_lapack.h"
/* core_lapack.h gives us plasma_zgeqrf; which is 10x slower than MKL's zgeqrf. */
/* #include "core_lapack.h" */
/* #include "lapack.h" */

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
 * returns the number of unique eigenvalues found in pFound. For a symmetric
 * matrix the maximum number of eigenvaues and the unique eigenvalues found
 * should be equal; but due to the limits of machine precision, multiple
 * arithmetically unique eigenvalue may be approximated by the same floating
 * point number. In that case, to the machine this looks like a multiplicity;
 * and we report it that way. We see this phenomenon in large (e.g. N=50000)
 * test matrices, for example. Our tester uses Kahan matrices, because the
 * eigenvalues are directly computable, between -2.0 and +2.0; and at N=50000 
 * vl=1.5, vu=2.0, we do get a multiplicities up to 6.
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
 *                         in the given range, so user can allocate 
 *                         pVal[Found], pMult[Found], pVec[n x Found].
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
 *          int. The space the user has allocated for eigenvalues; as reflected in
 *          pVal, pMul, pVec. 
 *
 * @param[in] diag
 *          double[n]. Vector of [n] diagonal entries of A. 
 *
 * @param[in] offd
 *          double[n-1]. A vector of [n-1] off-diagonal entries of A.
 *
 * @param[in] vl   
 *          double. Lowest eigenvalue in desired range [vl, vu).
 *          if less than Gerschgorin min; we use Gerschgorin min.
 *
 * @param[in] vu
 *          double. Highest eigenvalue in desired range, [vl,vu).
 *          if greater than Gerschgorin max, we use Gerschgorin max+eps.
 *
 * @param[in] il
 *          int. Low Index of range. Must be in range [1,n].
 *
 * @param[in] iu
 *          int. High index of range. Must be in range [1,n], >=il.
 *
 * @param[out] pFound
 *          int*. On exit, the number of distinct eigenvalues (or pairs) found.
 *          Due to machine-precision multiplicity, may be less than the maximum
 *          number of eigenvalues in the user's range.
 *          For jobtype=PlasmaCount, the maximum number of distinct
 *          eigenvalues in the interval selected by range, [vl,vu) or [il,iu].
 *
 * @param[out] pVal
 *          double*. expect double Val[k]. The first 'found' elements are the 
 *          found eigenvalues.
 *
 * @param[out] pMul
 *          int*. expect int Mul[k]. The first 'found' elements are the 
 *          multiplicity values.
 *
 * @param[out] pVec
 *          double*. Expect double Vec[n*k]. the first ('n'*'found') elements 
 *          contain an orthonormal set of 'found' eigenvectors, each of 'n'
 *          elements, in column major format. e.g. eigenvector j is found in
 *          Vec[n*j+0] ... Vec[n*j+n-1]. It corresponds to eigenvalue Val[j],
 *          with multiplicity Mul[j].
 *          if jobtype=PlasmaNoVec, then pVec is not referenced.
 *
 *******************************************************************************
 *
 * @retval PlasmaSuccess successful exit
 * @retval < 0 if -i, the i-th argument had an illegal value
 *
 ******************************************************************************/

/******************************************************************************
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

static void Bound_MinMax_Eigvalue(plasma_complex64_t *diag, 
            plasma_complex64_t *offd, int n, plasma_complex64_t *Min, 
            plasma_complex64_t *Max) {
    int i;
    plasma_complex64_t test, testdi, testdim1, min=DBL_MAX, max=-DBL_MAX;
 
    for (i=0; i<n; i++) {
        if (i == 0) testdim1=0.;
        else        testdim1=offd[i-1];
        
        if (i==(n-1)) testdi=0;
        else          testdi=offd[i];
        
        test=diag[i] - fabs(testdi) - fabs(testdim1);
        if (test < min) {
            min=test;
            if(0) fprintf(stderr,"Gerschgorin row=%i new min=%.16e.\n", i, min);
        } 
        
        test=diag[i] + fabs(testdi) + fabs(testdim1);
        if (test > max) {
            max=test;
            if(0) fprintf(stderr,"Gerschgorin row=%i new max=%.16e.\n", i, max);
        }      
    }
       
 
    plasma_complex64_t cp, minLB=min, minUB=max, maxLB=min, maxUB=max;
    /* Now, within that range, find the actual minimum. */
    while (1) {
        cp = (minLB+minUB)*0.5;
        if (cp == minLB || cp == minUB) break;
        if (plasma_zlaneg2(diag, offd, n, cp) == n) minLB = cp;
        else                                      minUB = cp;
    }
     
    /* Now find the max within that range. At each midpoint MidP: */
    while (1) {
        cp = (maxLB+maxUB)*0.5;
        if (cp == maxLB || cp == maxUB) break;
        if (plasma_zlaneg2(diag, offd, n, cp) == n) {
            if(0) fprintf(stderr,"maxLB=%.16e maxUB=%.16e new maxUB=cp=%.16e.\n", maxLB, maxUB, cp);
            maxUB=cp;
        } else {
            if(0) fprintf(stderr,"maxLB=%.16e maxUB=%.16e new maxLB=cp=%.16e.\n", maxLB, maxUB, cp);
            maxLB=cp;
        }
    }
 
    *Min = minLB;
    *Max = maxUB;
} /* end Bound_MinMax_Eigvalue */

/******************************************************************************
 * Matrix multiply; A * X = Y.
 * A = [diag[0], offd[0], 
 *     [offd[0], diag[1], offd[1]
 *     [      0, offd[1], diag[2], offd[2],
 *     ...
 *     [ 0...0                     offd[n-2], diag[n-1] ]
 * LAPACK does not do just Y=A*X for a packed symmetric tridiagonal matrix.
 * This routine is necessary to determine if eigenvectors should be swapped.
 *****************************************************************************/

static void MM(plasma_complex64_t *diag, plasma_complex64_t *offd, int n, 
            plasma_complex64_t *X, plasma_complex64_t *Y) {
    int i;
    Y[0] = diag[0]*X[0] + offd[0]*X[1];
    Y[n-1] = offd[n-2]*X[n-2] + diag[n-1]*X[n-1];
 
    for (i=1; i<(n-1); i++) {
        Y[i] = offd[i-1]*X[i-1] + diag[i]*X[i] + offd[i]*X[i+1];
    }
} /* END MM. */


/******************************************************************************
 * This routine is necessary to determine if eigenvectors should be swapped.
 * eigenpair error: If A*v = u*v, then A*v-u*v should == 0. We compute the
 * L_infinity norm of (A*v-u*v).
 * We return DBL_MAX if the eigenvector (v) is all zeros, or if we fail to 
 * allocate memory. 
 * If u==0.0, we'll return L_INF of (A*V). 
 *****************************************************************************/

static plasma_complex64_t eigp_error(plasma_complex64_t *diag, 
       plasma_complex64_t *offd, int n, plasma_complex64_t u, 
       plasma_complex64_t *v) {
    int i, zeros=0;
    plasma_complex64_t *AV;
    plasma_complex64_t norm, dtemp;
 
    AV = (plasma_complex64_t*) calloc(n, sizeof(plasma_complex64_t));
    if (AV == NULL) return(DBL_MAX);
     
    MM(diag, offd, n, v, AV); /* AV = A*v. */
 
    norm = -DBL_MAX;  /* Trying to find maximum. */
    zeros=0;
    for (i=0; i<n; i++) {
        dtemp = fabs(AV[i] - u*v[i]);    /* This should be zero. */
        if (dtemp > norm) norm=dtemp;
        if (v[i] == 0.) zeros++;
    }
 
    free(AV);
    if (zeros == n) return(DBL_MAX);
    return(norm);
} /* end eigp_error. */


/******************************************************************************
 * This is the main routine; plasma_zstevx2 
 *****************************************************************************/
int plasma_zstevx2(
  /* args 1,2  */ plasma_enum_t jobtype, plasma_enum_t range,
  /* args 3,4  */ int n, int k,
  /* arg 5     */ plasma_complex64_t *diag,
  /* arg 6     */ plasma_complex64_t *offd,
  /* arg 7     */ plasma_complex64_t vl,
  /* arg 8     */ plasma_complex64_t vu,
  /* args 9,10,*/ int il, int iu,
  /* arg 11    */ int *pFound,
  /* arg 12    */ plasma_complex64_t *pVal,
  /* arg 13    */ int    *pMul,
  /* arg 14    */ plasma_complex64_t *pVec)
{
    int i, max_threads;
    zlaebz2_Stein_Array_t *stein_arrays = NULL;
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

    /* arg 4: Any value of 'k' is legal on entry, we check it later. */

    if (diag == NULL) {
        plasma_error("illegal pointer diag");
        return -5;
    }
    if (offd == NULL) {
        plasma_error("illegal pointer offd");
        return -6;
    }
    
    /* Check args 7, 8. */
    if (range == PlasmaRangeV && vu <= vl ) {
        plasma_error("illegal value of vl and vu");
        return -7;
    }

    /* check args 9, 10. */
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
        stein_arrays = (zlaebz2_Stein_Array_t*) calloc(max_threads, sizeof(zlaebz2_Stein_Array_t));
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

    plasma_complex64_t globMinEval, globMaxEval; 

    zlaebz2_WorkStack_t workStack;
    memset(&workStack, 0, sizeof(zlaebz2_WorkStack_t)); 
    workStack.N = n;
    workStack.diag = diag;
    workStack.offd = offd;
    workStack.jobtype = jobtype;
    workStack.range = range;
    workStack.il = il;
    workStack.iu = iu;
    workStack.stein_arrays = stein_arrays;

    /* Find actual min and max eigenvalues. */
    Bound_MinMax_Eigvalue(workStack.diag, workStack.offd, workStack.N, &globMinEval, &globMaxEval);
    if (0) fprintf(stderr, "%s:%i globMinEval=%.15f, globMaxEval=%.15f, vl=%.15f vu=%.15f\n",
           __func__, __LINE__, globMinEval, globMaxEval, vl, vu);

    int evLessThanVL=0, evLessThanVU=n, nEigVals=0;
    if (range == PlasmaRangeV) {
        /* We don't call Sturm if we already know the answer. */
        if (vl >= globMinEval) evLessThanVL=plasma_zlaneg2(diag, offd, n, vl);
        else vl = globMinEval; /* optimize for computing step size. */

        if (vu <= globMaxEval) evLessThanVU=plasma_zlaneg2(diag, offd, n, vu);
        else vu = nexttoward(globMaxEval, DBL_MAX);  /* optimize for computing step size */
        /* Compute the number of eigenvalues in [vl, vu). */
        nEigVals = (evLessThanVU - evLessThanVL);
        if (0) fprintf(stderr, "%s:%i evLessThanVU=%i, evLessThanVL=%i, nEigVals=%i.\n", 
               __func__, __LINE__, evLessThanVU, evLessThanVL, nEigVals);

         workStack.baseIdx = evLessThanVL;
    } else {
        /* PlasmaRangeI: iu, il already vetted by code above. */
        nEigVals = iu+1-il; /* The range is inclusive. */
        /* We still bisect by values to discover eigenvalues, though. */
        vl = globMinEval;
        vu = nexttoward(globMaxEval, DBL_MAX); /* be sure to include globMaxVal. */
        workStack.baseIdx = 0; /* There are zero eigenvalues less than vl. */
    }

    /* if we just need to find the count of eigenvalues in a value range, */
    if (jobtype == PlasmaCount) {
        pFound[0] = nEigVals;
        return(PlasmaSuccess);
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
    /* Set up workStack controls. */
    workStack.eigenvalues = nEigVals;
    workStack.finished = 0;
    workStack.pVal = pVal;
    workStack.pMul = pMul;
    workStack.pVec = pVec;
    
    /* Create a bracket per processor to kick off the work stack.
     * here, low and high are the range values.
     * We need low+max_threads*step = hi-step. ==>
     * (hi-low) = step*(max_threads+1) ==>
     * step = (hi-low)/(max_threads+1).
     */

    plasma_complex64_t step = (vu - vl)/(max_threads+1);
    plasma_complex64_t prevUpper = vl;
    for (i=0; i<max_threads; i++) {
        zlaebz2_EV_Bracket_t *thisBracket = (zlaebz2_EV_Bracket_t*) calloc(1, sizeof(zlaebz2_EV_Bracket_t));
        thisBracket->stage = PlasmaStageInit;
        thisBracket->lowerBound = prevUpper;
        if (i == max_threads-1) thisBracket->upperBound = vu;
        else                    thisBracket->upperBound = thisBracket->lowerBound + step;
        prevUpper = thisBracket->upperBound; /* don't rely on arithmetic for final value. */
        thisBracket->nLT_low = -1;
        thisBracket->nLT_hi  = -1;
        thisBracket->numEV   = -1;
    
        /* Now add to the workstack. OMP not active yet. */          
        thisBracket->next = workStack.ToDo;
        workStack.ToDo = thisBracket;
    }

    /* We can launch the threads. */
    #pragma omp parallel /* proc_bind(close) requires gcc >=4.9. */
    {
       plasma_zlaebz2(&workStack);
    }
 
    /* Now, all the eigenvalues should have unit eigenvectors in the array workStack.Done.
     * We don't need to sort that, but we do want to compress it; in case of multiplicity.
     * We compute the final number of eigenvectors in vectorsFound, and mpcity is recorded.
     */
    int vectorsFound = 0;
    for (i=0; i<workStack.eigenvalues; i++) {
        if (pMul[i] > 0) {
            vectorsFound++;
        }
    }

    /* record for user. */
    pFound[0] = vectorsFound;

    /* compress the array in case vectorsFound < nEigVals (due to multiplicities).    */
    /* Note that pMul[] is initialized to zeros, if still zero, a multiplicity entry. */
    if (vectorsFound < workStack.eigenvalues) {
        int j=0;   
        for (i=0; i<workStack.eigenvalues; i++) {
            if (pMul[i] > 0) {                          /* If this is NOT a multiplicity, */
                pMul[j] = pMul[i];                      /* copy to next open slot j       */
                pVal[j] = pVal[i];      
                if (workStack.jobtype == PlasmaVec) {
                    if (j != i) {
                        memcpy(&pVec[j*workStack.N], &pVec[i*workStack.N], workStack.N*sizeof(plasma_complex64_t));
                    }
                }

                j++;
            } /* end if we found a non-multiplicity eigenvalue */
        }
    } /* end if compression is needed. */

    double orth_s;
    double start_orth;
    int ret;

    /* perform QR factorization, remember the descriptor. */
    plasma_desc_t T;

    if(1) start_orth = omp_get_wtime();
    ret = plasma_zgeqrf(workStack.N, vectorsFound, /* This leaves pVec in compressed state of Q+R */
                  pVec, workStack.N, &T);
    if (ret != 0) {
        fprintf(stderr, "%s:%i ret=%i for plasma_zqeqrf.\n", __func__, __LINE__, ret);
        exit(-1);
    }

    /* extract just the Q of the QR, in normal form, in workspace pQ */
    plasma_complex64_t* pQ = (plasma_complex64_t*) calloc(workStack.N*vectorsFound, sizeof(plasma_complex64_t));
    ret = plasma_zungqr(workStack.N, vectorsFound, vectorsFound,
                  pVec, workStack.N, T, pQ, workStack.N);

    if (ret != 0) {
        fprintf(stderr, "%s:%i ret=%i for plasma_zungqr.\n", __func__, __LINE__, ret);
        exit(-1);
    }

    /* copy orthonormal vectors from workspace pQ to pVec for user return. */
    memcpy(pVec, pQ, workStack.N*vectorsFound*sizeof(plasma_complex64_t));
    {free(pQ); pQ = NULL;}

    if(1) {
        orth_s = (omp_get_wtime() - start_orth);
        fprintf(stderr, "%s:%i plasma_qrf=%.6f sec\n", __func__, __LINE__, orth_s);
    }

    /*************************************************************************
     * When eigenvalue are crowded, it is possible that after orthogonalizing
     * vectors, it can be better to swap neighboring eigenvectors. We just 
     * test all the pairs; basically ||(A*V-e*V)||_max is the error.  if BOTH 
     * vectors in a pair have less error by being swapped, we swap them.
     ************************************************************************/
    int swaps=0;
    if (jobtype == PlasmaVec) {
        int N = workStack.N; 
        plasma_complex64_t *Y = calloc(N, sizeof(plasma_complex64_t));
        plasma_complex64_t *test = calloc(4, sizeof(plasma_complex64_t));
        for (i=0; i<vectorsFound-1; i++) {
            if (fabs(pVal[i+1]-pVal[i]) > 1.E-11) continue;

            /* We've tried to parallelize the following four tests
             * as four omp tasks. It works, but takes an average of
             * 8% longer (~3.6 ms) than just serial execution. 
             * omp schedule and taskwait overhead, I presume.
             */

            test[0]= eigp_error(workStack.diag, workStack.offd, N,
                    pVal[i], &pVec[i*N]);
            test[1] = eigp_error(workStack.diag, workStack.offd, N,
                    pVal[i+1], &pVec[(i+1)*N]);
            
            test[2] = eigp_error(workStack.diag, workStack.offd, N,
                    pVal[i], &pVec[(i+1)*N]);
            test[3] = eigp_error(workStack.diag, workStack.offd, N,
                    pVal[i+1], &pVec[i*N]);
            
            if ( (test[2] < test[0])         /* val1 with vec2 beats val1 with vec1 */
              && (test[3] < test[1]) ) {     /* val2 with vec1 beats val2 with vec2 */
                if(0) fprintf(stderr, "%s:%i Swapping vectors for %d and %d; eigenvalue diff=%.16e.\n", __func__, __LINE__, i, i+1, pVal[i+1]-pVal[i] );
                memcpy(Y, &pVec[i*N], N*sizeof(plasma_complex64_t));
                memcpy(&pVec[i*N], &pVec[(i+1)*N], N*sizeof(plasma_complex64_t));
                memcpy(&pVec[(i+1)*N], Y, N*sizeof(plasma_complex64_t));
                swaps++;
            }
        } /* end swapping. */

        if (test) free(test);
        if (Y) free(Y);
    } /* end if we want to swap at all. */

    /* Free all the blocks that got used. */
    for (i=0; i<max_threads; i++) {
       if (stein_arrays[i].IBLOCK) free(stein_arrays[i].IBLOCK);
       if (stein_arrays[i].ISPLIT) free(stein_arrays[i].ISPLIT);
       if (stein_arrays[i].WORK  ) free(stein_arrays[i].WORK  );
       if (stein_arrays[i].IWORK ) free(stein_arrays[i].IWORK );
       if (stein_arrays[i].IFAIL ) free(stein_arrays[i].IFAIL );
    }

    if (stein_arrays) free(stein_arrays);

    /* Return status. */
    if(0) fprintf(stderr, "plasma_stevx2 exit with return=%d.\n", sequence.status);
    return sequence.status;
} /* END plasma_zstevx2 */

