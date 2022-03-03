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

/******************************************************************************
 * This code is not designed to be called directly by users; it is a subroutine
 * for zstevx2.c. 
 *
 * Specifically, this is launched in parallel mode, and all the parameters are
 * contained in the already initialized and populated zlaebz2_WorkStack_t; For 
 * example, from zstevx2:
 *
 *  #pragma omp parallel proc_bind(close)
 *  {
 *     thread_work(&workStack);
 *  }
 *
 * We are given the WorkStack; we only exit when 
 * workStack->finished == workStack->eigenvalues.  The ToDo chain works like a
 * stack, we pop Brackets to work on.  Each Bracket is bisected down to one
 * eigenvalue; right-halves are pushed back on the ToDo stack as future work to
 * do.
 *
 * For range=PlasmaRangeI: We still bisect on range values, but the conditions
 * change. We begin with the full range Gerschgorin [low, hi+ulp). We compute
 * nLT_low, nLT_hi, but we need the indices il, iu. So these nLT_x values are
 * the conditions: The smallest (1 relative) index the bracket can contain is
 * nLT_low+1. e.g. if nLT_low = 0, then index 1 may be in the bracket.  if
 * nLT_low=7, then index 8 may be in the bracket.
 *
 * The largest (1 relative) index the bracket can contain is nLT_hi: If
 * nLT_hi=n, then the nth eigenvalue might be in the bracket. So the bracket
 * contains indices [nLT_low+1, nLT_hi]. How does that intersect [il, iu]?
 *
 * Low end: If iu < nLT_low+1, then [il,iu] is out of Bracket.  Hi end: If il >
 * nLT_hi, then [il,iu] is out of Bracket.  Otherwise there is overlap. We find
 * the midpoint, and compute nLT_midp.
 *
 * If nLT_midp were the new high (for the left bracket), then if il > nLT_midp,
 * we can discard the left bracket. 
 *
 * If nLT_midp were the new low (for the right bracket), then if iu <
 * nLT_midp+1, we can discard the right bracket. 
 *
 * The number of eigenvalues in the bracket is still (nLT_hi - nLT_low). if
 * that is ever zero, we can discard the bracket.
 *
 * Stage: Init: establish nLT_low, nLT_hi, numEV in the bracket.
 *
 * Stage: Bisection. We use Bisection to divide the range until lowerBound and
 * upperBound cannot be averaged. (The result is one or the other).  If we
 * subdivide and the two halves both have eigenvalues, we create a new bracket
 * and (Critical) add it to the ToDo stack, with Bisection as the workstage.
 *
 * Once a range can't be subdivided (the UpperBound=LowerBound+ULP); we store
 * the lowerBound as the eigenvalue and numEV as the multiplicity in the
 * caller's array (pVal, pMul), and enter the next Stage, GetVector.
 *
 * Stage: GetVector. At this point we know the index of the eigenpair within
 * the user's arrays. If PlasmaVec, we find the eigenvector using the LAPACK
 * routine zstein, storing it in its proper place in pVec.  Then we add numEV
 * to the count of finished eigenpairs, and free the bracket.
 *
 * The most comparable serial LAPACK routine is DSTEVX.
 *
 * Once all thread work is complete, the code will condense these arrays to
 * just the actual number of unique eigenvalues found.
 *
 * This routine is most similar to a portion of LAPACK DLAEBZ.
 *****************************************************************************/

/*******************************************************************************
 * Use LAPACK zstein to find a single eigenvector.  We may use this routine
 * hundreds or thousands of times, so instead of allocating/freeing the work
 * spaces repeatedly, we have an array of pointers, per thread, to workspaces
 * we allocate if not already allocated for this thread. So we don't allocate
 * more than once per thread. These are freed by the main program before exit.
 * Returns INFO. 0=success. <0, |INFO| is invalid argument index. >0, if
 * eigenvector failed to converge.
*******************************************************************************/

static int useStein( plasma_complex64_t *diag, plasma_complex64_t *offd, 
                     plasma_complex64_t u,     plasma_complex64_t *v, 
                     int N, zlaebz2_Stein_Array_t *myArrays) {
    int M=1, LDZ=N, INFO;
    int thread = omp_get_thread_num();

    if (myArrays[thread].IBLOCK == NULL) {
        myArrays[thread].IBLOCK = (int*) calloc(N, sizeof(int));
        if (myArrays[thread].IBLOCK != NULL) myArrays[thread].IBLOCK[0]=1;
    }

    if (myArrays[thread].ISPLIT == NULL) {
        myArrays[thread].ISPLIT = (int*) calloc(N, sizeof(int));
        if (myArrays[thread].ISPLIT != NULL) myArrays[thread].ISPLIT[0]=N;
    }

    if (myArrays[thread].WORK   == NULL) myArrays[thread].WORK   = (plasma_complex64_t*) calloc(5*N, sizeof(plasma_complex64_t));
    if (myArrays[thread].IWORK  == NULL) myArrays[thread].IWORK  = (int*) calloc(N, sizeof(int));
    if (myArrays[thread].IFAIL  == NULL) myArrays[thread].IFAIL  = (int*) calloc(N, sizeof(int));
    if (myArrays[thread].IBLOCK == NULL || 
        myArrays[thread].ISPLIT == NULL || 
        myArrays[thread].WORK   == NULL || 
        myArrays[thread].IWORK  == NULL || 
        myArrays[thread].IFAIL  == NULL) {
        if(0) fprintf(stderr, "%2i:%s:%i zstein failed to allocate workspaces.\n", omp_get_thread_num(), __func__, __LINE__);
        return(PlasmaErrorOutOfMemory);
    }

    plasma_complex64_t W = u;
 
    zstein(&N, diag, offd, &M, &W, myArrays[thread].IBLOCK, myArrays[thread].ISPLIT, v, 
            &LDZ, myArrays[thread].WORK, myArrays[thread].IWORK, myArrays[thread].IFAIL, &INFO);
    if(0) fprintf(stderr, "%2i:%s:%i ev=%.16e zstein returning INFO=%d.\n", 
                            omp_get_thread_num(), __func__, __LINE__, u, INFO);

    return(INFO);
} /* end useStein. */

void plasma_zlaebz2(zlaebz2_WorkStack_t* Stack) {
    plasma_complex64_t *diag = Stack->diag;
    plasma_complex64_t *offd = Stack->offd;
    int    N = Stack->N;
 
    plasma_complex64_t cp;
    int flag, evLess;
    zlaebz2_EV_Bracket_t *myB;
 
    while (1) {
        flag = 0;
        myB = NULL;
        #pragma omp critical (UpdateStack)
        {
            if (Stack->finished == Stack->eigenvalues) flag=1;
            else if (Stack->ToDo != NULL) {
                myB = Stack->ToDo;
                Stack->ToDo = myB->next;
            }
        }
        
        /* Exit, all the work is done. */
        if (flag==1) return;
        
        /* If all the work isn't done but I couldn't find any,
         * go back and look again. Another thread must still 
         * be subdividing or working on a vector.
         */
        if (myB == NULL) continue;
        
        /* Okay, myB is popped off the stack, we must resolve it. */
        switch (myB->stage) {
            case PlasmaStageInit:
                myB->nLT_low = plasma_zlaneg2(diag, offd, N, myB->lowerBound);
                myB->nLT_hi =  plasma_zlaneg2(diag, offd, N, myB->upperBound);
                /* compute number of eigenvalues in this range. */
                myB->numEV = (myB->nLT_hi - myB->nLT_low);
                if(0) fprintf(stderr, "%2i:%s:%i On entry, #EV in [%.7f, %.7f]==%d, nLT_low=%d, nLT_hi=%d.\n", omp_get_thread_num(), __func__, __LINE__, myB->lowerBound, myB->upperBound, myB->numEV, myB->nLT_low, myB->nLT_hi);
                
                /* If no eigenvalues in this bracket, we discard it and break
                 * from the switch() to continue the while().  This happens when
                 * ranges are part of the first arbitrary range division.
                 */
                if (myB->numEV == 0) {
                    free(myB);  
                    myB=NULL;
                    break;
                }
                
                if (Stack->range == PlasmaRangeI) {
                    if (myB->nLT_hi  < Stack->il ||     /* e.g if il=500, and nLT_hi=499, this bracket is under range of interest. */
                        myB->nLT_low > Stack->iu) {     /* e.g if iu=1000, and lLT_low=1001, this bracket is above range of interest. */
                        if(0) fprintf(stderr, "Line:%i, discard myB, nLT_hi=%i, nLT_low=%i, Stack.il=%i, Stack.iu=%i, myB=%p.\n", 
                        __LINE__, myB->nLT_hi, myB->nLT_low, Stack->il, Stack->iu, myB);
                        free(myB);
                        myB=NULL;
                        break;
                    }
                }
                
                myB->stage = PlasmaStageBisection;
                /* fall-thru into Bisection. */
               
            case PlasmaStageBisection:
                flag = 0;
                while (1) {
                    cp = (myB->lowerBound+myB->upperBound)*0.5;
                    if(0) fprintf(stderr, "%2i:%s:%i lowerBound=%.16f, upperBound=%.16f, cp=%.16f, nLT_low=%i, nLT_hi=%i.\n", 
                            omp_get_thread_num(), __func__, __LINE__, myB->lowerBound, myB->upperBound, cp, myB->nLT_low, myB->nLT_hi);
                    if (cp == myB->lowerBound || cp == myB->upperBound) {
                        /* Our bracket has been narrowed to machine epsilon for this magnitude (=ulp). 
                         * We are done; the bracket is always [low,high). 'high' is not included, so
                         * we have myB->numEV eigenvalues at low, whether it == 1 or is > 1. We find
                         * the eigenvector. (We can test multiplicity with GluedWilk).
                         */

                        if(0) fprintf(stderr, "%2i:%s:%i cutpoint found eigenvalue %.16e nLT_low=%d idx=%d numEV=%i.\n", omp_get_thread_num(), __func__, __LINE__, myB->lowerBound, myB->nLT_low, myB->nLT_low - Stack->baseIdx, myB->numEV);
                        break;
                    } else {
                        /* we have a cutpoint. */
                        evLess = plasma_zlaneg2(diag, offd, N, cp);
                        if(0) fprintf(stderr, "%2i:%s:%i cp=%.16f, evLess<cp=%i, nLT_low=%i, nLT_hi=%i.\n", 
                            omp_get_thread_num(), __func__, __LINE__, cp, evLess, myB->nLT_low, myB->nLT_hi);
                        if (evLess < 0) {
                            /* We could not compute the Sturm sequence for it. */
                            flag = -1; /* indicate an error. */
                            if(0) fprintf(stderr, "Sturm Sequence compute fails for this matrix.\n");
                            break; /* exit while true. */
                        }
                    
                        /* Discard empty halves in both PlasmaRangeV and PlasmaRangeI.
                         * If #EV < cutpoint is the same as the #EV < high, it means
                         * no EV are in [cutpoint, hi]. We can discard that range.
                         */

                        if (evLess == myB->nLT_hi) {
                            myB->upperBound = cp;
                            if(0) fprintf(stderr, "%2i:%s:%i cp=%.16f Discard high range, now [%.16f, %.16f].\n", omp_get_thread_num(), __func__, __LINE__, cp, myB->lowerBound, myB->upperBound);
                            continue;
                        }
                    
                        /* If #EV < cutpoint is the same as #EV < low, it means no
                         * EV are in [low, cutpoint]. We can discard that range. 
                         */

                        if (evLess == myB->nLT_low) {
                            myB->lowerBound = cp;
                            if(0) fprintf(stderr, "%2i:%s:%i cp=%.16f Discard Low range, now [%.16f, %.16f].\n", omp_get_thread_num(), __func__, __LINE__, cp, myB->lowerBound, myB->upperBound);
                            continue;
                        }
                    
                        /* Note: If we were PlasmaRangeV, the initial bounds given by the user are the ranges,
                         * so we have nothing further to do. In PlasmaRangeI; the initial bounds are Gerschgorin,
                         * limits and not enough: We must further narrow to the desired indices.
                         */

                        if (Stack->range == PlasmaRangeI) {
                            if(0) fprintf(stderr, "%2i:%s:%i PlasmaRangeI cp=%.16f, evLess=%i, Stack->il=%i, Stack->iu=%i.\n", 
                              omp_get_thread_num(), __func__, __LINE__, cp, evLess, Stack->il, Stack->iu);

                            /* For PlasmaRangeI:
                             * Recall that il, iu are 1-relative; while evLess is zero-relative; i.e.
                             * if [il,iu]=[1,2], evless must be 0, or 1. 
                             * when evLess<cp == il-1, or just <il, cp is a good boundary and 
                             * we can discard the lower half.
                             *
                             * To judge the upper half, the cutpoint must be < iu, so if it is >= iu,
                             * cannot contain eigenvalue[iu-1].
                             * if evLess >= iu, we can discard upper half.
                             */

                            if (evLess < Stack->il) {
                                /* The lower half [lowerBound, cp) is not needed, it has no indices >= il. */
                                myB->lowerBound = cp;
                                myB->nLT_low    = evLess;
                                myB->numEV = (myB->nLT_hi-myB->nLT_low);
                                if(0) fprintf(stderr, "%2i:%s:%i cp=%.16f PlasmaRangeI: Discard Low range, now [%.16f, %.16f] #Eval=%i.\n", omp_get_thread_num(), __func__, __LINE__, cp, myB->lowerBound, myB->upperBound, myB->numEV);
                                continue;
                            }
                    
                            if (evLess >= Stack->iu) {
                                /* The upper half [cp, upperBound) is not needed, it has no indices > iu; */
                                myB->upperBound = cp;
                                myB->nLT_hi     = evLess;
                                myB->numEV = (myB->nLT_hi-myB->nLT_low);
                                if(0) fprintf(stderr, "%2i:%s:%i cp=%.16f PlasmaRangeI: Discard High range, now [%.16f, %.16f] #Eval=%i.\n", omp_get_thread_num(), __func__, __LINE__, cp, myB->lowerBound, myB->upperBound, myB->numEV);
                                continue;
                            }
                        } /*end if index search. */
                    
                        /* Here, the cutpoint has some valid EV on the left and some on the right. */
                        if(0) fprintf(stderr, "%2i:%s:%i splitting Bracket [%.16f,%.16f,%.16f], nLT_low=%i,nLT_cp=%i,nLT_hi=%i\n", 
                               omp_get_thread_num(), __func__, __LINE__, myB->lowerBound, cp, myB->upperBound, myB->nLT_low, evLess, myB->nLT_hi);
                        zlaebz2_EV_Bracket_t* newBracket = (zlaebz2_EV_Bracket_t*) calloc(1, sizeof(zlaebz2_EV_Bracket_t)); 
                        memcpy(newBracket, myB, sizeof(zlaebz2_EV_Bracket_t));
                        /* the right side: Low is cp; Hi stays the same; stage is still Bisection. */
                        newBracket->lowerBound = cp;
                        newBracket->nLT_low = evLess;
                        newBracket->numEV = (myB->nLT_hi - evLess);
                        #pragma omp critical (UpdateStack)
                        {
                            /* make new Bracket head of the ToDo work, */
                            newBracket->next = Stack->ToDo;
                            Stack->ToDo = newBracket;
                        }
                    
                        /* Update the Bracket I kept. */               
                        myB->upperBound = cp;
                        myB->nLT_hi = evLess;
                        myB->numEV =( evLess - myB->nLT_low); 
                        continue; 
                     }
                } /* end while(true) for Bisection. */
                
                /* When we are done Bisecting, we have an Eigenvalue; proceed to GetVector. */
                if(0) fprintf(stderr, "%2i:%s:%i Exit while(true), found eigenvalue %.16e nLT_low=%d idx=%d numEV=%i.\n", omp_get_thread_num(), __func__, __LINE__, myB->lowerBound, myB->nLT_low, myB->nLT_low - Stack->baseIdx, myB->numEV);
                myB->stage=PlasmaStageGetVector;
                /* fall-thru to GetVector */
            
            case PlasmaStageGetVector:
                if(0) fprintf(stderr, "%2i:%s:%i Getvector ev=%.16e idx=%d\n", omp_get_thread_num(), __func__, __LINE__, myB->lowerBound, myB->nLT_low-Stack->baseIdx);
                /* Okay, count this eigenpair done, add to the Done list.
                 * NOTE: myB->nLT_low is the global zero-relative index
                 *       of this set of mpcity eigenvalues.
                 *       No other brackets can change our entry, so we
                 *       don't need any thread block or atomicity.
                 */

                int myIdx;
                if (Stack->range == PlasmaRangeI) {
                    myIdx = myB->nLT_low - (Stack->il-1);
                } else { /* range == PlasmaRangeV */
                    myIdx = myB->nLT_low - Stack->baseIdx;
                }
                
                if (Stack->jobtype == PlasmaVec) {
                    /* get the eigenvector. */
                    int ret=useStein(diag, offd, myB->lowerBound, &(Stack->pVec[myIdx*N]), N, Stack->stein_arrays);
                    if (ret != 0) {
                        #pragma omp critical (UpdateStack)
                        {
                            if (Stack->error != 0) Stack->error = ret;
                        }
                    }
                }
                
                /* Add eigenvalue and multiplicity. */
                Stack->pVal[myIdx]=myB->lowerBound;
                Stack->pMul[myIdx]=myB->numEV;
                
                if(0) fprintf(stderr, "%2i:%s:%i Success adding eigenvector #%d myIdx=%d of %d, value %.16f, mpcity=%d\n", omp_get_thread_num(), __func__, __LINE__, myB->nLT_low, myIdx, Stack->eigenvalues, myB->lowerBound, myB->numEV);
                #pragma omp atomic 
                    Stack->finished += myB->numEV;
                
                /* Done with this bracket. */
                free(myB);
                break;
        } /* End switch on stage. */
    } /* end Master Loop. */
 
    if(0) fprintf(stderr, "%2i:%s:%i Exiting plasma_zlaebz2.\n", omp_get_thread_num(), __func__, __LINE__);
} /* end plasma_zlaebz2 */

