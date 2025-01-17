/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
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
#include <core_lapack.h>

/***************************************************************************//**
 *
 * @ingroup plasma_gemm
 *
 * This code is not designed to be called directly by users; it is a subroutine
 * for dstevx2.c.
 *
 * Specifically, this is a task-based parallel algorithm, the parameters are
 * contained in the already initialized and populated dlaebz2_Control_t; For
 * example, from dstevx2:
 *
 *  #pragma omp parallel
 *  {
 *      #pragma omp single
 *      {
 *          plasma_dlaebz2(&Control, ...etc...);
 *      }
 *  }
 *
 *
 *******************************************************************************
 *
 * @param[in] *Control
 *          A pointer to the global variables needed.
 *
 * @param[in] Control->N
 *          int number of rows in the matrix.
 *
 * @param[in] Control->diag
 *          real array of [N] diagonal elements of the matrix.
 *
 * @param[in] Control->offd
 *          real array of [N-1] sub-diagonal elements of the matrix.
 *
 * @param[in] Control->range
 *          int enum.
 *              PlasmaRangeI if user is finding eigenvalues by index range.
 *              PlasmaRangeV if user is finding eigenvuales by value range.
 *
 * @param[in] Control->jobtype
 *          int enum.
 *              PlasmaNoVec if user does not want eigenvectors computed.
 *              PlasmaVec if user desires eigenvectors computed.
 *
 * @param[in] Control->il
 *          int enum. The lowerBound of an index range if range is
 *          PlasmaRangeI.
 *
 * @param[in] Control->iu
 *          int enum. The upperBound of an index range, if range is
 *          PlasmaRangeI.
 *
 * @param[in] Control->stein_arrays
 *          array of [max_threads], type dlaebz2_Stein_Array_t, contains work
 *          areas per thread for invoking _stein (inverse iteration to find
 *          eigenvectors).
 *
 * @param[in] Control->baseIdx
 *          The index of the least eigenvalue to be found in the bracket,
 *          used to calculate the offset into the return vectors/arrays.
 *
 * @param[out] Control->error
 *          If non-zero, the first error we encountered in the operation.
 *
 * @param[out] Control->pVal
 *          real vector of [eigenvaues] to store the eigenvalues discovered,
 *          these are returned in ascending sorted order.
 *
 * @param[out] Control->pVec
 *          real array of [N x eigenvalues] to store the eigenvectors, not
 *          references unless jobtype==PlasmaVec. Stored in the same order as
 *          their corresponding eigenvalue. Only referenced if jobtype is
 *          PlasmaVec.
 *
 * @param[out] Control->pMul
 *          int vector of [eigenvalues], the corresponding ULP-multiplicity of
 *          each eigenvalue, typically == 1.
 *
 * @param[in] lowerBound
 *          Real lowerBound (inclusive) for range of eigenvalues to find.
 *
 * @param[in] upperBound
 *          Real upperBound (non-inclusive) of the range of eigenvalues to find.
 *
 * @param[in] nLT_low
 *          int number of eigenvalues less than lowerBound. Computed if < 0.
 *
 * @param[in] nLT_hi
 *          int number of eigevalues less than upperBound. Computed if < 0.
 *
 * @param[in] numEV
 *          int number of eigenvalues in [lowerBound, upperBound). Computed if
 *          either nLT_low or nLT_hi were computed.
 *
 * A 'bracket' is a range of either real eigenvalues, or eigenvalue indices,
 * that this code is given to discover. It is provided in the arguments.  Upon
 * entry, the number of theoretical eigenvalues in this range has already been
 * determined, but the actual number may be less, due to ULP-multiplicity. (ULP
 * is the Unit of Least Precision, the magnitude of the smallest change
 * possible to a given real number). To explain: A real symmetric matrix in NxN
 * should have N distinct real eigenvalues; however, if eigenvalues are closely
 * packed either absolutely (their difference is close to zero) or relatively
 * (their ratio is close to 1.0) then in real arithmetic two such eigenvalues
 * may be within ULP of each other, and thus represented by the same real
 * number. Thus we have ULP-multiplicity, two theoretically distinct
 * eigenvalues represented by the same real number.
 *
 *
 * This algorithm uses Bisection by the Scaled Sturm Sequence, implemented in
 * plasma_dlaebz2, followed by the LAPACK routine _STEIN, which uses inverse
 * iteration to find the eigenvalue.  The initial 'bracket' parameters should
 * contain the full range for the eigenvalues we are to discover. The algorithm
 * is recursively task based, at each division the bracket is divided into two
 * brackets. If either is empty (no eigenvalues) we discard it, otherwise a new
 * task is created to further subdivide the right-hand bracket while the
 * current task continues dividing the left-hand side, until it can no longer
 * divide it, and proceeds to store the eigenvalue and compute the eigenvector
 * if needed. Thus the discovery process is complete when all tasks are
 * completed. We then proceed to orthogonalizing any eigenvectors discovered;
 * because inverse iteration does not inherently ensure orthogonal
 * eigenvectors.
 *
 * The most comparable serial LAPACK routine is DLAEBZ.
 *
 * Once all thread work is complete, the code will condense these arrays to
 * just the actual number of unique eigenvalues found, if any ULP-multiplicity
 * is present.
 *****************************************************************************/

/*******************************************************************************
 * Use LAPACK dstein to find a single eigenvector.  We may use this routine
 * multiple times, so instead of allocating/freeing the work spaces repeatedly,
 * we have an array of pointers, per thread, to workspaces we allocate if not
 * already allocated for this thread. So we don't allocate more than once per
 * thread. These are freed by the main program before exit.  Returns INFO.
 * 0=success. <0, |INFO| is invalid argument index. >0, if eigenvector failed
 * to converge.
*******************************************************************************/

int plasma_dstein( double *diag, double *offd,
        double u,     double *v, int N,
        dlaebz2_Stein_Array_t *myArrays) {
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

    if (myArrays[thread].WORK   == NULL) myArrays[thread].WORK   = (double*) calloc(5*N, sizeof(double));
    if (myArrays[thread].IWORK  == NULL) myArrays[thread].IWORK  = (int*) calloc(N, sizeof(int));
    if (myArrays[thread].IFAIL  == NULL) myArrays[thread].IFAIL  = (int*) calloc(N, sizeof(int));
    if (myArrays[thread].IBLOCK == NULL ||
        myArrays[thread].ISPLIT == NULL ||
        myArrays[thread].WORK   == NULL ||
        myArrays[thread].IWORK  == NULL ||
        myArrays[thread].IFAIL  == NULL) {
        return(PlasmaErrorOutOfMemory);
    }

    double W = u;

    /* We use the 'work' version so we can re-use our work arrays; using LAPACKE_dstein() */
    /* would re-allocate and release work areas on every call.                            */
    INFO = LAPACKE_dstein_work(LAPACK_COL_MAJOR, N, diag, offd, M, &W, myArrays[thread].IBLOCK,
            myArrays[thread].ISPLIT, v, LDZ, myArrays[thread].WORK, myArrays[thread].IWORK,
            myArrays[thread].IFAIL);
    return(INFO);
}

/******************************************************************************
 * This a task that subdivides a bracket, throwing off other tasks like this
 * if necessary, until the bracket zeroes in on a single eigenvalue, which it
 * then stores and possibly finds the corresponding eigenvector.
 * Parameters:
 *      Control:    Global variables.
 *      lowerBound: of bracket to subdivide.
 *      upperBound: of bracket to subdivide.
 *      nLT_low:    number of eigenvalues less than lower bound.
 *                  -1 if it needs to be found.
 *      nLT_hi:     number of eigevalues less than the upper bound.
 *                  -1 if it needs t obe found.
 *      numEV:      number of eigenvalues within bracket. Computed if either
 *                  nLT_Low or nLT_hi is computed.
 * ***************************************************************************/

void plasma_dlaebz2(dlaebz2_Control_t *Control, double lowerBound,
        double upperBound, int nLT_low, int nLT_hi, int numEV) {

    double *diag = Control->diag;
    double *offd = Control->offd;
    int    N = Control->N;

    double cp;
    int flag=0, evLess;

    if (nLT_low < 0) {
        nLT_low = plasma_dlaneg2(diag, offd, N, lowerBound);
        flag=1;
    }

    if (nLT_hi < 0) {
        nLT_hi =  plasma_dlaneg2(diag, offd, N, upperBound);
        flag=1;
    }

    if (flag) {
        numEV = (nLT_hi - nLT_low);
    }

    /* If there are no eigenvalues in the supplied range, we are done. */
    if (numEV < 1) return;

    if (Control->range == PlasmaRangeI) {
        if (nLT_hi  < Control->il ||    /* e.g if il=500, and nLT_hi=499, this bracket is under range of interest. */
            nLT_low > Control->iu) {    /* e.g if iu=1000, and nLT_low=1001, this bracket is above range of interest. */
            return;
        }
    }

    /* Bisect the bracket until we can't anymore. */

    flag = 0;
    for (;;) {
        cp = (lowerBound+upperBound)*0.5;
        if (cp == lowerBound || cp == upperBound) {
            /* Our bracket has been narrowed to machine epsilon for this magnitude (=ulp).
             * We are done; the bracket is always [low,high). 'high' is not included, so
             * we have numEV eigenvalues at low, whether it == 1 or is > 1. We find
             * the eigenvector. (We can test multiplicity with GluedWilk).
             */
            break; /* exit for(;;). */
        } else {
            /* we have a new cutpoint. */
            evLess = plasma_dlaneg2(diag, offd, N, cp);
            if (evLess < 0) {
                /* We could not compute the Sturm sequence for it. */
                flag = -1; /* indicate an error. */
                break; /* exit for (;;). */
            }

            /* Discard empty halves in both PlasmaRangeV and PlasmaRangeI.
             * If #EV < cutpoint is the same as the #EV < high, it means
             * no EV are in [cutpoint, hi]. We can discard that range.
             */

            if (evLess == nLT_hi) {
                upperBound = cp;
                continue;
            }

            /* If #EV < cutpoint is the same as #EV < low, it means no
             * EV are in [low, cutpoint]. We can discard that range.
             */

            if (evLess == nLT_low) {
                lowerBound = cp;
                continue;
            }

            /* Note: If we were PlasmaRangeV, the initial bounds given by the user are the ranges,
             * so we have nothing further to do. In PlasmaRangeI; the initial bounds are Gerschgorin
             * limits and not enough: We must further narrow to the desired indices.
             */

            if (Control->range == PlasmaRangeI) {
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

                if (evLess < Control->il) {
                    /* The lower half [lowerBound, cp) is not needed, it has no indices >= il. */
                    lowerBound = cp;
                    nLT_low    = evLess;
                    numEV = (nLT_hi-nLT_low);
                    continue;
                }

                if (evLess >= Control->iu) {
                    /* The upper half [cp, upperBound) is not needed, it has no indices > iu; */
                    upperBound = cp;
                    nLT_hi     = evLess;
                    numEV = (nLT_hi-nLT_low);
                    continue;
                }
            } /*end if index search. */

            /* Here, the cutpoint has EV on both left right. We push off the right bracket.
             * The new lowerBound is the cp, the upperBound is unchanged, the number of
             * eigenvalues changes. */
            #pragma omp task
                plasma_dlaebz2(Control, cp, upperBound, evLess, nLT_hi, (nLT_hi-evLess));

            /* Update the Left side I kept. The new number of EV less than upperBound
             * is evLess, recompute number of EV in the bracket. */
            upperBound = cp;
            nLT_hi = evLess;
            numEV =( evLess - nLT_low);
            continue;
         }
    } /* end for (;;) for Bisection. */

    /* Okay, count this eigenpair done, add to the Done list.
     * NOTE: nLT_low is the global zero-relative index of
     *       this set of mpcity eigenvalues.
     *       No other brackets can change our entry, so we
     *       don't need any thread block or atomicity.
     */

    int myIdx;
    if (Control->range == PlasmaRangeI) {
        myIdx = nLT_low - (Control->il-1);
    } else { /* range == PlasmaRangeV */
        myIdx = nLT_low - Control->baseIdx;
    }

    if (Control->jobtype == PlasmaVec) {
        /* get the eigenvector. */
        int ret=plasma_dstein(diag, offd, lowerBound, &(Control->pVec[myIdx*N]), N, Control->stein_arrays);
        if (ret != 0) {
            #pragma omp critical (UpdateStack)
            {
                /* Only store first error we encounter */
                if (Control->error == 0) Control->error = ret;
            }
        }
    }

    /* Add eigenvalue and multiplicity. */
    Control->pVal[myIdx]=lowerBound;
    Control->pMul[myIdx]=numEV;

//    #pragma omp atomic
//        Control->finished += numEV;
}

