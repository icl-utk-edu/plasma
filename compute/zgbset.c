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
#include "plasma_types.h"
#include "plasma_workspace.h"

/***************************************************************************//**
    @ingroup plasma_gbmm

    zero out all elements outside the band of a matrix described by
    pA, lda, m, n, kl, ku
*/
void plasma_zgbset(int m, int n, int kl, int ku,
                   plasma_complex64_t *pA, int lda)
{
    // quick return
    if (m == 0 || n == 0)
        return;

    if ((kl < 0) ||
        (kl > m-1)) {
        plasma_error("illegal value of kl");
        return;
    }
    if ((ku < 0) ||
        (ku > n-1)) {
        plasma_error("illegal value of ku");
        return;
    }
    // square matrix OR rectangular tall matrix
    if(m>=n)
    {
        int cornerI, cornerJ, extralower, extraupper;
        if(kl+ku < m) // can be < or <=, the difference is zlaset on m=0
        {
            plasma_zlaset(PlasmaGeneral, m-kl-ku, n-1, 0, 0, pA+1+kl, lda+1);
            extralower = ku-1;
            extraupper = kl-1;
        }
        else
        {
            extralower = m-1-kl;
            extraupper = n-1-ku;
        }
        for(; extralower > 0; extralower--)
        {
            cornerI = m-extralower;
            plasma_zlaset(PlasmaGeneral,1,(extralower < n ? extralower: n),
                                          0,0,pA+cornerI,lda+1);
        }
        for(; extraupper > 0; extraupper--)
        {
            cornerJ = n-extraupper;
            plasma_zlaset(PlasmaGeneral,1,extraupper,0,0,pA+lda*cornerJ,lda+1);
        }
        if(m>n+kl) // m strictly greater
        // clears a "tail" on bottom right below diagonal
        // note that for small bandwidths on large rectangular matrices of
        // this kind will lead the above code to neglect zeroing elements
        // in the final column of pA, which need to be cleared if 
        // m > n+kl
        {
            plasma_zlaset(PlasmaGeneral, (m-n)-kl, 1, 0, 0,
                            pA+(lda*(n-1))+(n+kl), lda);
        }
    }
    // wide rectangular matrix
    else /* if(n>m) */
    {
        // clear out the square part of the matrix (left side).
        // so we will use all 'm's instead of 'm's and 'n's. We 
        // will clear out the right side later.
        int cornerI, cornerJ, extralower, extraupper;
        if(kl+ku < m)
        {
            plasma_zlaset(PlasmaGeneral, m-kl-ku, m-1, 0, 0, pA+1+kl, lda+1);
            extralower = ku-1;
            extraupper = kl-1;
        }
        else
        {
            extralower = m-1-kl;
            extraupper = m-1-ku;
        }
        for(; extralower > 0; extralower--)
        {
            cornerI = m-extralower;
            plasma_zlaset(PlasmaGeneral,1,extralower,0,0,pA+cornerI,lda+1);
        }
        for(; extraupper > 0; extraupper--)
        {
            cornerJ = m-extraupper;
            plasma_zlaset(PlasmaGeneral,1,extraupper,0,0,pA+lda*cornerJ,lda+1);
        }
        // now zero out the right side
        // i: the row that we are zeroing
        for(int i = 0; i < m; i++)
        {
            // blcols: the number of columns in the row that belong to the band
            int blcols = (i-(m-1)+ku) > 0 ? (i-(m-1)+ku) : 0;
            if (n-(m+blcols) <= 0)
            {
               continue;
            }
            plasma_zlaset(PlasmaGeneral,1,n-(m+blcols),0,0,
                                          pA+(lda*(m+blcols))+i,lda);
        }
    }

}
