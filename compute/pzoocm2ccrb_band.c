/**
 *
 * @file pzoocm2ccrb.c
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> s d c
 *
 **/

#include "plasma_async.h"
#include "plasma_descriptor.h"
#include "plasma_types.h"
#include "plasma_internal.h"
#include "core_blas_z.h"

#define KUT ((A.ku+A.kl+A.nb-1)/A.nb) // number of tiles in upper band (not including diagonal)
#define KLT ((A.kl+A.nb)/A.nb-1)      // number of tiles in lower band (not including diagonal)
#define LDtile(m, n) BLKLDD(A, KUT+(m)-(n))
#define tileA(m, n) ((PLASMA_Complex64_t*)plasma_getaddr(A, KUT+(m)-(n), (n)))
#define bandA(m, n) (&(Af77[lda*(A.nb*(n)) + (A.ku+A.kl)+A.mb*((m)-(n))]))

/******************************************************************************/
void plasma_pzoocm2ccrb_band(PLASMA_Complex64_t *Af77, int lda, PLASMA_desc A,
                             PLASMA_sequence *sequence, PLASMA_request *request)
{
    int n, m;

    if (sequence->status != PLASMA_SUCCESS)
        return;
    
    for (n = 0; n < A.nt; n++)
    {
        int i_start = (imax(0, n*A.nb-A.ku-A.kl)) / A.nb;
        int i_end = (imin(A.m-1, (n+1)*A.nb+A.kl-1)) / A.nb;
        for (m = i_start; m <= i_end; m++)
        {
            int mb = imin(A.mb, A.m-m*A.mb);
            int nb = imin(A.nb, A.n-n*A.nb);
            CORE_OMP_zlacpy_lapack2tile_band(
                   m, n, mb, nb, A.mb, A.kl, A.ku,
                   bandA(m, n), lda-1,
                   tileA(m, n), LDtile(m, n));
                   //tileA(i_start,n), nb*nb, INOUT | GATHERV);
        }
    }
}
