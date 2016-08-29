/**
  *
  * @file core_zgemm.c
  *
  *  PLASMA is a software package provided by:
  *  University of Tennessee, US,
  *  University of Manchester, UK.
  *
  * @precisions normal z -> c d s
  *
  **/

#include "core_blas.h"
#include "plasma_descriptor.h"
#include "plasma_types.h"

#ifdef PLASMA_WITH_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

#define KUT ((descA.ku+descA.kl+descA.nb-1)/descA.nb) 
#define A(m,n) ((PLASMA_Complex64_t*)plasma_getaddr(descA, KUT+(i_+(m))-(j_+(n)), j_+(n)))
#define BLKLDD_BAND(A, m, n) BLKLDD((A), KUT+(i_+(m))-(j_+(n)))

/*******************************************************************************
 * 
 */
int CORE_zlaswp_ontile_fill(PLASMA_desc descA, int i_, int j_, int n,
                            int i1, int i2, const int *ipiv, int inc)
{
    int i, j, ip, it;
    PLASMA_Complex64_t *A1;
    int lda1, lda2;

    /* Change i1 to C notation */
    i1--;

    /* Check parameters */
    /*if ( descA.nt > 1 ) {
        //coreblas_error(1, "Illegal value of descA.nt");
        return -1;
    }*/
    if ( i1 < 0 ) {
        //coreblas_error(2, "Illegal value of i1");
        return -2;
    }
    if ( (i2 <= i1) || (i2 > descA.m) ) {
        //coreblas_error(3, "Illegal value of i2");
        return -3;
    }
    if ( ! ( (i2 - i1 - i1%descA.mb -1) < descA.mb ) ) {
        //coreblas_error(2, "Illegal value of i1,i2. They have to be part of the same block.");
        return -3;
    }

    if (inc > 0) {
        it = i1 / descA.mb;
        A1 = A(it, 0);
        lda1 = BLKLDD_BAND(descA, it, 0);

        for (j = i1; j < i2; ++j, ipiv+=inc) {
            ip = (*ipiv) - 1;
            if ( ip != j )
            {
                it = ip / descA.mb;
                i  = ip % descA.mb;
                lda2 = BLKLDD_BAND(descA, it, 0);
                cblas_zswap(n, A1       + j, lda1,
                               A(it, 0) + i, lda2 );
            }
        }
    }
    else
    {
        it = (i2-1) / descA.mb;
        A1 = A(it, 0);
        lda1 = BLKLDD_BAND(descA, it, 0);

        i1--;
        ipiv = &ipiv[(1-i2)*inc];
        for (j = i2-1; j > i1; --j, ipiv+=inc) {
            ip = (*ipiv) - descA.i - 1;
            if ( ip != j )
            {
                it = ip / descA.mb;
                i  = ip % descA.mb;
                lda2 = BLKLDD_BAND(descA, it, 0);
                cblas_zswap(n, A1       + j, lda1,
                               A(it, 0) + i, lda2 );
            }
        }
    }

    return PLASMA_SUCCESS;
}

/*******************************************************************************
 * 
 */
int CORE_zswptr_ontile_fill(PLASMA_desc descA, int i_, int j_, int m, int n, 
                            int i1, int i2, const int *ipiv, int inc,
                            const PLASMA_Complex64_t *Akk, int ldak,
                            int j, int *fill)
{
    PLASMA_Complex64_t zone  = 1.0;

    /*if ( descA.nt > 1 ) {
        //coreblas_error(1, "Illegal value of descA.nt");
        return -1;
    }*/
    if ( i1 < 1 ) {
        //coreblas_error(2, "Illegal value of i1");
        return -2;
    }
    if ( (i2 < i1) || (i2 > m) ) {
        //coreblas_error(3, "Illegal value of i2");
        return -3;
    }

    if (j <= fill[0]) {
        CORE_zlaswp_ontile_fill(descA, i_, j_, n, i1, i2, ipiv, inc);

        int mb = i_ == descA.mt-1 ? descA.m-i_*descA.mb : descA.mb;
        int lda = BLKLDD_BAND(descA, 0, 0);
        cblas_ztrsm( CblasColMajor, CblasLeft, CblasLower,
                     CblasNoTrans, CblasUnit,
                     mb, n, CBLAS_SADDR(zone),
                     Akk,     ldak,
                     A(0, 0), lda );
    }
    return PLASMA_SUCCESS;
}

/******************************************************************************/
void CORE_OMP_zswptr_ontile_fill(PLASMA_desc descA, int i_, int j_, int m, int n,
                                 int i1, int i2, const int *ipiv, int inc,
                                 const PLASMA_Complex64_t *Akk, int ldak,
                                 int j, int *fill, int *fake)
{
    int info;
    PLASMA_Complex64_t *Akn = A(0,0);
    #pragma omp task depend(in:Akk[0:ldak*m]) \
                     depend(inout:Akn[0:m*n]) \
                     depend(in:ipiv[0])       \
                     depend(inout:fake[0])       
    info = 
    CORE_zswptr_ontile_fill(descA, i_, j_, m, n,
                            i1, i2, ipiv, inc,
                            Akk, ldak,
                            j, fill);

    return;
}
