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

#define A(m, n) ((PLASMA_Complex64_t*)plasma_getaddr(A, (m), (n)))

/***************************************************************************//**
 *
 * @ingroup CORE_PLASMA_Complex64_t
 *
 *  CORE_zlaswp_ontile apply the zlaswp function on a block column of a matrix 
 *  stored in tile layout
 *
 *******************************************************************************
 *
 *  @param[in,out] A
 *          The descriptor of the matrix A to permute.
 *
 *  @param[in] i1
 *          The first element of IPIV for which a row interchange will
 *          be done.
 *
 *  @param[in] i2
 *          The last element of IPIV for which a row interchange will
 *          be done.
 *
 *  @param[in] ipiv
 *          The pivot indices; Only the element in position i1 to i2
 *          are accessed. The pivot are offset by A.i.
 *
 *  @param[in] inc
 *          The increment between successive values of IPIV.  If IPIV
 *          is negative, the pivots are applied in reverse order.
 *
 *******************************************************************************
 *
 * @return
 *         \retval PLASMA_SUCCESS successful exit
 *         \retval <0 if INFO = -k, the k-th argument had an illegal value
 *
 *******************************************************************************
 */
int CORE_zlaswp_ontile(PLASMA_desc A, int i_, int j_, int m, int n,
                       int i1, int i2, const int *ipiv, int inc)
{
    int i, j, ip, it;
    PLASMA_Complex64_t *A1;
    int lda1, lda2;

    /* Change i1 to C notation */
    i1--;

    /* Check parameters */
    /*if ( A.nt > 1 ) {
        //coreblas_error(1, "Illegal value of A.nt");
        return -1;
    }*/
    if ( i1 < 0 ) {
        //coreblas_error(2, "Illegal value of i1");
        return -2;
    }
    if ( (i2 <= i1) || (i2 > m) ) {
        //coreblas_error(3, "Illegal value of i2");
        return -3;
    }
    if ( ! ( (i2 - i1 -1) < A.mb ) ) {
        //coreblas_error(2, "Illegal value of i1,i2. They have to be part of the same block.");
        return -3;
    }

    if (inc > 0) {
        it = i1 / A.mb;
        A1 = A(i_+it, j_);
        lda1 = BLKLDD(A, i_);

        for (j = i1; j < i2; ++j, ipiv+=inc) {
            ip = (*ipiv) - 1;
            if ( ip != j )
            {
                it = ip / A.mb;
                i  = ip % A.mb;
                lda2 = BLKLDD(A, i_+it);
                cblas_zswap(n, A1           + j, lda1,
                               A(i_+it, j_) + i, lda2 );
            }
        }
    }
    else
    {
        it = (i2-1) / A.mb;
        A1 = A(i_+it, j_);
        lda1 = BLKLDD(A, i_+it);

        i1--;
        ipiv = &ipiv[(1-i2)*inc];
        for (j = i2-1; j > i1; --j, ipiv+=inc) {
            ip = (*ipiv) - 1;
            if ( ip != j )
            {
                it = ip / A.mb;
                i  = ip % A.mb;
                lda2 = BLKLDD(A, i_+it);
                cblas_zswap(n, A1           + j, lda1,
                               A(it+i_, j_) + i, lda2 );
            }
        }
    }

    return PLASMA_SUCCESS;
}


/******************************************************************************/
void CORE_OMP_zlaswp_ontile(PLASMA_desc A, int i_, int j_, int m, int n,
                            int i1, int i2, const int *ipiv, int inc)
{
    int lda = A.lm - i_*A.mb;
    PLASMA_Complex64_t *Aij = A(i_, j_);
    #pragma omp task depend(inout:Aij[0:lda*n]) \
                     depend(in:ipiv[0])
    CORE_zlaswp_ontile(A, i_, j_, m, n, i1, i2, ipiv, inc);
}
