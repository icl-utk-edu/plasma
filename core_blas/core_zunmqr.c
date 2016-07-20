/**
 *
 * @file core_zunmqr.c
 *
 *  PLASMA core_blas kernel
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver and
 *  Univ. of Manchester.
 *
 * @version 3.0.0
 * @author Hatem Ltaief
 * @author Mathieu Faverge
 * @author Jakub Kurzak
 * @author Jakub Sistek
 * @date 2016-7-8
 * @precisions normal z -> c d s
 *
 **/

#include "core_blas.h"
#include "plasma_types.h"
#include "plasma_internal.h"

#ifdef PLASMA_WITH_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

/***************************************************************************//**
 *
 * @ingroup CORE_PLASMA_Complex64_t
 *
 *  Overwrites the general m-by-n tile C with
 *
 *                    SIDE = 'L'     SIDE = 'R'
 *    TRANS = 'N':      Q * C          C * Q
 *    TRANS = 'T':      Q' * C         C * Q'
 *    TRANS = 'C':      Q**H * C       C * Q**H
 *
 *  where Q is a unitary matrix defined as the product of k
 *  elementary reflectors
 *    \f[ 
 *        Q = H(1) H(2) ... H(k)
 *    \f]
 *  as returned by CORE_zgeqrt. Q is of order m if side = 'L' and of order n
 *  if side = 'R'.
 *
 *******************************************************************************
 *
 * @param[in] side
 *         - PlasmaLeft  : apply Q or Q**H from the Left;
 *         - PlasmaRight : apply Q or Q**H from the Right.
 *
 * @param[in] trans
 *         - PlasmaNoTrans   :  No transpose, apply Q;
 *         - PlasmaTrans     :  Transpose, apply Q'.
 *         - PlasmaConjTrans :  Transpose, apply Q**H.
 *
 * @param[in] m
 *         The number of rows of the tile C.  m >= 0.
 *
 * @param[in] n
 *         The number of columns of the tile C.  n >= 0.
 *
 * @param[in] k
 *         The number of elementary reflectors whose product defines
 *         the matrix Q.
 *         If side = PlasmaLeft,  m >= k >= 0;
 *         if side = PlasmaRight, n >= k >= 0.
 *
 * @param[in] ib
 *         The inner-blocking size.  ib >= 0.
 *
 * @param[in] nb
 *         Number of rows in a block.  nb >= 0.
 *
 * @param[in] A
 *         Dimension:  (lda,k)
 *         The i-th column must contain the vector which defines the
 *         elementary reflector H(i), for i = 1,2,...,k, 
 *         as returned by CORE_zgeqrt in the first k columns of its 
 *         array argument A.
 *
 * @param[in] lda
 *         The leading dimension of the array A.
 *         If side = PlasmaLeft,  lda >= max(1,m);
 *         if side = PlasmaRight, lda >= max(1,n).
 *
 * @param[in] T
 *         The ib-by-k triangular factor T of the block reflector.
 *         T is upper triangular by block (economic storage);
 *         The rest of the array is not referenced.
 *
 * @param[in] ldt
 *         The leading dimension of the array T. ldt >= ib.
 *
 * @param[in,out] C
 *         On entry, the m-by-n tile C.
 *         On exit, C is overwritten by Q*C or Q**T*C or C*Q**T or C*Q.
 *
 * @param[in] ldc
 *         The leading dimension of the array C. ldc >= max(1,m).
 *
 ******************************************************************************/
void CORE_zunmqr(PLASMA_enum side, PLASMA_enum trans,
                 int m, int n, int k, int ib, int nb,
                 const PLASMA_Complex64_t *A, int lda,
                 const PLASMA_Complex64_t *T, int ldt,
                 PLASMA_Complex64_t *C, int ldc)
{
    // prepare memory for the auxiliary array
    int lwork = ib*nb;
    PLASMA_Complex64_t *WORK = (PLASMA_Complex64_t *) malloc(sizeof(PLASMA_Complex64_t) * lwork);
    int ldwork = nb;

    int i, kb;
    int i1, i3;
    int nq, nw;
    int ic = 0;
    int jc = 0;
    int ni = n;
    int mi = m;

    // Check input arguments
    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        plasma_error("Illegal value of side");
        return;
    }
    // nq is the order of Q and nw is the minimum dimension of WORK
    if (side == PlasmaLeft) {
        nq = m;
        nw = n;
    }
    else {
        nq = n;
        nw = m;
    }

    // Plasma_ConjTrans will be converted to PlasmaTrans in 
    // automatic datatype conversion, which is what we want here.
    // PlasmaConjTrans is protected from this conversion.
    if ((trans != PlasmaNoTrans) && (trans != Plasma_ConjTrans)) {
        plasma_error("Illegal value of trans");
        return;
    }
    if (m < 0) {
        plasma_error("Illegal value of m");
        return;
    }
    if (n < 0) {
        plasma_error("Illegal value of n");
        return;
    }
    if ((k < 0) || (k > nq)) {
        plasma_error("Illegal value of k");
        return;
    }
    if ((ib < 0) || ( (ib == 0) && ((m > 0) && (n > 0)) )) {
        plasma_error("Illegal value of ib");
        return;
    }
    if ((lda < imax(1,nq)) && (nq > 0)) {
        plasma_error("Illegal value of lda");
        return;
    }
    if ((ldc < imax(1,m)) && (m > 0)) {
        plasma_error("Illegal value of ldc");
        return;
    }
    if ((ldwork < imax(1,nw)) && (nw > 0)) {
        plasma_error("Illegal value of ldwork");
        return;
    }

    // Quick return
    if ((m == 0) || (n == 0) || (k == 0))
        return;

    if (((side == PlasmaLeft) && (trans != PlasmaNoTrans))
        || ((side == PlasmaRight) && (trans == PlasmaNoTrans))) {
        i1 = 0;
        i3 = ib;
    }
    else {
        i1 = ( ( k-1 ) / ib )*ib;
        i3 = -ib;
    }

    for(i = i1; (i >- 1) && (i < k); i+=i3 ) {
        kb = imin(ib, k-i);

        if (side == PlasmaLeft) {
            // H or H' is applied to C(i:m,1:n)
            mi = m - i;
            ic = i;
        }
        else {
            // H or H' is applied to C(1:m,i:n)
            ni = n - i;
            jc = i;
        }
        // Apply H or H'
        LAPACKE_zlarfb_work(
            LAPACK_COL_MAJOR,
            lapack_const(side),
            lapack_const(trans),
            lapack_const(PlasmaForward),
            lapack_const(PlasmaColumnwise),
            mi, ni, kb,
            &A[lda*i+i], lda,
            &T[ldt*i], ldt,
            &C[ldc*jc+ic], ldc,
            WORK, ldwork);
    }

    // deallocate auxiliary array
    free(WORK);
}

/******************************************************************************/
void CORE_OMP_zunmqr(PLASMA_enum side, PLASMA_enum trans,
                     int m, int n, int k, int ib, int nb,
                     const PLASMA_Complex64_t *A, int lda,
                     const PLASMA_Complex64_t *T, int ldt,
                     PLASMA_Complex64_t *C,       int ldc)
{
#pragma omp task depend(in:A[0:nb*nb]) \
                 depend(in:T[0:ib*nb]) \
                 depend(inout:C[0:nb*nb])
    CORE_zunmqr(side, trans,
                m, n, k, ib, nb,
                A, lda,
                T, ldt,
                C, ldc);
}
