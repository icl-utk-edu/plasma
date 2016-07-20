/**
 *
 * @file core_ztsmqr.c
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
 * @author Azzam Haidar
 * @author Dulceneia Becker
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
 *  Overwrites the general m1-by-n1 tile A1 and
 *  m2-by-n2 tile A2 with
 *
 *                        SIDE = 'L'        SIDE = 'R'
 *    TRANS = 'N':         Q * | A1 |     | A1 A2 | * Q
 *                             | A2 |
 *
 *    TRANS = 'T':        Q' * | A1 |     | A1 A2 | * Q'
 *                             | A2 |
 *
 *    TRANS = 'C':      Q**H * | A1 |     | A1 A2 | * Q**H
 *                             | A2 |
 *
 *  where Q is a complex unitary matrix defined as the product of k
 *  elementary reflectors
 *
 *    Q = H(1) H(2) . . . H(k)
 *
 *  as returned by CORE_ZTSQRT.
 *
 *******************************************************************************
 *
 * @param[in] side
 *         - PlasmaLeft  : apply Q or Q' from the Left;
 *         - PlasmaRight : apply Q or Q' from the Right.
 *
 * @param[in] trans
 *         - PlasmaNoTrans   :  No transpose, apply Q;
 *         - PlasmaTrans     :  Transpose, apply Q'.
 *         - PlasmaConjTrans :  ConjTranspose, apply Q**H.
 *
 * @param[in] m1
 *         The number of rows of the tile A1. m1 >= 0.
 *
 * @param[in] n1
 *         The number of columns of the tile A1. n1 >= 0.
 *
 * @param[in] m2
 *         The number of rows of the tile A2. m2 >= 0.
 *         m2 = m1 if side == PlasmaRight.
 *
 * @param[in] n2
 *         The number of columns of the tile A2. n2 >= 0.
 *         n2 = n1 if side == PlasmaLeft.
 *
 * @param[in] k
 *         The number of elementary reflectors whose product defines
 *         the matrix Q.
 *
 * @param[in] ib
 *         The inner-blocking size.  ib >= 0.
 *
 * @param[in] nb
 *         Number of rows in a block.  nb >= 0.
 *
 * @param[in,out] A1
 *         On entry, the m1-by-n1 tile A1.
 *         On exit, A1 is overwritten by the application of Q.
 *
 * @param[in] lda1
 *         The leading dimension of the array A1. lda1 >= max(1,m1).
 *
 * @param[in,out] A2
 *         On entry, the m2-by-n2 tile A2.
 *         On exit, A2 is overwritten by the application of Q.
 *
 * @param[in] lda2
 *         The leading dimension of the tile A2. lda2 >= max(1,m2).
 *
 * @param[in] V
 *         The i-th row must contain the vector which defines the
 *         elementary reflector H(i), for i = 1,2,...,k, as returned by
 *         CORE_ZTSQRT in the first k columns of its array argument V.
 *
 * @param[in] ldv
 *         The leading dimension of the array V. ldv >= max(1,k).
 *
 * @param[in] T
 *         The ib-by-k triangular factor T of the block reflector.
 *         T is upper triangular by block (economic storage);
 *         The rest of the array is not referenced.
 *
 * @param[in] ldt
 *         The leading dimension of the array T. ldt >= ib.
 *
 ******************************************************************************/
void CORE_ztsmqr(PLASMA_enum side, PLASMA_enum trans,
                 int m1, int n1, int m2, int n2, int k, int ib, int nb,
                 PLASMA_Complex64_t *A1, int lda1,
                 PLASMA_Complex64_t *A2, int lda2,
                 const PLASMA_Complex64_t *V, int ldv,
                 const PLASMA_Complex64_t *T, int ldt)
{
    // prepare memory for the auxiliary array
    int lwork = ib*nb;
    PLASMA_Complex64_t *WORK = 
        (PLASMA_Complex64_t *) malloc(sizeof(PLASMA_Complex64_t) * lwork);
    int ldwork = nb;

    int i, i1, i3;
    int nq, nw;
    int kb;
    int ic = 0;
    int jc = 0;
    int mi = m1;
    int ni = n1;

    // Check input arguments
    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        plasma_error("Illegal value of side");
        return;
    }

    // nq is the order of Q 
    if (side == PlasmaLeft) {
        nq = m2;
        nw = ib;
    }
    else {
        nq = n2;
        nw = m1;
    }

    // Plasma_ConjTrans will be converted to PlasmaTrans in 
    // automatic datatype conversion, which is what we want here.
    // PlasmaConjTrans is protected from this conversion.
    if ((trans != PlasmaNoTrans) && (trans != Plasma_ConjTrans)) {
        plasma_error("Illegal value of trans");
        return;
    }
    if (m1 < 0) {
        plasma_error("Illegal value of m1");
        return;
    }
    if (n1 < 0) {
        plasma_error("Illegal value of n1");
        return;
    }
    if ( (m2 < 0) ||
         ( (m2 != m1) && (side == PlasmaRight) ) ){
        plasma_error("Illegal value of m2");
        return;
    }
    if ( (n2 < 0) ||
         ( (n2 != n1) && (side == PlasmaLeft) ) ){
        plasma_error("Illegal value of n2");
        return;
    }
    if ((k < 0) ||
        ( (side == PlasmaLeft)  && (k > m1) ) ||
        ( (side == PlasmaRight) && (k > n1) ) ) {
        plasma_error("Illegal value of k");
        return;
    }
    if (ib < 0) {
        plasma_error("Illegal value of ib");
        return;
    }
    if (lda1 < imax(1,m1)){
        plasma_error("Illegal value of lda1");
        return;
    }
    if (lda2 < imax(1,m2)){
        plasma_error("Illegal value of lda2");
        return;
    }
    if (ldv < imax(1,nq)){
        plasma_error("Illegal value of ldv");
        return;
    }
    if (ldt < imax(1,ib)){
        plasma_error("Illegal value of ldt");
        return;
    }
    if (ldwork < imax(1,nw)){
        plasma_error("Illegal value of ldwork");
        return;
    }

    // Quick return
    if ((m1 == 0) || (n1 == 0) || (m2 == 0) || 
        (n2 == 0) || (k == 0) || (ib == 0))
        return;

    if (((side == PlasmaLeft)  && (trans != PlasmaNoTrans))
        || ((side == PlasmaRight) && (trans == PlasmaNoTrans))) {
        i1 = 0;
        i3 = ib;
    }
    else {
        i1 = ((k-1) / ib)*ib;
        i3 = -ib;
    }

    for(i = i1; (i > -1) && (i < k); i += i3) {
        kb = imin(ib, k-i);

        if (side == PlasmaLeft) {
            // H or H' is applied to C(i:m,1:n)
            mi = m1 - i;
            ic = i;
        }
        else {
            // H or H' is applied to C(1:m,i:n)
            ni = n1 - i;
            jc = i;
        }
        // Apply H or H' (NOTE: CORE_zparfb used to be CORE_ztsrfb)
        CORE_zparfb(
            side, trans, PlasmaForward, PlasmaColumnwise,
            mi, ni, m2, n2, kb, 0,
            &A1[lda1*jc+ic], lda1,
            A2, lda2,
            &V[ldv*i], ldv,
            &T[ldt*i], ldt,
            WORK, ldwork);
    }

    // deallocate auxiliary array
    free(WORK);
}

/******************************************************************************/
void CORE_OMP_ztsmqr(PLASMA_enum side, PLASMA_enum trans,
                     int m1, int n1, int m2, int n2, int k, int ib, int nb,
                     PLASMA_Complex64_t *A1, int lda1,
                     PLASMA_Complex64_t *A2, int lda2,
                     const PLASMA_Complex64_t *V, int ldv,
                     const PLASMA_Complex64_t *T, int ldt)
{
#pragma omp task depend(inout:A1[0:nb*nb]) \
                 depend(inout:A2[0:nb*nb]) \
                 depend(in:V[0:nb*nb]) \
                 depend(in:T[0:ib*nb])
    CORE_ztsmqr(side, trans,
                m1, n1, m2, n2, k, ib, nb,
                A1, lda1,
                A2, lda2,
                V, ldv,
                T, ldt);
}
