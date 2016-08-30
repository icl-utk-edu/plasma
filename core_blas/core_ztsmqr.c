/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
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
 * @ingroup core_tsmqr
 *
 *  Overwrites the general m1-by-n1 tile A1 and
 *  m2-by-n2 tile A2 with
 *
 *                                side = PlasmaLeft        side = PlasmaRight
 *    trans = PlasmaNoTrans            Q * | A1 |           | A1 A2 | * Q
 *                                         | A2 |
 *
 *    trans = Plasma_ConjTrans       Q^H * | A1 |           | A1 A2 | * Q^H
 *                                         | A2 |
 *
 *  where Q is a complex unitary matrix defined as the product of k
 *  elementary reflectors
 *
 *    Q = H(1) H(2) . . . H(k)
 *
 *  as returned by CORE_ztsqrt.
 *
 *******************************************************************************
 *
 * @param[in] side
 *         - PlasmaLeft  : apply Q or Q^H from the Left;
 *         - PlasmaRight : apply Q or Q^H from the Right.
 *
 * @param[in] trans
 *         - PlasmaNoTrans    : Apply Q;
 *         - Plasma_ConjTrans : Apply Q^H.
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
 * @param WORK
 *         Auxiliary workspace array of length
 *         ldwork-by-n1 if side == PlasmaLeft
 *         ldwork-by-ib if side == PlasmaRight
 *
 * @param[in] ldwork
 *         The leading dimension of the array WORK.
 *             ldwork >= max(1,ib) if side == PlasmaLeft
 *             ldwork >= max(1,m1) if side == PlasmaRight
 *
 ******************************************************************************/
void CORE_ztsmqr(PLASMA_enum side, PLASMA_enum trans,
                 int m1, int n1, int m2, int n2, int k, int ib,
                       PLASMA_Complex64_t *A1,   int lda1,
                       PLASMA_Complex64_t *A2,   int lda2,
                 const PLASMA_Complex64_t *V,    int ldv,
                 const PLASMA_Complex64_t *T,    int ldt,
                       PLASMA_Complex64_t *WORK, int ldwork)
{
    int i, i1, i3;
    int nq, nw;
    int kb;
    int ic = 0;
    int jc = 0;
    int mi = m1;
    int ni = n1;

    // Check input arguments.
    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        plasma_error("illegal value of side");
        return;
    }

    // nq is the order of Q.
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
        plasma_error("illegal value of trans");
        return;
    }
    if (m1 < 0) {
        plasma_error("illegal value of m1");
        return;
    }
    if (n1 < 0) {
        plasma_error("illegal value of n1");
        return;
    }
    if ((m2 < 0) ||
         ((m2 != m1) && (side == PlasmaRight))) {
        plasma_error("illegal value of m2");
        return;
    }
    if ((n2 < 0) ||
         ((n2 != n1) && (side == PlasmaLeft))) {
        plasma_error("illegal value of n2");
        return;
    }
    if ((k < 0) ||
        ((side == PlasmaLeft)  && (k > m1)) ||
        ((side == PlasmaRight) && (k > n1))) {
        plasma_error("illegal value of k");
        return;
    }
    if (ib < 0) {
        plasma_error("illegal value of ib");
        return;
    }
    if (lda1 < imax(1,m1)) {
        plasma_error("illegal value of lda1");
        return;
    }
    if (lda2 < imax(1,m2)) {
        plasma_error("illegal value of lda2");
        return;
    }
    if (ldv < imax(1,nq)) {
        plasma_error("illegal value of ldv");
        return;
    }
    if (ldt < imax(1,ib)) {
        plasma_error("illegal value of ldt");
        return;
    }
    if (ldwork < imax(1,nw)) {
        plasma_error("illegal value of ldwork");
        return;
    }

    // quick return
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

    for (i = i1; (i > -1) && (i < k); i += i3) {
        kb = imin(ib, k-i);

        if (side == PlasmaLeft) {
            // H or H^H is applied to C(i:m,1:n).
            mi = m1 - i;
            ic = i;
        }
        else {
            // H or H^H is applied to C(1:m,i:n).
            ni = n1 - i;
            jc = i;
        }
        // Apply H or H^H (NOTE: CORE_zparfb used to be CORE_ztsrfb).
        CORE_zparfb(side, trans, PlasmaForward, PlasmaColumnwise,
                    mi, ni, m2, n2, kb, 0,
                    &A1[lda1*jc+ic], lda1,
                    A2, lda2,
                    &V[ldv*i], ldv,
                    &T[ldt*i], ldt,
                    WORK, ldwork);
    }
}

/******************************************************************************/
void CORE_OMP_ztsmqr(PLASMA_enum side, PLASMA_enum trans,
                     int m1, int n1, int m2, int n2, int k, int ib, int nb,
                           PLASMA_Complex64_t *A1, int lda1,
                           PLASMA_Complex64_t *A2, int lda2,
                     const PLASMA_Complex64_t *V, int ldv,
                     const PLASMA_Complex64_t *T, int ldt)
{
    // omp depends assume m1 == nb, n1 == nb, m2 == nb, n2 == nb.
    #pragma omp task depend(inout:A1[0:nb*nb]) \
                     depend(inout:A2[0:nb*nb]) \
                     depend(in:V[0:nb*nb]) \
                     depend(in:T[0:ib*nb])
    {
        // Allocate an auxiliary array.
        PLASMA_Complex64_t *WORK =
            (PLASMA_Complex64_t *) malloc((size_t)ib*nb *
                                          sizeof(PLASMA_Complex64_t));
        if (WORK == NULL) {
            plasma_error("malloc() failed");
        }

        int ldwork = side == PlasmaLeft ? ib : nb;

        // Call the kernel.
        CORE_ztsmqr(side, trans,
                    m1, n1, m2, n2, k, ib,
                    A1, lda1,
                    A2, lda2,
                    V, ldv,
                    T, ldt,
                    WORK, ldwork);

        // Free the auxiliary array.
        free(WORK);
    }
}
