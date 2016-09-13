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
#include "core_lapack.h"

#include <omp.h>

/***************************************************************************//**
 *
 * @ingroup core_tsmlq
 *
 *  Overwrites the general complex m1-by-n1 tile A1 and
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
 *    Q = H(k)^H . . . H(2)^H H(1)^H
 *
 *  as returned by core_ztslqt.
 *
 *******************************************************************************
 *
 * @param[in] side
 *         - PlasmaLeft  : apply Q or Q^H from the Left;
 *         - PlasmaRight : apply Q or Q^H from the Right.
 *
 * @param[in] trans
 *         @arg PlasmaNoTrans   :  Apply Q;
 *         @arg Plasma_ConjTrans:  Apply Q^H.
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
 *         core_ztslqt in the first k rows of its array argument V.
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
 *             ldwork-by-m1 if side == PlasmaLeft
 *             ldwork-by-ib if side == PlasmaRight
 *
 * @param[in] ldwork
 *         The leading dimension of the array WORK.
 *             ldwork >= max(1,ib) if side == PlasmaLeft
 *             ldwork >= max(1,n1) if side == PlasmaRight
 *
 *******************************************************************************
 *
 * @retval PLASMA_SUCCESS successful exit
 * @retval < 0 if -i, the i-th argument had an illegal value
 *
 ******************************************************************************/
int core_ztsmlq(PLASMA_enum side, PLASMA_enum trans,
                int m1, int n1, int m2, int n2, int k, int ib,
                      PLASMA_Complex64_t *A1, int lda1,
                      PLASMA_Complex64_t *A2, int lda2,
                const PLASMA_Complex64_t *V, int ldv,
                const PLASMA_Complex64_t *T, int ldt,
                      PLASMA_Complex64_t *WORK, int ldwork)
{
    int i, i1, i3;
    int nw;
    int kb;
    int ic = 0;
    int jc = 0;
    int mi = m1;
    int ni = n1;

    // Check input arguments
    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        coreblas_error("Illegal value of side");
        return -1;
    }

    // nw is the minimum dimension of WORK
    if (side == PlasmaLeft) {
        nw = ib;
    }
    else {
        nw = n1;
    }

    // Plasma_ConjTrans will be converted to PlasmaTrans in
    // automatic datatype conversion, which is what we want here.
    // PlasmaConjTrans is protected from this conversion.
    if ((trans != PlasmaNoTrans) && (trans != Plasma_ConjTrans)) {
        coreblas_error("Illegal value of trans");
        return -2;
    }
    if (m1 < 0) {
        coreblas_error("Illegal value of m1");
        return -3;
    }
    if (n1 < 0) {
        coreblas_error("Illegal value of n1");
        return -4;
    }
    if ((m2 < 0) ||
        ((m2 != m1) && (side == PlasmaRight))) {
        coreblas_error("Illegal value of m2");
        return -5;
    }
    if ((n2 < 0) ||
        ((n2 != n1) && (side == PlasmaLeft))) {
        coreblas_error("Illegal value of n2");
        return -6;
    }
    if ((k < 0) ||
        ( (side == PlasmaLeft)  && (k > m1) ) ||
        ( (side == PlasmaRight) && (k > n1) ) ) {
        coreblas_error("Illegal value of k");
        return -7;
    }
    if (ib < 0) {
        coreblas_error("Illegal value of ib");
        return -8;
    }
    if (lda1 < imax(1,m1)) {
        coreblas_error("Illegal value of lda1");
        return -10;
    }
    if (lda2 < imax(1,m2)) {
        coreblas_error("Illegal value of lda2");
        return -12;
    }
    if (ldv < imax(1,k)) {
        coreblas_error("Illegal value of ldv");
        return -14;
    }
    if (ldt < imax(1,ib)) {
        coreblas_error("Illegal value of ldt");
        return -16;
    }
    if (ldwork < imax(1,nw)) {
        coreblas_error("Illegal value of ldwork");
        return -18;
    }

    // Quick return
    if ((m1 == 0) || (n1 == 0) ||
        (m2 == 0) || (n2 == 0) ||
        (k == 0)  || (ib == 0))
        return PLASMA_SUCCESS;

    if (((side == PlasmaLeft) && (trans == PlasmaNoTrans))
        || ((side == PlasmaRight) && (trans != PlasmaNoTrans))) {
        i1 = 0;
        i3 = ib;
    }
    else {
        i1 = ((k-1) / ib)*ib;
        i3 = -ib;
    }

    if (trans == PlasmaNoTrans) {
        trans = Plasma_ConjTrans;
    }
    else {
        trans = PlasmaNoTrans;
    }

    for (i = i1; (i > -1) && (i < k); i += i3) {
        kb = imin(ib, k-i);

        if (side == PlasmaLeft) {
            // H or H^H is applied to C(i:m,1:n)
            mi = m1 - i;
            ic = i;
        }
        else {
            // H or H^H is applied to C(1:m,i:n)
            ni = n1 - i;
            jc = i;
        }
        // Apply H or H^H (NOTE: core_zparfb used to be core_ztsrfb)
        core_zparfb(
            side, trans, PlasmaForward, PlasmaRowwise,
            mi, ni, m2, n2, kb, 0,
            &A1[lda1*jc+ic], lda1,
            A2, lda2,
            &V[i], ldv,
            &T[ldt*i], ldt,
            WORK, ldwork);
    }

    return PLASMA_SUCCESS;
}

/******************************************************************************/
void core_omp_ztsmlq(PLASMA_enum side, PLASMA_enum trans,
                     int m1, int n1, int m2, int n2, int k, int ib, int nb,
                           PLASMA_Complex64_t *A1, int lda1,
                           PLASMA_Complex64_t *A2, int lda2,
                     const PLASMA_Complex64_t *V,  int ldv,
                     const PLASMA_Complex64_t *T,  int ldt,
                     PLASMA_workspace *work,
                     plasma_sequence_t *sequence, PLASMA_request *request)
{
    // assuming m1 == nb, n1 == nb, m2 == nb, n2 == nb
    #pragma omp task depend(inout:A1[0:nb*nb]) \
                     depend(inout:A2[0:nb*nb]) \
                     depend(in:V[0:nb*nb]) \
                     depend(in:T[0:ib*nb])
    {
        if (sequence->status == PLASMA_SUCCESS) {
            int tid = omp_get_thread_num();
            PLASMA_Complex64_t *W   =
                ((PLASMA_Complex64_t*)work->spaces[tid]);

            int ldwork = side == PlasmaLeft ? ib : nb;

            // call the kernel
            int info = core_ztsmlq(side, trans,
                                   m1, n1, m2, n2, k, ib,
                                   A1, lda1,
                                   A2, lda2,
                                   V, ldv,
                                   T, ldt,
                                   W, ldwork);

            if (info != PLASMA_SUCCESS) {
                plasma_error_with_code("Error in call to COREBLAS in argument",
                                       -info);
                plasma_request_fail(sequence, request,
                                    PLASMA_ERR_ILLEGAL_VALUE);
            }
        }
    }
}
