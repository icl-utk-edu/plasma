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

#include <plasma_core_blas.h>
#include "plasma_types.h"
#include "plasma_internal.h"
#include "core_lapack.h"

#include <omp.h>

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
 *  as returned by plasma_core_ztsqrt.
 *
 *******************************************************************************
 *
 * @param[in] side
 *         - PlasmaLeft  : apply Q or Q^H from the Left;
 *         - PlasmaRight :  apply Q or Q^H from the Right.
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
 *         plasma_core_ZTSQRT in the first k columns of its array argument V.
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
 * @param work
 *         Auxiliary workspace array of length
 *         ldwork-by-n1 if side == PlasmaLeft
 *         ldwork-by-ib if side == PlasmaRight
 *
 * @param[in] ldwork
 *         The leading dimension of the array work.
 *             ldwork >= max(1,ib) if side == PlasmaLeft
 *             ldwork >= max(1,m1) if side == PlasmaRight
 *
 *******************************************************************************
 *
 * @retval PlasmaSuccess successful exit
 * @retval < 0 if -i, the i-th argument had an illegal value
 *
 ******************************************************************************/
__attribute__((weak))
int plasma_core_ztsmqr(plasma_enum_t side, plasma_enum_t trans,
                int m1, int n1, int m2, int n2, int k, int ib,
                      plasma_complex64_t *A1,   int lda1,
                      plasma_complex64_t *A2,   int lda2,
                const plasma_complex64_t *V,    int ldv,
                const plasma_complex64_t *T,    int ldt,
                      plasma_complex64_t *work, int ldwork)
{
    // Check input arguments.
    if (side != PlasmaLeft && side != PlasmaRight) {
        plasma_coreblas_error("illegal value of side");
        return -1;
    }
    if (trans != PlasmaNoTrans && trans != Plasma_ConjTrans) {
        plasma_coreblas_error("illegal value of trans");
        return -2;
    }
    if (m1 < 0) {
        plasma_coreblas_error("illegal value of m1");
        return -3;
    }
    if (n1 < 0) {
        plasma_coreblas_error("illegal value of n1");
        return -4;
    }
    if (m2 < 0 || (m2 != m1 && side == PlasmaRight)) {
        plasma_coreblas_error("illegal value of m2");
        return -5;
    }
    if (n2 < 0 || (n2 != n1 && side == PlasmaLeft)) {
        plasma_coreblas_error("illegal value of n2");
        return -6;
    }
    if (k < 0 ||
        (side == PlasmaLeft  && k > m1) ||
        (side == PlasmaRight && k > n1)) {
        plasma_coreblas_error("illegal value of k");
        return -7;
    }
    if (ib < 0) {
        plasma_coreblas_error("illegal value of ib");
        return -8;
    }
    if (A1 == NULL) {
        plasma_coreblas_error("NULL A1");
        return -9;
    }
    if (lda1 < imax(1, m1)) {
        plasma_coreblas_error("illegal value of lda1");
        return -10;
    }
    if (A2 == NULL) {
        plasma_coreblas_error("NULL A2");
        return -11;
    }
    if (lda2 < imax(1, m2)) {
        plasma_coreblas_error("illegal value of lda2");
        return -12;
    }
    if (V == NULL) {
        plasma_coreblas_error("NULL V");
        return -13;
    }
    if (ldv < imax(1, side == PlasmaLeft ? m2 : n2)) {
        plasma_coreblas_error("illegal value of ldv");
        return -14;
    }
    if (T == NULL) {
        plasma_coreblas_error("NULL T");
        return -15;
    }
    if (ldt < imax(1, ib)) {
        plasma_coreblas_error("illegal value of ldt");
        return -16;
    }
    if (work == NULL) {
        plasma_coreblas_error("NULL work");
        return -17;
    }
    if (ldwork < imax(1, side == PlasmaLeft ? ib : m1)) {
        plasma_coreblas_error("illegal value of ldwork");
        return -18;
    }

    // quick return
    if (m1 == 0 || n1 == 0 || m2 == 0 || n2 == 0 || k == 0 || ib == 0)
        return PlasmaSuccess;

    int i1, i3;

    if ((side == PlasmaLeft  && trans != PlasmaNoTrans) ||
        (side == PlasmaRight && trans == PlasmaNoTrans)) {
        i1 = 0;
        i3 = ib;
    }
    else {
        i1 = ((k-1)/ib)*ib;
        i3 = -ib;
    }

    for (int i = i1; i > -1 && i < k; i += i3) {
        int kb = imin(ib, k-i);
        int ic = 0;
        int jc = 0;
        int mi = m1;
        int ni = n1;

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

        // Apply H or H^H (NOTE: plasma_core_zparfb used to be core_ztsrfb).
        plasma_core_zparfb(side, trans, PlasmaForward, PlasmaColumnwise,
                    mi, ni, m2, n2, kb, 0,
                    &A1[lda1*jc+ic], lda1,
                    A2, lda2,
                    &V[ldv*i], ldv,
                    &T[ldt*i], ldt,
                    work, ldwork);
    }

    return PlasmaSuccess;
}

/******************************************************************************/
void plasma_core_omp_ztsmqr(plasma_enum_t side, plasma_enum_t trans,
                     int m1, int n1, int m2, int n2, int k, int ib,
                           plasma_complex64_t *A1, int lda1,
                           plasma_complex64_t *A2, int lda2,
                     const plasma_complex64_t *V,  int ldv,
                     const plasma_complex64_t *T,  int ldt,
                     plasma_workspace_t work,
                     plasma_sequence_t *sequence, plasma_request_t *request)
{
    #pragma omp task depend(inout:A1[0:lda1*n1]) \
                     depend(inout:A2[0:lda2*n2]) \
                     depend(in:V[0:ldv*k]) \
                     depend(in:T[0:ib*k])
    {
        if (sequence->status == PlasmaSuccess) {
            // Prepare workspaces.
            int tid = omp_get_thread_num();
            plasma_complex64_t *W = (plasma_complex64_t*)work.spaces[tid];
            int ldwork = side == PlasmaLeft ? ib : m1; // TODO: double check

            // Call the kernel.
            int info = plasma_core_ztsmqr(side, trans,
                                   m1, n1, m2, n2, k, ib,
                                   A1, lda1,
                                   A2, lda2,
                                   V,  ldv,
                                   T,  ldt,
                                   W,  ldwork);

            if (info != PlasmaSuccess) {
                plasma_error("core_ztsmqr() failed");
                plasma_request_fail(sequence, request, PlasmaErrorInternal);
            }
        }
    }
}
