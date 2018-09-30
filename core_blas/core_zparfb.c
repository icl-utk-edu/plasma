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

/***************************************************************************//**
 *
 * @ingroup core_parfb
 *
 *  Applies an upper triangular block reflector H
 *  or its transpose H^H to a rectangular matrix formed by
 *  coupling two tiles A1 and A2. Matrix V is:
 *
 *          COLUMNWISE                    ROWWISE
 *
 *         |     K     |                 |      N2-L     |   L  |
 *      __ _____________ __           __ _________________        __
 *         |    |      |                 |               | \
 *         |    |      |                 |               |   \    L
 *    M2-L |    |      |              K  |_______________|_____\  __
 *         |    |      | M2              |                      |
 *      __ |____|      |                 |                      | K-L
 *         \    |      |              __ |______________________| __
 *       L   \  |      |
 *      __     \|______| __              |          N2          |
 *
 *         | L |  K-L  |
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
 * @param[in] direct
 *         Indicates how H is formed from a product of elementary
 *         reflectors
 *         - PlasmaForward  : H = H(1) H(2) . . . H(k) (Forward)
 *         - PlasmaBackward : H = H(k) . . . H(2) H(1) (Backward)
 *
 * @param[in] storev
 *         Indicates how the vectors which define the elementary
 *         reflectors are stored:
 *         - PlasmaColumnwise
 *         - PlasmaRowwise
 *
 * @param[in] m1
 *         The number of columns of the tile A1. m1 >= 0.
 *
 * @param[in] n1
 *         The number of rows of the tile A1. n1 >= 0.
 *
 * @param[in] m2
 *         The number of columns of the tile A2. m2 >= 0.
 *
 * @param[in] n2
 *         The number of rows of the tile A2. n2 >= 0.
 *
 * @param[in] k
 *         The order of the matrix T (= the number of elementary
 *         reflectors whose product defines the block reflector).
 *
 * @param[in] l
 *         The size of the triangular part of V
 *
 * @param[in,out] A1
 *         On entry, the m1-by-n1 tile A1.
 *         On exit, A1 is overwritten by the application of Q.
 *
 * @param[in] lda1
 *         The leading dimension of the array A1. lda1 >= max(1,n1).
 *
 * @param[in,out] A2
 *         On entry, the m2-by-n2 tile A2.
 *         On exit, A2 is overwritten by the application of Q.
 *
 * @param[in] lda2
 *         The leading dimension of the tile A2. lda2 >= max(1,n2).
 *
 * @param[in] V
 *         (ldv,k)  if storev = 'C'
 *         (ldv,m2) if storev = 'R' and side = 'L'
 *         (ldv,n2) if storev = 'R' and side = 'R'
 *         Matrix V.
 *
 * @param[in] ldv
 *         The leading dimension of the array V.
 *         If storev = 'C' and side = 'L', ldv >= max(1,m2);
 *         if storev = 'C' and side = 'R', ldv >= max(1,n2);
 *         if storev = 'R', ldv >= k.
 *
 * @param[out] T
 *         The triangular k-by-k matrix T in the representation of the
 *         block reflector.
 *         T is upper triangular by block (economic storage);
 *         The rest of the array is not referenced.
 *
 * @param[in] ldt
 *         The leading dimension of the array T. ldt >= k.
 *
 * @param[in,out] work
 *
 * @param[in] ldwork
 *         The leading dimension of the array work.
 *
 *******************************************************************************
 *
 * @retval PlasmaSuccess successful exit
 * @retval < 0 if -i, the i-th argument had an illegal value
 *
 ******************************************************************************/
__attribute__((weak))
int plasma_core_zparfb(plasma_enum_t side, plasma_enum_t trans,
                plasma_enum_t direct, plasma_enum_t storev,
                int m1, int n1, int m2, int n2, int k, int l,
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
    if (direct != PlasmaForward && direct != PlasmaBackward) {
        plasma_coreblas_error("illegal value of direct");
        return -3;
    }
    if (storev != PlasmaColumnwise && storev != PlasmaRowwise) {
        plasma_coreblas_error("illegal value of storev");
        return -4;
    }
    if (m1 < 0) {
        plasma_coreblas_error("illegal value of m1");
        return -5;
    }
    if (n1 < 0) {
        plasma_coreblas_error("illegal value of n1");
        return -6;
    }
    if (m2 < 0 || (side == PlasmaRight && m1 != m2)) {
        plasma_coreblas_error("illegal value of m2");
        return -7;
    }
    if (n2 < 0 ||
        (side == PlasmaLeft && n1 != n2)) {
        plasma_coreblas_error("illegal value of n2");
        return -8;
    }
    if (k < 0) {
        plasma_coreblas_error("illegal value of k");
        return -9;
    }
    if (l < 0) {
        plasma_coreblas_error("illegal value of l");
        return -10;
    }
    if (A1 == NULL) {
        plasma_coreblas_error("NULL A1");
        return -11;
    }
    if (lda1 < 0) {
        plasma_coreblas_error("illegal value of lda1");
        return -12;
    }
    if (A2 == NULL) {
        plasma_coreblas_error("NULL A2");
        return -13;
    }
    if (lda2 < 0) {
        plasma_coreblas_error("illegal value of lda2");
        return -14;
    }
    if (V == NULL) {
        plasma_coreblas_error("NULL V");
        return -15;
    }
    if (ldv < 0) {
        plasma_coreblas_error("illegal value of ldv");
        return -16;
    }
    if (T == NULL) {
        plasma_coreblas_error("NULL T");
        return -17;
    }
    if (ldt < 0) {
        plasma_coreblas_error("illegal value of ldt");
        return -18;
    }
    if (work == NULL) {
        plasma_coreblas_error("NULL work");
        return -19;
    }
    if (ldwork < 0) {
        plasma_coreblas_error("illegal value of ldwork");
        return -20;
    }

    // quick return
    if (m1 == 0 || n1 == 0 || m2 == 0 || n2 == 0 || k == 0)
        return PlasmaSuccess;

    plasma_complex64_t zone  =  1.0;
    plasma_complex64_t zmone = -1.0;

    if (direct == PlasmaForward) {
        //=============================
        // PlasmaForward / PlasmaLeft
        //=============================
        if (side == PlasmaLeft) {
            // Form  H * A  or  H^H * A  where  A = ( A1 )
            //                                      ( A2 )

            // W = A1 + op(V) * A2
            plasma_core_zpamm(PlasmaW, PlasmaLeft, storev,
                       k, n1, m2, l,
                       A1,   lda1,
                       A2,   lda2,
                       V,    ldv,
                       work, ldwork);

            // W = op(T) * W
            cblas_ztrmm(CblasColMajor,
                        CblasLeft, CblasUpper,
                        (CBLAS_TRANSPOSE)trans, CblasNonUnit,
                        k, n2,
                        CBLAS_SADDR(zone), T,    ldt,
                                           work, ldwork);

            // A1 = A1 - W
            for (int j = 0; j < n1; j++) {
                cblas_zaxpy(k, CBLAS_SADDR(zmone),
                            &work[ldwork*j], 1,
                            &A1[lda1*j], 1);
            }

            // A2 = A2 - op(V) * W
            // W = V * W, A2 = A2 - W
            plasma_core_zpamm(PlasmaA2, PlasmaLeft, storev,
                       m2, n2, k, l,
                       A1,   lda1,
                       A2,   lda2,
                       V,    ldv,
                       work, ldwork);
        }
        //==============================
        // PlasmaForward / PlasmaRight
        //==============================
        else {
            // Form  H * A  or  H^H * A  where A  = ( A1 A2 )

            // W = A1 + A2 * op(V)
            plasma_core_zpamm(PlasmaW, PlasmaRight, storev,
                       m1, k, n2, l,
                       A1,   lda1,
                       A2,   lda2,
                       V,    ldv,
                       work, ldwork);

            // W = W * op(T)
            cblas_ztrmm(CblasColMajor,
                        CblasRight, CblasUpper,
                        (CBLAS_TRANSPOSE)trans, CblasNonUnit,
                        m2, k,
                        CBLAS_SADDR(zone), T,    ldt,
                                           work, ldwork);

            // A1 = A1 - W
            for (int j = 0; j < k; j++) {
                cblas_zaxpy(m1, CBLAS_SADDR(zmone),
                            &work[ldwork*j], 1,
                            &A1[lda1*j], 1);
            }

            // A2 = A2 - W * op(V)
            // W = W * V^H, A2 = A2 - W
            plasma_core_zpamm(PlasmaA2, PlasmaRight, storev,
                       m2, n2, k, l,
                       A1,   lda1,
                       A2,   lda2,
                       V,    ldv,
                       work, ldwork);
        }
    }
    else {
        plasma_coreblas_error("Backward / Left or Right not implemented");
        return PlasmaErrorNotSupported;
    }

    return PlasmaSuccess;
}
