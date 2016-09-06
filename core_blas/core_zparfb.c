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
 * @param[in,out] WORK
 *
 * @param[in] ldwork
 *         The leading dimension of the array WORK.
 *
 ******************************************************************************/
void CORE_zparfb(PLASMA_enum side, PLASMA_enum trans,
                 PLASMA_enum direct, PLASMA_enum storev,
                 int m1, int n1, int m2, int n2, int k, int l,
                       PLASMA_Complex64_t *A1, int lda1,
                       PLASMA_Complex64_t *A2, int lda2,
                 const PLASMA_Complex64_t *V,  int ldv,
                 const PLASMA_Complex64_t *T,  int ldt,
                       PLASMA_Complex64_t *WORK, int ldwork)
{
    static PLASMA_Complex64_t zone  =  1.0;
    static PLASMA_Complex64_t mzone = -1.0;

    int j;

    // Check input arguments.
    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        plasma_error("illegal value of side");
        return;
    }
    // Plasma_ConjTrans will be converted to PlasmaTrans in
    // automatic datatype conversion, which is what we want here.
    // PlasmaConjTrans is protected from this conversion.
    if ((trans != PlasmaNoTrans) && (trans != Plasma_ConjTrans)) {
        plasma_error("illegal value of trans");
        return;
    }
    if ((direct != PlasmaForward) && (direct != PlasmaBackward)) {
        plasma_error("illegal value of direct");
        return;
    }
    if ((storev != PlasmaColumnwise) && (storev != PlasmaRowwise)) {
        plasma_error("illegal value of storev");
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
        ( (side == PlasmaRight) && (m1 != m2) ) ) {
        plasma_error("illegal value of m2");
        return;
    }
    if ((n2 < 0) ||
        ( (side == PlasmaLeft) && (n1 != n2) ) ) {
        plasma_error("illegal value of n2");
        return;
    }
    if (k < 0) {
        plasma_error("illegal value of k");
        return;
    }

    // quick return
    if ((m1 == 0) || (n1 == 0) || (m2 == 0) || (n2 == 0) || (k == 0))
        return;

    if (direct == PlasmaForward) {
        if (side == PlasmaLeft) {
            // Column or Rowwise / Forward / Left
            // ----------------------------------
            //
            // Form  H * A  or  H^H * A  where  A = ( A1 )
            //                                      ( A2 )

            // W = A1 + op(V) * A2
            CORE_zpamm(PlasmaW, PlasmaLeft, storev,
                       k, n1, m2, l,
                       A1, lda1,
                       A2, lda2,
                       V,  ldv,
                       WORK, ldwork);

            // W = op(T) * W
            cblas_ztrmm(CblasColMajor,
                        CblasLeft, CblasUpper,
                        (CBLAS_TRANSPOSE)trans, CblasNonUnit,
                        k, n2,
                        CBLAS_SADDR(zone), T, ldt,
                        WORK, ldwork);

            // A1 = A1 - W
            for (j = 0; j < n1; j++) {
                cblas_zaxpy(k, CBLAS_SADDR(mzone),
                            &WORK[ldwork*j], 1,
                            &A1[lda1*j], 1);
            }

            // A2 = A2 - op(V) * W
            // W also changes: W = V * W, A2 = A2 - W
            CORE_zpamm(PlasmaA2, PlasmaLeft, storev,
                       m2, n2, k, l,
                       A1, lda1,
                       A2, lda2,
                       V,  ldv,
                       WORK, ldwork);
        }
        else {
            // Column or Rowwise / Forward / Right
            // -----------------------------------
            //
            // Form  H * A  or  H^H * A  where A  = ( A1 A2 )

            // W = A1 + A2 * op(V)
            CORE_zpamm(PlasmaW, PlasmaRight, storev,
                       m1, k, n2, l,
                       A1, lda1,
                       A2, lda2,
                       V,  ldv,
                       WORK, ldwork);

            // W = W * op(T)
            cblas_ztrmm(CblasColMajor,
                        CblasRight, CblasUpper,
                        (CBLAS_TRANSPOSE)trans, CblasNonUnit,
                        m2, k,
                        CBLAS_SADDR(zone), T, ldt,
                        WORK, ldwork);

            // A1 = A1 - W
            for (j = 0; j < k; j++) {
                cblas_zaxpy(m1, CBLAS_SADDR(mzone),
                            &WORK[ldwork*j], 1,
                            &A1[lda1*j], 1);
            }

            // A2 = A2 - W * op(V)
            // W also changes: W = W * V^H, A2 = A2 - W
            CORE_zpamm(PlasmaA2, PlasmaRight, storev,
                       m2, n2, k, l,
                       A1, lda1,
                       A2, lda2,
                       V,  ldv,
                       WORK, ldwork);
        }
    }
    else {
        plasma_error("Not implemented (Backward / Left or Right)");
        return;
    }
}
