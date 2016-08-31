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

static inline void CORE_zpamm_a2(PLASMA_enum side, PLASMA_enum trans,
                                 PLASMA_enum uplo,
                                 int m, int n, int k, int l,
                                 int vi2, int vi3,
                                       PLASMA_Complex64_t *A2, int lda2,
                                 const PLASMA_Complex64_t *V,  int ldv,
                                       PLASMA_Complex64_t *W,  int ldw);

static inline void CORE_zpamm_w(PLASMA_enum side, PLASMA_enum trans,
                                PLASMA_enum uplo,
                                int m, int n, int k, int l,
                                int vi2, int vi3,
                                const PLASMA_Complex64_t *A1, int lda1,
                                      PLASMA_Complex64_t *A2, int lda2,
                                const PLASMA_Complex64_t *V,  int ldv,
                                      PLASMA_Complex64_t *W,  int ldw);

/***************************************************************************//**
 *
 * @ingroup core_pamm
 *
 *  Performs one of the matrix-matrix operations
 *
 *                    PlasmaLeft                PlasmaRight
 *     OP PlasmaW  :  W  = A1 + op(V) * A2  or  W  = A1 + A2 * op(V)
 *     OP PlasmaA2 :  A2 = A2 - op(V) * W   or  A2 = A2 - W * op(V)
 *
 *  where  op( V ) is one of
 *
 *     op( V ) = V   or   op( V ) = V^H
 *
 *  A1, A2 and W are general matrices, and V is:
 *
 *        l = k: rectangle + triangle
 *        l < k: rectangle + trapezoid
 *        l = 0: rectangle
 *
 *  Size of V, both rowwise and columnwise, is:
 *
 *         ----------------------
 *          side   trans    size
 *         ----------------------
 *          left     N     M x K
 *                   T     K x M
 *          right    N     K x N
 *                   T     N x K
 *         ----------------------
 *
 *  PlasmaLeft (columnwise and rowwise):
 *
 *              |    K    |                 |         M         |
 *           _  __________   _              _______________        _
 *              |    |    |                 |             | \
 *     V:       |    |    |           V^H:  |_____________|___\    K
 *              |    |    | M-L             |                  |
 *           M  |    |    |                 |__________________|   _
 *              |____|    |  _
 *              \    |    |                 |    M - L    | L  |
 *                \  |    |  L
 *           _      \|____|  _
 *
 *
 *  PlasmaRight (columnwise and rowwise):
 *
 *          |         K         |                   |    N    |
 *          _______________        _             _  __________   _
 *          |             | \                       |    |    |
 *    V^H:  |_____________|___\    N        V:      |    |    |
 *          |                  |                    |    |    | K-L
 *          |__________________|   _             K  |    |    |
 *                                                  |____|    |  _
 *          |    K - L    | L  |                    \    |    |
 *                                                    \  |    |  L
 *                                               _      \|____|  _
 *
 *  Arguments
 *  ==========
 *
 * @param[in] op
 *
 *         OP specifies which operation to perform:
 *
 *         - PlasmaW  : W  = A1 + op(V) * A2  or  W  = A1 + A2 * op(V)
 *         - PlasmaA2 : A2 = A2 - op(V) * W   or  A2 = A2 - W * op(V)
 *
 * @param[in] side
 *
 *         SIDE specifies whether  op( V ) multiplies A2
 *         or W from the left or right as follows:
 *
 *         - PlasmaLeft  : multiply op( V ) from the left
 *                            OP PlasmaW  :  W  = A1 + op(V) * A2
 *                            OP PlasmaA2 :  A2 = A2 - op(V) * W
 *
 *         - PlasmaRight : multiply op( V ) from the right
 *                            OP PlasmaW  :  W  = A1 + A2 * op(V)
 *                            OP PlasmaA2 :  A2 = A2 - W * op(V)
 *
 * @param[in] storev
 *
 *         Indicates how the vectors which define the elementary
 *         reflectors are stored in V:
 *
 *         - PlasmaColumnwise
 *         - PlasmaRowwise
 *
 * @param[in] m
 *         The number of rows of the A1, A2 and W
 *         If SIDE is PlasmaLeft, the number of rows of op( V )
 *
 * @param[in] n
 *         The number of columns of the A1, A2 and W
 *         If SIDE is PlasmaRight, the number of columns of op( V )
 *
 * @param[in] k
 *         If SIDE is PlasmaLeft, the number of columns of op( V )
 *         If SIDE is PlasmaRight, the number of rows of op( V )
 *
 * @param[in] l
 *         The size of the triangular part of V
 *
 * @param[in] A1
 *         On entry, the m-by-n tile A1.
 *
 * @param[in] lda1
 *         The leading dimension of the array A1. lda1 >= max(1,m).
 *
 * @param[in,out] A2
 *         On entry, the m-by-n tile A2.
 *         On exit, if OP is PlasmaA2 A2 is overwritten
 *
 * @param[in] lda2
 *         The leading dimension of the tile A2. lda2 >= max(1,m).
 *
 * @param[in] V
 *         The matrix V as described above.
 *         If SIDE is PlasmaLeft : op( V ) is m-by-k
 *         If SIDE is PlasmaRight: op( V ) is k-by-n
 *
 * @param[in] ldv
 *         The leading dimension of the array V.
 *
 * @param[in,out] W
 *         On entry, the m-by-n matrix W.
 *         On exit, W is overwritten either if OP is PlasmaA2 or PlasmaW.
 *         If OP is PlasmaA2, W is an input and is used as a workspace.
 *
 * @param[in] ldw
 *         The leading dimension of array W.
 *
 ******************************************************************************/
void CORE_zpamm(int op, PLASMA_enum side, PLASMA_enum storev,
                int m, int n, int k, int l,
                const PLASMA_Complex64_t *A1, int lda1,
                      PLASMA_Complex64_t *A2, int lda2,
                const PLASMA_Complex64_t *V,  int ldv,
                      PLASMA_Complex64_t *W,  int ldw)
{
    int vi2, vi3, uplo, trans;

    // Check input arguments.
    if ((op != PlasmaW) && (op != PlasmaA2)) {
        plasma_error("illegal value of op");
        return;
    }
    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        plasma_error("illegal value of side");
        return;
    }
    if ((storev != PlasmaColumnwise) && (storev != PlasmaRowwise)) {
        plasma_error("illegal value of storev");
        return;
    }
    if (m < 0) {
        plasma_error("illegal value of m");
        return;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return;
    }
    if (k < 0) {
        plasma_error("illegal value of k");
        return;
    }
    if (l < 0) {
        plasma_error("illegal value of l");
        return;
    }
    if (lda1 < 0) {
        plasma_error("illegal value of lda1");
        return;
    }
    if (lda2 < 0) {
        plasma_error("illegal value of lda2");
        return;
    }
    if (ldv < 0) {
        plasma_error("illegal value of ldv");
        return;
    }
    if (ldw < 0) {
        plasma_error("illegal value of ldw");
        return;
    }

    // quick return
    if ((m == 0) || (n == 0) || (k == 0))
        return;

    // TRANS is set as:
    //
    //        -------------------------------------
    //         side   direct     PlasmaW  PlasmaA2
    //        -------------------------------------
    //         left   colwise       T        N
    //                rowwise       N        T
    //         right  colwise       N        T
    //                rowwise       T        N
    //        -------------------------------------

    // columnwise
    if (storev == PlasmaColumnwise) {
        uplo = CblasUpper;
        if (side == PlasmaLeft) {
            trans = op == PlasmaA2 ? PlasmaNoTrans : Plasma_ConjTrans;
            vi2 = trans == PlasmaNoTrans ? m - l : k - l;
        }
        else {
            trans = op == PlasmaW ? PlasmaNoTrans : Plasma_ConjTrans;
            vi2 = trans == PlasmaNoTrans ? k - l : n - l;
        }
        vi3 = ldv * l;
    }

    // rowwise
    else {
        uplo = CblasLower;
        if (side == PlasmaLeft) {
            trans = op == PlasmaW ? PlasmaNoTrans : Plasma_ConjTrans;
            vi2 = trans == PlasmaNoTrans ? k - l : m - l;
        }
        else {
            trans = op == PlasmaA2 ? PlasmaNoTrans : Plasma_ConjTrans;
            vi2 = trans == PlasmaNoTrans ? n - l : k - l;
        }
        vi2 *= ldv;
        vi3  = l;
    }

    if (op == PlasmaW) {
        CORE_zpamm_w(side, trans, uplo, m, n, k, l, vi2, vi3,
                     A1, lda1, A2, lda2, V, ldv, W, ldw);
    }
    else if (op == PlasmaA2) {
        CORE_zpamm_a2(side, trans, uplo, m, n, k, l, vi2, vi3,
                      A2, lda2, V, ldv, W, ldw);
    }
}

/******************************************************************************/
static inline void CORE_zpamm_w(
        PLASMA_enum side, PLASMA_enum trans, PLASMA_enum uplo,
        int m, int n, int k, int l,
        int vi2, int vi3,
        const PLASMA_Complex64_t *A1, int lda1,
              PLASMA_Complex64_t *A2, int lda2,
        const PLASMA_Complex64_t *V,  int ldv,
              PLASMA_Complex64_t *W,  int ldw)
{
    // W = A1 + op(V) * A2  or  W = A1 + A2 * op(V)

    int j;
    static PLASMA_Complex64_t zone  =  1.0;
    static PLASMA_Complex64_t zzero =  0.0;

    if (side == PlasmaLeft) {
        if (((trans == Plasma_ConjTrans) && (uplo == CblasUpper)) ||
            ((trans == PlasmaNoTrans) && (uplo == CblasLower))) {
            // W = A1 + V^H * A2

            // W = A2_2
            LAPACKE_zlacpy_work(LAPACK_COL_MAJOR,
                                lapack_const(PlasmaFull),
                                l, n,
                                &A2[k-l], lda2,
                                W, ldw);

            // W = V_2^H * W + V_1^H * A2_1 (ge+tr, top L rows of V^H)
            if (l > 0) {
                // W = V_2^H * W
                cblas_ztrmm(CblasColMajor,
                            CblasLeft, (CBLAS_UPLO)uplo,
                            (CBLAS_TRANSPOSE)trans, CblasNonUnit,
                            l, n,
                            CBLAS_SADDR(zone), &V[vi2], ldv,
                            W, ldw);

                // W = W + V_1^H * A2_1
                if (k > l) {
                    cblas_zgemm(CblasColMajor,
                                (CBLAS_TRANSPOSE)trans, CblasNoTrans,
                                l, n, k-l,
                                CBLAS_SADDR(zone), V, ldv,
                                A2, lda2,
                                CBLAS_SADDR(zone), W, ldw);
                }
            }

            // W_2 = V_3^H * A2: (ge, bottom M-L rows of V^H)
            if (m > l) {
                cblas_zgemm(CblasColMajor,
                            (CBLAS_TRANSPOSE)trans, CblasNoTrans,
                            (m-l), n, k,
                            CBLAS_SADDR(zone), &V[vi3], ldv,
                            A2, lda2,
                            CBLAS_SADDR(zzero), &W[l], ldw);
            }

            // W = A1 + W
            for (j = 0; j < n; j++) {
                cblas_zaxpy(m, CBLAS_SADDR(zone),
                            &A1[lda1*j], 1,
                            &W[ldw*j], 1);
            }
        }
        else {
            plasma_error(
                "Left Upper/NoTrans & Lower/[Conj]Trans not implemented");
            return;
        }
    }
    else { //side right
        if (((trans == Plasma_ConjTrans) && (uplo == CblasUpper)) ||
            ((trans == PlasmaNoTrans) && (uplo == CblasLower))) {
            plasma_error(
                "Right Upper/[Conj]Trans & Lower/NoTrans not implemented");
            return;
        }
        else {
            // W = A1 + A2 * V
            if (l > 0) {
                // W = A2_2
                LAPACKE_zlacpy_work(LAPACK_COL_MAJOR,
                                    lapack_const(PlasmaFull),
                                    m, l,
                                    &A2[lda2*(k-l)], lda2,
                                    W, ldw);

                // W = W * V_2 --> W = A2_2 * V_2
                cblas_ztrmm(CblasColMajor,
                            CblasRight, (CBLAS_UPLO)uplo,
                            (CBLAS_TRANSPOSE)trans, CblasNonUnit,
                            m, l,
                            CBLAS_SADDR(zone), &V[vi2], ldv,
                            W, ldw);

                // W = W + A2_1 * V_1
                if (k > l) {
                    cblas_zgemm(CblasColMajor,
                                CblasNoTrans, (CBLAS_TRANSPOSE)trans,
                                m, l, k-l,
                                CBLAS_SADDR(zone), A2, lda2,
                                V, ldv,
                                CBLAS_SADDR(zone), W, ldw);
                }
            }

            // W = W + A2 * V_3
            if (n > l) {
                cblas_zgemm(CblasColMajor,
                            CblasNoTrans, (CBLAS_TRANSPOSE)trans,
                            m, n-l, k,
                            CBLAS_SADDR(zone), A2, lda2,
                            &V[vi3], ldv,
                            CBLAS_SADDR(zzero), &W[ldw*l], ldw);
            }

            // W = A1 + W
            for (j = 0; j < n; j++) {
                cblas_zaxpy(m, CBLAS_SADDR(zone),
                            &A1[lda1*j], 1,
                            &W[ldw*j], 1);
            }
        }
    }
}

/******************************************************************************/
static inline void CORE_zpamm_a2(
        PLASMA_enum side, PLASMA_enum trans, PLASMA_enum uplo,
        int m, int n, int k, int l,
        int vi2, int vi3,
              PLASMA_Complex64_t *A2, int lda2,
        const PLASMA_Complex64_t *V,  int ldv,
              PLASMA_Complex64_t *W,  int ldw)
{
   // A2 = A2 + op(V) * W  or  A2 = A2 + W * op(V)

    int j;
    static PLASMA_Complex64_t zone  =  1.0;
    static PLASMA_Complex64_t mzone  =  -1.0;

    if (side == PlasmaLeft) {
        if (((trans == Plasma_ConjTrans) && (uplo == CblasUpper)) ||
            ((trans == PlasmaNoTrans) && (uplo == CblasLower))) {
            plasma_error(
                "Left Upper/[Conj]Trans & Lower/NoTrans not implemented");
            return;
        }
        else {  //trans
            // A2 = A2 - V * W

            // A2_1 = A2_1 - V_1  * W_1
            if (m > l) {
                cblas_zgemm(CblasColMajor,
                            (CBLAS_TRANSPOSE)trans, CblasNoTrans,
                            m-l, n, l,
                            CBLAS_SADDR(mzone), V, ldv,
                            W, ldw,
                            CBLAS_SADDR(zone), A2, lda2);
            }

            // W_1 = V_2 * W_1
            cblas_ztrmm(CblasColMajor,
                        CblasLeft, (CBLAS_UPLO)uplo,
                        (CBLAS_TRANSPOSE)trans, CblasNonUnit,
                        l, n,
                        CBLAS_SADDR(zone), &V[vi2], ldv,
                        W, ldw);

            // A2_2 = A2_2 - W_1
            for (j = 0; j < n; j++) {
                cblas_zaxpy(l, CBLAS_SADDR(mzone),
                            &W[ldw*j], 1,
                            &A2[lda2*j+(m-l)], 1);
            }

            // A2 = A2 - V_3  * W_2
            if (k > l) {
                cblas_zgemm(CblasColMajor,
                            (CBLAS_TRANSPOSE)trans, CblasNoTrans,
                            m, n, (k-l),
                            CBLAS_SADDR(mzone), &V[vi3], ldv,
                            &W[l], ldw,
                            CBLAS_SADDR(zone), A2, lda2);
            }
        }
    }
    else { //side right
        if (((trans == Plasma_ConjTrans) && (uplo == CblasUpper)) ||
            ((trans == PlasmaNoTrans) && (uplo == CblasLower))) {
            // A2 = A2 - W * V^H

            // A2 = A2 - W_2 * V_3^H
            if (k > l) {
                cblas_zgemm(CblasColMajor,
                            CblasNoTrans, (CBLAS_TRANSPOSE)trans,
                            m, n, k-l,
                            CBLAS_SADDR(mzone), &W[ldw*l], ldw,
                            &V[vi3], ldv,
                            CBLAS_SADDR(zone), A2, lda2);
            }

            // A2_1 = A2_1 - W_1 * V_1^H
            if (n > l) {
                cblas_zgemm(CblasColMajor,
                            CblasNoTrans, (CBLAS_TRANSPOSE)trans,
                            m, n-l, l,
                            CBLAS_SADDR(mzone), W, ldw,
                            V, ldv,
                            CBLAS_SADDR(zone), A2, lda2);
            }

            // A2_2 =  A2_2 -  W_1 * V_2^H
            if (l > 0) {
                cblas_ztrmm(CblasColMajor,
                            CblasRight, (CBLAS_UPLO)uplo,
                            (CBLAS_TRANSPOSE)trans, CblasNonUnit, m, l,
                            CBLAS_SADDR(mzone), &V[vi2], ldv,
                            W, ldw);

                for (j = 0; j < l; j++) {
                    cblas_zaxpy(m, CBLAS_SADDR(zone),
                                &W[ldw*j], 1,
                                &A2[lda2*(n-l+j)], 1);
                }
            }
        }
        else {
            plasma_error(
                "Right Upper/NoTrans & Lower/[Conj]Trans not implemented");
            return;
        }
    }
}
