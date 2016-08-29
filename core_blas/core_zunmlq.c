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
 * @ingroup CORE_PLASMA_Complex64_t
 *
 *  Overwrites the general complex m-by-n tile C with
 *
 *                                 side = 'PlasmaLeft'   side = 'PlasmaRight'
 *    trans = 'PlasmaNoTrans':           Q * C                  C * Q
 *    trans = 'Plasma_ConjTrans':      Q^H * C                  C * Q^H
 *
 *  where Q is a unitary matrix defined as the product of k
 *  elementary reflectors
 *    \f[
 *        Q = H(k) . . . H(2) H(1)
 *    \f]
 *  as returned by CORE_zgelqt. Q is of order m if side = 'PlasmaLeft' 
 *  and of order n if side = 'PlasmaRight'.
 *
 *******************************************************************************
 *
 * @param[in] side
 *         - PlasmaLeft  : apply Q or Q^H from the Left;
 *         - PlasmaRight : apply Q or Q^H from the Right.
 *
 * @param[in] trans
 *         - PlasmaNoTrans    :  No transpose, apply Q;
 *         - Plasma_ConjTrans :  Transpose, apply Q^H.
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
 *         The inner-blocking size. ib >= 0.
 *
 * @param[in] A
 *         Dimension:  (lda,m) if SIDE = PlasmaLeft,
 *                     (lda,n) if SIDE = PlasmaRight,
 *         The i-th row must contain the vector which defines the
 *         elementary reflector H(i), for i = 1,2,...,k, as returned by
 *         CORE_zgelqt in the first k rows of its array argument A.
 *
 * @param[in] lda
 *         The leading dimension of the array A.  lda >= max(1,k).
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
 *         On exit, C is overwritten by Q*C or Q^H*C or C*Q^H or C*Q.
 *
 * @param[in] ldc
 *         The leading dimension of the array C. ldc >= max(1,m).
 *
 * @param WORK
 *         Auxiliary workspace array of length
 *         ldwork-by-m   if side == PlasmaLeft
 *         ldwork-by-ib  if side == PlasmaRight
 *
 * @param[in] ldwork
 *         The leading dimension of the array WORK.
 *             ldwork >= max(1,ib) if side == PlasmaLeft
 *             ldwork >= max(1,n)  if side == PlasmaRight
 *
 ******************************************************************************/
void CORE_zunmlq(PLASMA_enum side, PLASMA_enum trans,
                 int m, int n, int k, int ib,
                 const PLASMA_Complex64_t *A, int lda,
                 const PLASMA_Complex64_t *T, int ldt,
                 PLASMA_Complex64_t *C, int ldc,
                 PLASMA_Complex64_t *WORK, int ldwork)
{
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
    if ((lda < imax(1,k)) && (k > 0)) {
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

    if (((side == PlasmaLeft) && (trans == PlasmaNoTrans))
        || ((side == PlasmaRight) && (trans != PlasmaNoTrans))) {
        i1 = 0;
        i3 = ib;
    }
    else {
        i1 = ( ( k-1 ) / ib )*ib;
        i3 = -ib;
    }

    if( trans == PlasmaNoTrans) {
        trans = Plasma_ConjTrans;
    }
    else {
        trans = PlasmaNoTrans;
    }

    for(i = i1; (i >- 1) && (i < k); i+=i3 ) {
        kb = imin(ib, k-i);

        if (side == PlasmaLeft) {
            // H or H^H is applied to C(i:m,1:n)
            mi = m - i;
            ic = i;
        }
        else {
            // H or H^H is applied to C(1:m,i:n)
            ni = n - i;
            jc = i;
        }
        // Apply H or H^H
        LAPACKE_zlarfb_work(LAPACK_COL_MAJOR,
            lapack_const(side),
            lapack_const(trans),
            lapack_const(PlasmaForward),
            lapack_const(PlasmaRowwise),
            mi, ni, kb,
            &A[lda*i+i], lda,
            &T[ldt*i], ldt,
            &C[ldc*jc+ic], ldc,
            WORK, ldwork);
    }
}

/******************************************************************************/
void CORE_OMP_zunmlq(PLASMA_enum side, PLASMA_enum trans,
                     int m, int n, int k, int ib, int nb,
                     const PLASMA_Complex64_t *A, int lda,
                     const PLASMA_Complex64_t *T, int ldt,
                     PLASMA_Complex64_t *C,       int ldc)
{
    // assuming m == nb, n == nb
    #pragma omp task depend(in:A[0:nb*nb]) \
                     depend(in:T[0:ib*nb]) \
                     depend(inout:C[0:nb*nb])
    {
        // prepare memory for the auxiliary array
        PLASMA_Complex64_t *WORK =
            (PLASMA_Complex64_t *) malloc((size_t)ib*nb *
                                          sizeof(PLASMA_Complex64_t));
        if (WORK == NULL) {
            plasma_error("malloc() failed");
        }

        int ldwork = nb;

        // call the kernel
        CORE_zunmlq(side, trans,
                    m, n, k, ib,
                    A, lda,
                    T, ldt,
                    C, ldc,
                    WORK, ldwork);

        // deallocate the auxiliary array
        free(WORK);
    }
}
