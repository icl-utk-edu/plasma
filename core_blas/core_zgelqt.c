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
 * @ingroup core_gelqt
 *
 *  Computes the LQ factorization of an m-by-n tile A:
 *  The factorization has the form
 *    \f[
 *        A = L \times Q
 *    \f]
 *  The tile Q is represented as a product of elementary reflectors
 *    \f[
 *        Q = H(k)^H . . . H(2)^H H(1)^H,
 *    \f]
 *  where \f$ k = min(m,n) \f$.
 *
 *  Each \f$ H(i) \f$ has the form
 *    \f[
 *        H(i) = I - \tau \times v \times v^H
 *    \f]
 *  where \f$ tau \f$ is a scalar, and \f$ v \f$ is a vector with
 *  v(1:i-1) = 0 and v(i) = 1; v(i+1:n)^H is stored on exit in A(i,i+1:n),
 *  and \f$ tau \f$ in TAU(i).
 *
 *******************************************************************************
 *
 * @param[in] m
 *          The number of rows of the tile A.  m >= 0.
 *
 * @param[in] n
 *         The number of columns of the tile A.  n >= 0.
 *
 * @param[in] ib
 *         The inner-blocking size.  ib >= 0.
 *
 * @param[in,out] A
 *         On entry, the m-by-n tile A.
 *         On exit, the elements on and below the diagonal of the array
 *         contain the m-by-min(m,n) lower trapezoidal tile L (L is
 *         lower triangular if m <= n); the elements above the diagonal,
 *         with the array TAU, represent the unitary tile Q as a
 *         product of elementary reflectors (see Further Details).
 *
 * @param[in] lda
 *         The leading dimension of the array A.  lda >= max(1,m).
 *
 * @param[out] T
 *         The ib-by-n triangular factor T of the block reflector.
 *         T is upper triangular by block (economic storage);
 *         The rest of the array is not referenced.
 *
 * @param[in] ldt
 *         The leading dimension of the array T. ldt >= ib.
 *
 * @param TAU
 *         Auxiliarry workspace array of length m.
 *
 * @param WORK
 *         Auxiliary workspace array of length ib*m.
 *
 ******************************************************************************/
void CORE_zgelqt(int m, int n, int ib,
                 PLASMA_Complex64_t *A, int lda,
                 PLASMA_Complex64_t *T, int ldt,
                 PLASMA_Complex64_t *TAU,
                 PLASMA_Complex64_t *WORK)
{
    // Check input arguments
    if (m < 0) {
        plasma_error("Illegal value of m");
        return;
    }
    if (n < 0) {
        plasma_error("Illegal value of n");
        return;
    }
    if ((ib < 0) || ( (ib == 0) && ((m > 0) && (n > 0)) )) {
        plasma_error("Illegal value of ib");
        return;
    }
    if ((lda < imax(1,m)) && (m > 0)) {
        plasma_error("Illegal value of lda");
        return;
    }
    if ((ldt < imax(1,ib)) && (ib > 0)) {
        plasma_error("Illegal value of ldt");
        return;
    }

    // Quick return
    if ((m == 0) || (n == 0) || (ib == 0))
        return;

    int k = imin(m, n);
    for (int i = 0; i < k; i += ib) {
        int sb = imin(ib, k-i);

        LAPACKE_zgelq2_work(LAPACK_COL_MAJOR, sb, n-i,
                            &A[lda*i+i], lda, &TAU[i], WORK);

        LAPACKE_zlarft_work(LAPACK_COL_MAJOR,
            lapack_const(PlasmaForward),
            lapack_const(PlasmaRowwise),
            n-i, sb,
            &A[lda*i+i], lda, &TAU[i],
            &T[ldt*i], ldt);

        if (m > i+sb) {
            LAPACKE_zlarfb_work(
                LAPACK_COL_MAJOR,
                lapack_const(PlasmaRight),
                lapack_const(PlasmaNoTrans),
                lapack_const(PlasmaForward),
                lapack_const(PlasmaRowwise),
                m-i-sb, n-i, sb,
                &A[lda*i+i],      lda,
                &T[ldt*i],        ldt,
                &A[lda*i+(i+sb)], lda,
                WORK, m-i-sb);
        }
    }
}

/******************************************************************************/
void CORE_OMP_zgelqt(int m, int n, int ib, int nb,
                     PLASMA_Complex64_t *A, int lda,
                     PLASMA_Complex64_t *T, int ldt)
{
    // assuming lda == m and nb == n
    #pragma omp task depend(inout:A[0:lda*nb]) \
                     depend(out:T[0:ldt*nb])
    {
        // prepare memory for auxiliary arrays
        PLASMA_Complex64_t *TAU =
            (PLASMA_Complex64_t *) malloc((size_t)nb *
                                          sizeof(PLASMA_Complex64_t));
        if (TAU == NULL) {
            plasma_error("malloc() failed");
        }
        PLASMA_Complex64_t *WORK =
            (PLASMA_Complex64_t *) malloc((size_t)ib*nb *
                                          sizeof(PLASMA_Complex64_t));
        if (WORK == NULL) {
            plasma_error("malloc() failed");
        }

        // call the kernel
        CORE_zgelqt(m, n, ib,
                    A, lda,
                    T, ldt,
                    TAU,
                    WORK);

        // deallocate auxiliary arrays
        free(TAU);
        free(WORK);
    }
}
