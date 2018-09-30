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
 * @ingroup core_geqrt
 *
 *  Computes a QR factorization of an m-by-n tile A:
 *  The factorization has the form
 *    \f[
 *        A = Q \times R
 *    \f]
 *  The tile Q is represented as a product of elementary reflectors
 *    \f[
 *        Q = H(1) H(2) ... H(k),
 *    \f]
 *  where \f$ k = min(m,n) \f$.
 *
 *  Each \f$ H(i) \f$ has the form
 *    \f[
 *        H(i) = I - \tau \times v \times v^H
 *    \f]
 *  where \f$ tau \f$ is a scalar, and \f$ v \f$ is a vector with
 *  v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
 *  and \f$ tau \f$ in tau(i).
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
 *         On exit, the elements on and above the diagonal of the array
 *         contain the min(m,n)-by-n upper trapezoidal tile R (R is
 *         upper triangular if m >= n); the elements below the diagonal,
 *         with the array tau, represent the unitary tile Q as a
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
 * @param tau
 *         Auxiliary workspace array of length n.
 *
 * @param work
 *         Auxiliary workspace array of length ib*n.
 *
 * @param[in] lwork
 *         Size of the array work. Should be at least ib*n.
 *
 *******************************************************************************
 *
 * @retval PlasmaSuccess successful exit
 * @retval < 0 if -i, the i-th argument had an illegal value
 *
 ******************************************************************************/
__attribute__((weak))
int plasma_core_zgeqrt(int m, int n, int ib,
                plasma_complex64_t *A, int lda,
                plasma_complex64_t *T, int ldt,
                plasma_complex64_t *tau,
                plasma_complex64_t *work)
{
    // Check input arguments.
    if (m < 0) {
        plasma_coreblas_error("illegal value of m");
        return -1;
    }
    if (n < 0) {
        plasma_coreblas_error("illegal value of n");
        return -2;
    }
    if ((ib < 0) || ( (ib == 0) && (m > 0) && (n > 0) )) {
        plasma_coreblas_error("illegal value of ib");
        return -3;
    }
    if (A == NULL) {
        plasma_coreblas_error("NULL A");
        return -4;
    }
    if (lda < imax(1, m) && m > 0) {
        plasma_coreblas_error("illegal value of lda");
        return -5;
    }
    if (T == NULL) {
        plasma_coreblas_error("NULL T");
        return -6;
    }
    if (ldt < imax(1, ib) && ib > 0) {
        plasma_coreblas_error("illegal value of ldt");
        return -7;
    }
    if (tau == NULL) {
        plasma_coreblas_error("NULL tau");
        return -8;
    }
    if (work == NULL) {
        plasma_coreblas_error("NULL work");
        return -9;
    }

    // quick return
    if (m == 0 || n == 0 || ib == 0)
        return PlasmaSuccess;

    int k = imin(m, n);
    for (int i = 0; i < k; i += ib) {
        int sb = imin(ib, k-i);

        LAPACKE_zgeqr2_work(LAPACK_COL_MAJOR,
                            m-i, sb,
                            &A[lda*i+i], lda,
                            &tau[i], work);

        LAPACKE_zlarft_work(LAPACK_COL_MAJOR,
                            lapack_const(PlasmaForward),
                            lapack_const(PlasmaColumnwise),
                            m-i, sb,
                            &A[lda*i+i], lda,
                            &tau[i],
                            &T[ldt*i], ldt);

        if (n > i+sb) {
            LAPACKE_zlarfb_work(LAPACK_COL_MAJOR,
                                lapack_const(PlasmaLeft),
                                lapack_const(Plasma_ConjTrans),
                                lapack_const(PlasmaForward),
                                lapack_const(PlasmaColumnwise),
                                m-i, n-i-sb, sb,
                                &A[lda*i+i],      lda,
                                &T[ldt*i],        ldt,
                                &A[lda*(i+sb)+i], lda,
                                work, n-i-sb);
        }
    }

    return PlasmaSuccess;
}

/******************************************************************************/
void plasma_core_omp_zgeqrt(int m, int n, int ib,
                     plasma_complex64_t *A, int lda,
                     plasma_complex64_t *T, int ldt,
                     plasma_workspace_t work,
                     plasma_sequence_t *sequence, plasma_request_t *request)
{
    #pragma omp task depend(inout:A[0:lda*n]) \
                     depend(out:T[0:ib*n])
    {
        if (sequence->status == PlasmaSuccess) {
            // Prepare workspaces.
            int tid = omp_get_thread_num();
            plasma_complex64_t *tau = ((plasma_complex64_t*)work.spaces[tid]);

            // Call the kernel.
            int info = plasma_core_zgeqrt(m, n, ib,
                                   A, lda,
                                   T, ldt,
                                   tau,
                                   tau+n);

            if (info != PlasmaSuccess) {
                plasma_error("core_zgeqrt() failed");
                plasma_request_fail(sequence, request, PlasmaErrorInternal);
            }
        }
    }
}
