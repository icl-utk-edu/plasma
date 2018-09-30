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
 * @ingroup core_pemv
 *
 *  performs one of the matrix-vector operations
 *
 *     y = alpha*op(A)*x + beta*y
 *
 *  where  op(A) is one of
 *
 *     op(A) = A   or   op(A) = A^T   or   op(A) = A^H,
 *
 *  alpha and beta are scalars, x and y are vectors and A is a
 *  pentagonal matrix (see further details).
 *
 *
 *  Arguments
 *  ==========
 *
 * @param[in] storev
 *         - PlasmaColumnwise :  array A stored columwise
 *         - PlasmaRowwise    :  array A stored rowwise
 *
 * @param[in] trans
 *         - PlasmaNoTrans   :  y := alpha*A*x   + beta*y.
 *         - PlasmaTrans     :  y := alpha*A^T*x + beta*y.
 *         - PlasmaConjTrans :  y := alpha*A^H*x + beta*y.
 *
 * @param[in] m
 *         Number of rows of the matrix A.
 *         m must be at least zero.
 *
 * @param[in] n
 *         Number of columns of the matrix A.
 *         n must be at least zero.
 *
 * @param[in] l
 *         Order of triangle within the matrix A (l specifies the shape
 *         of the matrix A; see further details).
 *
 * @param[in] alpha
 *         Scalar alpha.
 *
 * @param[in] A
 *         Array of size lda-by-n.  On entry, the leading m-by-n part
 *         of the array A must contain the matrix of coefficients.
 *
 * @param[in] lda
 *         Leading dimension of array A.
 *
 * @param[in] X
 *         On entry, the incremented array X must contain the vector x.
 *
 * @param[in] incx
 *         Increment for the elements of X. incx must not be zero.
 *
 * @param[in] beta
 *         Scalar beta.
 *
 * @param[in,out] Y
 *         On entry, the incremented array Y must contain the vector y.
 *
 * @param[out] incy
 *         Increment for the elements of Y. incy must not be zero.
 *
 * @param[out] work
 *         Workspace array of size at least l.
 *
 *  Further Details
 *  ===============
 *
 *               |     n    |
 *            _   ___________   _
 *               |          |
 *     A:        |          |
 *          m-l  |          |
 *               |          |  m
 *            _  |.....     |
 *               \    :     |
 *            l    \  :     |
 *            _      \:_____|  _
 *
 *               |  l | n-l |
 *
 *
 *******************************************************************************
 *
 * @retval PlasmaSuccess successful exit
 * @retval < 0 if -i, the i-th argument had an illegal value
 *
 ******************************************************************************/
__attribute__((weak))
int plasma_core_zpemv(plasma_enum_t trans, int storev,
               int m, int n, int l,
               plasma_complex64_t alpha,
               const plasma_complex64_t *A, int lda,
               const plasma_complex64_t *X, int incx,
               plasma_complex64_t beta,
               plasma_complex64_t *Y, int incy,
               plasma_complex64_t *work)
{
    // Check input arguments.
    if ((trans != PlasmaNoTrans) &&
        (trans != PlasmaTrans)   &&
        (trans != PlasmaConjTrans)) {
        plasma_coreblas_error("Illegal value of trans");
        return -1;
    }
    if ((storev != PlasmaColumnwise) && (storev != PlasmaRowwise)) {
        plasma_coreblas_error("Illegal value of storev");
        return -2;
    }
    if (!( ((storev == PlasmaColumnwise) && (trans != PlasmaNoTrans)) ||
           ((storev == PlasmaRowwise) && (trans == PlasmaNoTrans)) )) {
        plasma_coreblas_error("Illegal values of trans/storev");
        return -2;
    }
    if (m < 0) {
        plasma_coreblas_error("Illegal value of m");
        return -3;
    }
    if (n < 0) {
        plasma_coreblas_error("Illegal value of n");
        return -4;
    }
    if (l > imin(m, n)) {
        plasma_coreblas_error("Illegal value of l");
        return -5;
    }
    if (lda < imax(1, m)) {
        plasma_coreblas_error("Illegal value of lda");
        return -8;
    }
    if (incx < 1) {
        plasma_coreblas_error("Illegal value of incx");
        return -10;
    }
    if (incy < 1) {
        plasma_coreblas_error("Illegal value of incy");
        return -13;
    }

    // quick return
    if ((m == 0) || (n == 0))
        return PlasmaSuccess;
    if ((alpha == 0.0) && (beta == 0.0))
        return PlasmaSuccess;

    // If l < 2, there is no triangular part.
    if (l == 1) l = 0;

    // Columnwise
    if (storev == PlasmaColumnwise) {
        //        ______________
        //        |      |     |    A1: A[0]
        //        |      |     |    A2: A[m-l]
        //        |  A1  |     |    A3: A[(n-l)*lda]
        //        |      |     |
        //        |______| A3  |
        //        \      |     |
        //          \ A2 |     |
        //            \  |     |
        //              \|_____|

        // Columnwise / NoTrans
        if (trans == PlasmaNoTrans) {
            plasma_coreblas_error("PlasmaNoTrans/PlasmaColumnwise not implemented");
            return -1;
        }
        // Columnwise / [Conj]Trans
        else {
            // l top rows of y
            if (l > 0) {
                // w = A_2^H * x_2
                cblas_zcopy(l, &X[incx*(m-l)], incx, work, 1);
                cblas_ztrmv(CblasColMajor, (CBLAS_UPLO)PlasmaUpper,
                            (CBLAS_TRANSPOSE)trans,
                            (CBLAS_DIAG)PlasmaNonUnit,
                            l, &A[m-l], lda, work, 1);

                if (m > l) {
                    // y_1 = beta * y_1 [ + alpha * A_1 * x_1 ]
                    cblas_zgemv(CblasColMajor, (CBLAS_TRANSPOSE)trans,
                                m-l, l, CBLAS_SADDR(alpha), A, lda,
                                X, incx, CBLAS_SADDR(beta), Y, incy);

                    // y_1 = y_1 + alpha * w
                    cblas_zaxpy(l, CBLAS_SADDR(alpha), work, 1, Y, incy);
                }
                else {
                    // y_1 = y_1 + alpha * w
                    if (beta == 0.0) {
                        cblas_zscal(l, CBLAS_SADDR(alpha), work, 1);
                        cblas_zcopy(l, work, 1, Y, incy);
                    }
                    else {
                        cblas_zscal(l, CBLAS_SADDR(beta), Y, incy);
                        cblas_zaxpy(l, CBLAS_SADDR(alpha), work, 1, Y, incy);
                    }
                }
            }

            // n-l bottom rows of Y
            if (n > l) {
                int k = n - l;
                cblas_zgemv(CblasColMajor, (CBLAS_TRANSPOSE)trans,
                            m, k, CBLAS_SADDR(alpha), &A[lda*l], lda,
                            X, incx, CBLAS_SADDR(beta), &Y[incy*l], incy);
            }
        }
    }
    // Rowwise
    else {
        // --------------
        // |            | \           A1:  A[0]
        // |    A1      |   \         A2:  A[(n-l) * lda]
        // |            | A2  \       A3:  A[l]
        // |--------------------|
        // |        A3          |
        // ----------------------

        // Rowwise / NoTrans
        if (trans == PlasmaNoTrans) {
            // l top rows of A and y
            if (l > 0) {
                // w = A_2 * x_2
                cblas_zcopy(l, &X[incx*(n-l)], incx, work, 1);
                cblas_ztrmv(CblasColMajor, (CBLAS_UPLO)PlasmaLower,
                            (CBLAS_TRANSPOSE)PlasmaNoTrans,
                            (CBLAS_DIAG)PlasmaNonUnit,
                            l, &A[lda*(n-l)], lda, work, 1);

                if (n > l) {
                    // y_1 = beta * y_1 [ + alpha * A_1 * x_1 ]
                    cblas_zgemv(CblasColMajor, (CBLAS_TRANSPOSE)PlasmaNoTrans,
                                l, n-l, CBLAS_SADDR(alpha), A, lda,
                                X, incx, CBLAS_SADDR(beta), Y, incy);

                    // y_1 = y_1 + alpha * w
                    cblas_zaxpy(l, CBLAS_SADDR(alpha), work, 1, Y, incy);
                }
                else {
                    // y_1 = y_1 + alpha * w
                    if (beta == 0.0) {
                        cblas_zscal(l, CBLAS_SADDR(alpha), work, 1);
                        cblas_zcopy(l, work, 1, Y, incy);
                    }
                    else {
                        cblas_zscal(l, CBLAS_SADDR(beta), Y, incy);
                        cblas_zaxpy(l, CBLAS_SADDR(alpha), work, 1, Y, incy);
                    }
                }
            }

            // m-l bottom rows of Y
            if (m > l) {
                cblas_zgemv(
                        CblasColMajor, (CBLAS_TRANSPOSE)PlasmaNoTrans,
                        m-l, n, CBLAS_SADDR(alpha), &A[l], lda,
                        X, incx, CBLAS_SADDR(beta), &Y[incy*l], incy);
            }
        }
        // Rowwise/[Conj]Trans
        else {
            plasma_coreblas_error("Plasma[Conj]Trans/PlasmaRowwise not implemented");
            return -1;
        }
    }

    return PlasmaSuccess;
}
