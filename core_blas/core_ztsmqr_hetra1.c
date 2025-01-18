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

#include "plasma_core_blas.h"
#include "plasma_types.h"
#include "plasma_internal.h"
#include "core_lapack.h"

#include <omp.h>

/***************************************************************************//**
 *
 * @ingroup core_tsmqr_hetra1
 *
 *  This kernel applies a left transformation on | A1'|
 *                                               | A2 |
 *
 * Needs therefore to make the explicit transpose of A1 before
 * and after the application of the block of reflectors
 * Can be further optimized by changing accordingly the underneath
 * kernel ztsrfb!
 *
 *******************************************************************************
 *
 * @param[in] side
 *         - PlasmaLeft  : apply Q or Q**H from the Left;
 *         - PlasmaRight : apply Q or Q**H from the Right.
 *
 * @param[in] trans
 *         - PlasmaNoTrans   :  apply Q;
 *         - PlasmaConjTrans :  apply Q**H.
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
 *         plasma_core_ztsqrt in the first k columns of its array argument V.
 *
 * @param[in] ldv
 *         The leading dimension of the array V. ldv >= max(1,K).
 *
 * @param[in] T
 *         The ib-by-n1 triangular factor T of the block reflector.
 *         T is upper triangular by block (economic storage);
 *         The rest of the array is not referenced.
 *
 * @param[in] ldt
 *         The leading dimension of the array T. ldt >= ib.
 *
 * @param[out] work
 *         Workspace array of size
 *             ldwork-by-n1 if side == PlasmaLeft
 *             ldwork-by-ib if side == PlasmaRight
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
int plasma_core_ztsmqr_hetra1(
    plasma_enum_t side, plasma_enum_t trans,
    int m1, int n1, int m2, int n2, int k, int ib,
          plasma_complex64_t *A1, int lda1,
          plasma_complex64_t *A2, int lda2,
    const plasma_complex64_t *V,  int ldv,
    const plasma_complex64_t *T,  int ldt,
          plasma_complex64_t *work, int ldwork)
{
    int i, j;

    // Check input arguments.
    if (m1 != n1) {
        plasma_coreblas_error("Illegal value of m1, n1");
        return -3;
    }

    // in-place transposition of A1
    for (j = 0; j < n1; ++j) {
        A1[j + j*lda1] = conj(A1[j + j*lda1]);

        for (i = j+1; i < m1; ++i) {
            *work = *(A1 + i + j*lda1);
            *(A1 + i + j*lda1) = conj(*(A1 + j + i*lda1));
            *(A1 + j + i*lda1) = conj(*work);
        }
    }

    plasma_core_ztsmqr(
        side, trans,
        m1, n1, m2, n2, k, ib,
        A1, lda1,
        A2, lda2,
        V, ldv,
        T, ldt,
        work, ldwork);

    // in-place transposition of A1
    for (j = 0; j < n1; ++j) {
        A1[j + j*lda1] = conj(A1[j + j*lda1]);

        for (i = j+1; i < m1; ++i) {
            *work = *(A1 + i + j*lda1);
            *(A1 + i + j*lda1) = conj(*(A1 + j + i*lda1));
            *(A1 + j + i*lda1) = conj(*work);
        }
    }

    return PlasmaSuccess;
}

/******************************************************************************/
void plasma_core_omp_ztsmqr_hetra1(
    plasma_enum_t side, plasma_enum_t trans,
    int m1, int n1, int m2, int n2, int k, int ib,
          plasma_complex64_t *A1, int lda1,
          plasma_complex64_t *A2, int lda2,
    const plasma_complex64_t *V,  int ldv,
    const plasma_complex64_t *T,  int ldt,
    plasma_workspace_t work,
    plasma_sequence_t *sequence, plasma_request_t *request)
{
    int nb = n1;
    // omp depends assume m1 == nb, n1 == nb, m2 == nb, n2 == nb.
    #pragma omp task depend(inout:A1[0:nb*nb]) \
                     depend(inout:A2[0:nb*nb]) \
                     depend(in:V[0:nb*nb]) \
                     depend(in:T[0:ib*nb])
    {
        if (sequence->status == PlasmaSuccess) {
            int tid = omp_get_thread_num();
            plasma_complex64_t *W   =
                ((plasma_complex64_t*)work.spaces[tid]);

            int ldwork = side == PlasmaLeft ? ib : nb;

            // Call the kernel.
            int info = plasma_core_ztsmqr_hetra1(
                side, trans,
                m1, n1, m2, n2, k, ib,
                A1, lda1,
                A2, lda2,
                V, ldv,
                T, ldt,
                W, ldwork);

            if (info != PlasmaSuccess) {
                plasma_error_with_code("Error in call to COREBLAS in argument",
                                       -info);
                plasma_request_fail(sequence, request,
                                    PlasmaErrorIllegalValue);
            }
        }
    }
}
