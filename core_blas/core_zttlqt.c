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

// This will be swapped during the automatic code generation.
#undef REAL
#define COMPLEX

/***************************************************************************//**
 *
 * @ingroup core_ttlqt
 *
 *  Computes an LQ factorization of a rectangular matrix
 *  formed by coupling side-by-side an m-by-m lower triangular tile A1
 *  and an m-by-n lower triangular tile A2:
 *
 *    | A1 A2 | = L * Q
 *
 *
 *******************************************************************************
 *
 * @param[in] m
 *         The number of rows of the tile A1 and A2. m >= 0.
 *         The number of columns of the tile A1.
 *
 * @param[in] n
 *         The number of columns of the tile A2. n >= 0.
 *
 * @param[in] ib
 *         The inner-blocking size.  ib >= 0.
 *
 * @param[in,out] A1
 *         On entry, the m-by-m tile A1.
 *         On exit, the elements on and below the diagonal of the array
 *         contain the m-by-m lower trapezoidal tile L;
 *         the elements above the diagonal are not referenced.
 *
 * @param[in] lda1
 *         The leading dimension of the array A1.  lda1 >= max(1,m).
 *
 * @param[in,out] A2
 *         On entry, the m-by-n lower triangular tile A2.
 *         On exit, the elements on and below the diagonal of the array
 *         with the matrix T represent
 *         the unitary tile Q as a product of elementary reflectors.
 *
 * @param[in] lda2
 *         The leading dimension of the array A2.  lda2 >= max(1,m).
 *
 * @param[out] T
 *         The ib-by-m triangular factor T of the block reflector.
 *         T is upper triangular by block (economic storage);
 *         The rest of the array is not referenced.
 *
 * @param[in] ldt
 *         The leading dimension of the array T. ldt >= ib.
 *
 * @param tau
 *         Auxiliary workspace array of length m.
 *
 * @param work
 *         Auxiliary workspace array of length ib*m.
 *
 *******************************************************************************
 *
 * @retval PlasmaSuccess successful exit
 * @retval < 0 if -i, the i-th argument had an illegal value
 *
 ******************************************************************************/
__attribute__((weak))
int plasma_core_zttlqt(int m, int n, int ib,
                plasma_complex64_t *A1, int lda1,
                plasma_complex64_t *A2, int lda2,
                plasma_complex64_t *T,  int ldt,
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
    if (ib < 0) {
        plasma_coreblas_error("illegal value of ib");
        return -3;
    }
    if (A1 == NULL) {
        plasma_coreblas_error("NULL A1");
        return -4;
    }
    if (lda1 < imax(1, m) && m > 0) {
        plasma_coreblas_error("illegal value of lda1");
        return -5;
    }
    if (A2 == NULL) {
        plasma_coreblas_error("NULL A2");
        return -6;
    }
    if (lda2 < imax(1, m) && m > 0) {
        plasma_coreblas_error("illegal value of lda2");
        return -7;
    }
    if (T == NULL) {
        plasma_coreblas_error("NULL T");
        return -8;
    }
    if (ldt < imax(1, ib) && ib > 0) {
        plasma_coreblas_error("illegal value of ldt");
        return -9;
    }
    if (tau == NULL) {
        plasma_coreblas_error("NULL tau");
        return -10;
    }
    if (work == NULL) {
        plasma_coreblas_error("NULL work");
        return -11;
    }

    // quick return
    if ((m == 0) || (n == 0) || (ib == 0))
        return PlasmaSuccess;

    // TODO: Need to check why some cases require this to avoid
    // uninitialized values
    //core_zlaset(PlasmaGeneral, ib, m, 0.0, 0.0, T, ldt);

    for (int ii = 0; ii < m; ii += ib) {
        int sb = imin(m-ii, ib);
        for (int i = 0; i < sb; i++) {
            int j  = ii + i;
            int mi = sb-i-1;
            int ni = imin( j + 1, n);

            // Generate elementary reflector H(ii*ib+i) to annihilate
            // A(ii*ib+i, ii*ib+i:m).
#ifdef COMPLEX
            LAPACKE_zlacgv_work(ni, &A2[j], lda2);
            LAPACKE_zlacgv_work(1, &A1[lda1*j+j], lda1);
#endif
            LAPACKE_zlarfg_work(ni+1, &A1[lda1*j+j], &A2[j], lda2, &tau[j]);

            plasma_complex64_t alpha;
            if (mi > 0) {
                // Apply H(j-1) to A(j:ii+ib-1, j-1:m) from the right.
                cblas_zcopy(
                    mi,
                    &A1[lda1*j+(j+1)], 1,
                    work, 1);

                plasma_complex64_t zone  = 1.0;
                cblas_zgemv(
                    CblasColMajor, (CBLAS_TRANSPOSE)PlasmaNoTrans,
                    mi, ni,
                    CBLAS_SADDR(zone), &A2[j+1], lda2,
                    &A2[j], lda2,
                    CBLAS_SADDR(zone), work, 1);

                alpha = -(tau[j]);
                cblas_zaxpy(
                    mi, CBLAS_SADDR(alpha),
                    work, 1,
                    &A1[lda1*j+j+1], 1);

                cblas_zgerc(
                    CblasColMajor, mi, ni,
                    CBLAS_SADDR(alpha), work, 1,
                    &A2[j], lda2,
                    &A2[j+1], lda2);
            }

            // Calculate T.
            if (i > 0 ) {
                int l = imin(i, imax(0, n-ii));
                alpha = -(tau[j]);
                plasma_core_zpemv(
                        PlasmaNoTrans, PlasmaRowwise,
                        i, imin(j, n), l,
                        alpha, &A2[ii], lda2,
                        &A2[j], lda2,
                        0.0, &T[ldt*j], 1,
                        work);

                // T(0:i-1, j) = T(0:i-1, ii:j-1) * T(0:i-1, j)
                cblas_ztrmv(
                        CblasColMajor, (CBLAS_UPLO)PlasmaUpper,
                        (CBLAS_TRANSPOSE)PlasmaNoTrans,
                        (CBLAS_DIAG)PlasmaNonUnit,
                        i, &T[ldt*ii], ldt,
                        &T[ldt*j], 1);
            }

#ifdef COMPLEX
            LAPACKE_zlacgv_work(ni, &A2[j], lda2 );
            LAPACKE_zlacgv_work(1, &A1[lda1*j+j], lda1 );
#endif
            T[ldt*j+i] = tau[j];
        }

        // Apply Q to the rest of the matrix to the right.
        if (m > ii+sb) {
            int mi = m-(ii+sb);
            int ni = imin(ii+sb, n);
            int l  = imin(sb, imax(0, ni-ii));
            plasma_core_zparfb(
                PlasmaRight, PlasmaNoTrans,
                PlasmaForward, PlasmaRowwise,
                mi, ib, mi, ni, sb, l,
                &A1[lda1*ii+ii+sb], lda1,
                &A2[ii+sb], lda2,
                &A2[ii], lda2,
                &T[ldt*ii], ldt,
                work, m);
        }
    }
    return PlasmaSuccess;
}

/******************************************************************************/
void plasma_core_omp_zttlqt(int m, int n, int ib,
                     plasma_complex64_t *A1, int lda1,
                     plasma_complex64_t *A2, int lda2,
                     plasma_complex64_t *T,  int ldt,
                     plasma_workspace_t work,
                     plasma_sequence_t *sequence, plasma_request_t *request)
{
    #pragma omp task depend(inout:A1[0:lda1*m]) \
                     depend(inout:A2[0:lda2*n]) \
                     depend(out:T[0:ib*m]) // T should be mxib, but is stored
                                           // as ibxm
    {
        if (sequence->status == PlasmaSuccess) {
            // Prepare workspaces.
            int tid = omp_get_thread_num();
            plasma_complex64_t *tau = ((plasma_complex64_t*)work.spaces[tid]);

            // Call the kernel.
            int info = plasma_core_zttlqt(m, n, ib,
                                   A1, lda1,
                                   A2, lda2,
                                   T,  ldt,
                                   tau,
                                   tau+m);

            if (info != PlasmaSuccess) {
                plasma_error("core_ztslqt() failed");
                plasma_request_fail(sequence, request, PlasmaErrorInternal);
            }
        }
    }
}
