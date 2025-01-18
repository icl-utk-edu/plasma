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
 * @ingroup cire_tsmlq_corner
 *
 * Applies left and right transformations as depicted below:
 * |I -VTV'| * | A1  A2 | * |I - VT'V'|
 *             | A2' A3 |
 * where A1 and A3 are symmetric matrices.
 * Only the lower part is referenced.
 * This is an adhoc implementation, can be further optimized...
 *
 *******************************************************************************
 *
 * @param[in] m1
 *         The number of rows of the tile A1. m1 >= 0.
 *
 * @param[in] n1
 *         The number of columns of the tile A1. n1 >= 0.
 *
 * @param[in] m2
 *         The number of rows of the tile A2. m2 >= 0.
 *
 * @param[in] n2
 *         The number of columns of the tile A2. n2 >= 0.
 *
 * @param[in] m3
 *         The number of rows of the tile A3. m3 >= 0.
 *
 * @param[in] n3
 *         The number of columns of the tile A3. n3 >= 0.
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
 * @param[in,out] A3
 *         On entry, the m3-by-n3 tile A3.
 *
 * @param[in] lda3
 *         The leading dimension of the tile A3. lda3 >= max(1,m3).
 *
 * @param[in] V
 *         The i-th row must contain the vector which defines the
 *         elementary reflector H(i), for i = 1,2,...,k, as returned by
 *         CORE_ZTSLQT in the first k rows of its array argument V.
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
 *             ldwork-by-m1 if side == PlasmaLeft
 *             ldwork-by-ib if side == PlasmaRight
 *
 * @param[in] ldwork
 *         The leading dimension of the array work.
 *             ldwork >= max(1,ib) if side == PlasmaLeft
 *             ldwork >= max(1,n1) if side == PlasmaRight
 *
 *******************************************************************************
 *
 * @retval PlasmaSuccess successful exit
 * @retval < 0 if -i, the i-th argument had an illegal value
 *
 ******************************************************************************/
int plasma_core_ztsmlq_corner(
    int m1, int n1, int m2, int n2,
    int m3, int n3, int k, int ib,
          plasma_complex64_t *A1, int lda1,
          plasma_complex64_t *A2, int lda2,
          plasma_complex64_t *A3, int lda3,
    const plasma_complex64_t *V,  int ldv,
    const plasma_complex64_t *T,  int ldt,
    plasma_complex64_t *work, int ldwork)
{
    plasma_enum_t side;
    plasma_enum_t trans;
    int i, j;

    // Check input arguments
    if (m1 != n1) {
        plasma_coreblas_error("Illegal value of M1, N1");
        return -1;
    }
    int nb = n1;
    // Rebuild the symmetric block: work <- A1
    for (i = 0; i < m1; ++i)
        for (j = i; j < n1; ++j) {
            *(work + i + j*ldwork) = *(A1 + i + j*lda1);
            if (j > i) {
                *(work + j + i*ldwork) =  conj( *(work + i + j*ldwork) );
            }
        }

    //  Copy the transpose of A2: work+nb*ldwork <- A2'
    for (j = 0; j < n2; ++j)
        for (i = 0; i < m2; ++i) {
            *(work + j + (i + nb) * ldwork) = conj( *(A2 + i + j*lda2) );
        }

    side = PlasmaRight;
    trans = Plasma_ConjTrans;

    //  Right application on |A1 A2|
    plasma_core_ztsmlq(
        side, trans, m1, n1, m2, n2, k, ib,
        work, ldwork, A2, lda2,
        V, ldv, T, ldt,
        work+3*nb*ldwork, ldwork);

    //  Rebuild the symmetric block: work+2*nb*ldwork <- A3
    for (i = 0; i < m3; ++i)
        for (j = i; j < n3; ++j) {
            *(work + i + (j + 2*nb) * ldwork) = *(A3 + i + j*lda3);
            if (j > i) {
                *(work + j + (i + 2*nb) * ldwork) =  conj ( *(work + i + (j + 2*nb) * ldwork) );
            }
        }

    //  Right application on | A2' A3 |
    plasma_core_ztsmlq(
        side, trans, n2, m2, m3, n3, k, ib,
        work+nb*ldwork, ldwork, work+2*nb*ldwork, ldwork,
        V, ldv, T, ldt,
        work + 3*nb*ldwork, ldwork);

    side = PlasmaLeft;
    trans = PlasmaNoTrans;

    //========================================================
    //  Left application on | A1  |
    //                      | A2' |
    //========================================================
    plasma_core_ztsmlq(
        side, trans, m1, n1, n2, m2, k, ib,
        work, ldwork, work+nb*ldwork, ldwork,
        V, ldv, T, ldt,
        work + 3*nb*ldwork, ldwork);

    //========================================================
    //  Copy back the final result to the upper part of A1
    //  A1 = work
    //========================================================
    for (i = 0; i < m1; ++i)
        for (j = i; j < n1; ++j)
            *(A1 + i + j*lda1) = *(work + i + j*ldwork);

    //========================================================
    //  Left application on | A2 |
    //                     | A3 |
    //========================================================
    plasma_core_ztsmlq(
        side, trans, m2, n2, m3, n3, k, ib,
        A2, lda2, work+2*nb*ldwork, ldwork,
        V, ldv, T, ldt,
        work + 3*nb*ldwork, ldwork);

    //========================================================
    //  Copy back the final result to the upper part of A3
    //  A3 = work+2*nb*ldwork
    //========================================================
    for (i = 0; i < m3; ++i)
        for (j = i; j < n3; ++j)
            *(A3 + i + j*lda3) = *(work + i + (j+ 2*nb) * ldwork);

    return PlasmaSuccess;
}

void plasma_core_omp_ztsmlq_corner(
    int m1, int n1, int m2, int n2,
    int m3, int n3, int k, int ib,
          plasma_complex64_t *A1, int lda1,
          plasma_complex64_t *A2, int lda2,
          plasma_complex64_t *A3, int lda3,
    const plasma_complex64_t *V,  int ldv,
    const plasma_complex64_t *T,  int ldt,
    plasma_workspace_t work,
    plasma_sequence_t *sequence, plasma_request_t *request)
{
    int nb = n1;
    // assuming m1 == nb, n1 == nb, m2 == nb, n2 == nb
    // m3 == nb, n3 == nb
    #pragma omp task depend(inout:A1[0:nb*nb]) \
                     depend(inout:A2[0:nb*nb]) \
                     depend(inout:A3[0:nb*nb]) \
                     depend(in:V[0:nb*nb]) \
                     depend(in:T[0:ib*nb])
    {
        if (sequence->status == PlasmaSuccess) {
            int tid = omp_get_thread_num();
            plasma_complex64_t *W   =
                ((plasma_complex64_t*)work.spaces[tid]);

            int ldwork = nb;

            // call the kernel
            int info = plasma_core_ztsmlq_corner(
                m1, n1, m2, n2, m3, n3, k, ib,
                A1, lda1,
                A2, lda2,
                A3, lda3,
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

