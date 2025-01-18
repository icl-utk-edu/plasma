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
 * @ingroup core_herfb
 *
 *  Overwrites the Hermitian complex n-by-n tile C with
 *
 *    Q^H*C*Q
 *
 *  where Q is a complex unitary matrix defined as the product of k
 *  elementary reflectors
 *
 *    Q = H(1) H(2) . . . H(k)
 *
 *  as returned by CORE_zgeqrt. Only PlasmaLower supported!
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *         - PlasmaLower : the upper part of the Hermitian matrix C
 *                         is not referenced.
 *         - PlasmaUpper : the lower part of the Hermitian matrix C
 *                         is not referenced (not supported).
 * @param[in] n
 *          The number of rows/columns of the tile C. n >= 0.
 *
 * @param[in] k
 *         The number of elementary reflectors whose product defines
 *         the matrix Q. k >= 0.
 *
 * @param[in] ib
 *         The inner-blocking size. ib >= 0.

 * @param[in] A
 *         The i-th column must contain the vector which defines the
 *         elementary reflector H(i), for i = 1,2,...,k, as returned by
 *         CORE_zgeqrt in the first k columns of its array argument A.
 *
 * @param[in] lda
 *         The leading dimension of the array A. lda >= max(1,n).
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
 *         On entry, the Hermitian n-by-n tile C.
 *         On exit, C is overwritten by Q^H*C*Q.
 *
 * @param[in] ldc
 *         The leading dimension of the array C. ldc >= max(1,m).
 *
 * @param[in,out] work
 *         On exit, if INFO = 0, work(1) returns the optimal ldwork.
 *
 * @param[in] ldwork
 *         The dimension of the array work. ldwork >= max(1,n);
 *
 *******************************************************************************
 *
 * @retval  PlasmaSuccess successful exit
 * @retval  < 0 if -i, the i-th argument had an illegal value
 *
 ******************************************************************************/
int plasma_core_zherfb(
    plasma_enum_t uplo,
    int n, int k, int ib,
    const plasma_complex64_t *A,    int lda,
    const plasma_complex64_t *T,    int ldt,
          plasma_complex64_t *C,    int ldc,
          plasma_complex64_t *work, int ldwork )
{
    plasma_complex64_t tmp;
    int i, j;

    // Check input arguments.
    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        plasma_coreblas_error("Illegal value of uplo");
        return -1;
    }
    if (n < 0) {
        plasma_coreblas_error("Illegal value of n");
        return -2;
    }
    if (k < 0) {
        plasma_coreblas_error("Illegal value of k");
        return -3;
    }
    if (ib < 0) {
        plasma_coreblas_error("Illegal value of ib");
        return -4;
    }
    if (lda < imax(1,n) && n > 0) {
        plasma_coreblas_error("Illegal value of lda");
        return -6;
    }
    if (ldt < imax(1,ib) && ib > 0) {
        plasma_coreblas_error("Illegal value of ldt");
        return -8;
    }
    if (ldc < imax(1,n) && n > 0) {
        plasma_coreblas_error("Illegal value of ldc");
        return -10;
    }
    if (ldwork < imax(1,n)) {
        plasma_coreblas_error("Illegal value of ldwork");
        return -12;
    }

    // Quick return
    if (n == 0 || k == 0 || ib == 0)
        return PlasmaSuccess;

    int nb = n;

    if (uplo == PlasmaLower) {
        // Rebuild the Hermitian block: work <- C
        for (j = 0; j < n; ++j) {
            *(work + j + j * ldwork) =  *(C + ldc*j + j);
            for (i = j+1; i < n; ++i) {
                tmp = *(C + i + j*ldc);
                *(work + i + j * ldwork) = tmp;
                *(work + j + i * ldwork) = conj( tmp );
            }
        }

        // Left
        plasma_core_zunmqr(
            PlasmaLeft, Plasma_ConjTrans, n, n, k, ib,
            A, lda, T, ldt, work, ldwork, work+nb*ldwork, ldwork);
        // Right
        plasma_core_zunmqr(
            PlasmaRight, PlasmaNoTrans, n, n, k, ib,
            A, lda, T, ldt, work, ldwork, work+nb*ldwork, ldwork);

        //====================================================
        // Copy back the final result to the lower part of C
        //===================================================
        LAPACKE_zlacpy_work(
            LAPACK_COL_MAJOR, lapack_const(PlasmaLower),
            n, n, work, ldwork, C, ldc );
    }
    else {
        //===================================================
        // Rebuild the Hermitian block: work <- C
        //===================================================
        for (j = 0; j < n; ++j) {
            for (i = 0; i < j; ++i) {
                tmp = *(C + i + j*ldc);
                *(work + i + j * ldwork) = tmp;
                *(work + j + i * ldwork) = conj( tmp );
            }
            *(work + j + j * ldwork) =  *(C + ldc*j + j);
        }

        // Right
        plasma_core_zunmlq(
            PlasmaRight, Plasma_ConjTrans, n, n, k, ib,
            A, lda, T, ldt, work, ldwork, work+nb*ldwork, ldwork);
        // Left
        plasma_core_zunmlq(
            PlasmaLeft, PlasmaNoTrans, n, n, k, ib,
            A, lda, T, ldt, work, ldwork, work+nb*ldwork, ldwork);

        //===================================================
        // Copy back the final result to the upper part of C
        //==================================================
        LAPACKE_zlacpy_work(
            LAPACK_COL_MAJOR, lapack_const(PlasmaUpper),
            n, n, work, ldwork, C, ldc );
    }
    return PlasmaSuccess;
}

/******************************************************************************/
void plasma_core_omp_zherfb(
    plasma_enum_t uplo,
    int n, int k, int ib,
    const plasma_complex64_t *A, int lda,
    const plasma_complex64_t *T, int ldt,
          plasma_complex64_t *C, int ldc,
    plasma_workspace_t work,
    plasma_sequence_t *sequence, plasma_request_t *request)
{
    // OpenMP depends on lda == n == nb, ldc == nb, ldt == ib.
    #pragma omp task depend(in:A[0:lda*k]) \
                     depend(in:T[0:ib*k]) \
                     depend(inout:C[0:ldc*n])
    {
        if (sequence->status == PlasmaSuccess) {
            // Prepare workspaces.
            int tid = omp_get_thread_num();
            plasma_complex64_t *W = (plasma_complex64_t*)work.spaces[tid];

            int ldwork = n;

            // Call the kernel.
            int info = plasma_core_zherfb(
                uplo,
                n, k, ib,
                A, lda,
                T, ldt,
                C, ldc,
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
