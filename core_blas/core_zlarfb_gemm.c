/**
 *
 * @file
 *
 *  plasma is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> c d s
 *
 **/

#include "plasma_core_blas.h"
#include "plasma_types.h"
#include "core_lapack.h"

/***************************************************************************//**
 *
 * @ingroup CORE_plasma_Complex64_t
 *
 *  CORE_zlarfb_gemm applies a complex block reflector H or its transpose H'
 *  to a complex M-by-N matrix C, from either the left or the right.
 *  this kernel is similar to the lapack zlarfb but it do a full gemm on the
 *  triangular Vs assuming that the upper part of Vs is zero and ones are on
 *  the diagonal. It is also based on the fact that a gemm on a small block of
 *  k reflectors is faster than a trmm on the triangular (k,k) + gemm below.
 *
 *  NOTE THAT:
 *  Only Columnwise/Forward cases are treated here.
 *
 *******************************************************************************
 *
 * @param[in] side
 *         @arg PlasmaLeft  : apply Q or Q**H from the Left;
 *         @arg PlasmaRight : apply Q or Q**H from the Right.
 *
 * @param[in] trans
 *         @arg PlasmaNoTrans   : No transpose, apply Q;
 *         @arg PlasmaConjTrans : ConjTranspose, apply Q**H.
 *
 * @param[in] direct
 *         Indicates how H is formed from a product of elementary
 *         reflectors
 *         @arg PlasmaForward  : H = H(1) H(2) . . . H(k) (Forward)
 *         @arg PlasmaBackward : H = H(k) . . . H(2) H(1) (Backward)
 *
 * @param[in] storev
 *         Indicates how the vectors which define the elementary
 *         reflectors are stored:
 *         @arg PlasmaColumnwise
 *         @arg PlasmaRowwise
 *
 * @param[in] M
 *         The number of rows of the matrix C.
 *
 * @param[in] N
 *         The number of columns of the matrix C.
 *
 * @param[in] K
 *          The order of the matrix T (= the number of elementary
 *          reflectors whose product defines the block reflector).
 *
 * @param[in] V
 *          COMPLEX*16 array, dimension
 *              (LDV,K) if storev = 'C'
 *              (LDV,M) if storev = 'R' and side = 'L'
 *              (LDV,N) if storev = 'R' and side = 'R'
 *          The matrix V. See further details.
 *
 * @param[in] LDV
 *         The leading dimension of the array V.
 *         If storev = 'C' and side = 'L', LDV >= max(1,M);
 *         if storev = 'C' and side = 'R', LDV >= max(1,N);
 *         if storev = 'R', LDV >= K.
 *
 * @param[in] T
 *         The triangular K-by-K matrix T in the representation of the
 *         block reflector.
 *         T is upper triangular by block (economic storage);
 *         The rest of the array is not referenced.
 *
 * @param[in] LDT
 *         The leading dimension of the array T. LDT >= K.
 *
 * @param[in,out] C
 *         COMPLEX*16 array, dimension (LDC,N)
 *         On entry, the M-by-N matrix C.
 *         On exit, C is overwritten by H*C or H'*C or C*H or C*H'.
 *
 * @param[in] LDC
 *         The leading dimension of the array C. LDC >= max(1,M).
 *
 * @param[in,out] WORK
 *         (workspace) COMPLEX*16 array, dimension (LDWORK,K).
 *
 * @param[in] LDWORK
 *         The dimension of the array WORK.
 *         If side = PlasmaLeft,  LDWORK >= max(1,N);
 *         if side = PlasmaRight, LDWORK >= max(1,M).
 *
 *******************************************************************************
 *
 * @return
 *          \retval plasma_SUCCESS successful exit
 *          \retval <0 if -i, the i-th argument had an illegal value
 *
 ******************************************************************************/
#if defined(plasma_HAVE_WEAK)
#pragma weak CORE_zlarfb_gemm = PCORE_zlarfb_gemm
#define CORE_zlarfb_gemm PCORE_zlarfb_gemm
#endif
int plasma_core_zlarfb_gemm(plasma_enum_t side, plasma_enum_t trans, int direct, int storev,
                     int M, int N, int K,
                     const plasma_complex64_t *V, int LDV,
                     const plasma_complex64_t *T, int LDT,
                     plasma_complex64_t *C, int LDC,
                     plasma_complex64_t *WORK, int LDWORK)
{
    static plasma_complex64_t zzero =  0.0;
    static plasma_complex64_t zone  =  1.0;
    static plasma_complex64_t mzone = -1.0;

    // Quick return //
    if ((M == 0) || (N == 0) || (K == 0) )
        return PlasmaSuccess;

    /* For Left case, switch the trans. noswitch for right case */
    if( side == PlasmaLeft){
        if ( trans == PlasmaNoTrans) {
            trans = PlasmaConjTrans;
        }
        else {
            trans = PlasmaNoTrans;
        }
    }

    // main code //
    if (storev == PlasmaColumnwise )
    {
        if ( direct == PlasmaForward )
        {
            /*
             * Let  V =  ( V1 )    (first K rows are unit Lower triangular)
             *           ( V2 )
             */
            if ( side == PlasmaLeft )
            {
                /*
                 * Columnwise / Forward / Left
                 */
                /*
                 * Form  H * C  or  H' * C  where  C = ( C1 )
                 *                                     ( C2 )
                 *
                 * W := C' * V    (stored in WORK)
                 */
                cblas_zgemm(
                    CblasColMajor, CblasConjTrans, CblasNoTrans,
                    N, K, M,
                    CBLAS_SADDR(zone), C, LDC,
                    V, LDV,
                    CBLAS_SADDR(zzero), WORK, LDWORK );
                /*
                 * W := W * T'  or  W * T
                 */
                cblas_ztrmm(
                    CblasColMajor, CblasRight, CblasUpper,
                    (CBLAS_TRANSPOSE)trans, CblasNonUnit, N, K,
                    CBLAS_SADDR(zone), T, LDT, WORK, LDWORK );
                /*
                 * C := C - V * W'
                 */
                cblas_zgemm(
                    CblasColMajor, CblasNoTrans, CblasConjTrans,
                    M, N, K,
                    CBLAS_SADDR(mzone), V, LDV,
                    WORK, LDWORK,
                    CBLAS_SADDR(zone), C, LDC);
            }
            else {
                /*
                 * Columnwise / Forward / Right
                 */
                /*
                 * Form  C * H  or  C * H'  where  C = ( C1  C2 )
                 * W := C * V
                 */
                cblas_zgemm(
                    CblasColMajor, CblasNoTrans, CblasNoTrans,
                    M, K, N,
                    CBLAS_SADDR(zone), C, LDC,
                    V, LDV,
                    CBLAS_SADDR(zzero), WORK, LDWORK);
                /*
                 * W := W * T  or  W * T'
                 */
                cblas_ztrmm(
                    CblasColMajor, CblasRight, CblasUpper,
                    (CBLAS_TRANSPOSE)trans, CblasNonUnit, M, K,
                    CBLAS_SADDR(zone), T, LDT, WORK, LDWORK );
                /*
                 * C := C - W * V'
                 */
                cblas_zgemm(
                    CblasColMajor, CblasNoTrans, CblasConjTrans,
                    M, N, K,
                    CBLAS_SADDR(mzone), WORK, LDWORK,
                    V, LDV,
                    CBLAS_SADDR(zone), C, LDC);
            }
        }
        else {
            /*
             * Columnwise / Backward / Left or Right
             */
            //coreblas_error(3, "Not implemented (ColMajor / Backward / Left or Right)");
            return PlasmaErrorNotSupported;
        }
    }
    else {
        if (direct == PlasmaForward) {
            /*
             * Rowwise / Forward / Left or Right
             */
            //coreblas_error(3, "Not implemented (RowMajor / Backward / Left or Right)");
            return PlasmaErrorNotSupported;;
        }
        else {
            /*
             * Rowwise / Backward / Left or Right
             */
            //coreblas_error(3, "Not implemented (RowMajor / Backward / Left or Right)");
            return PlasmaErrorNotSupported;
        }
    }
    return PlasmaSuccess;
}
