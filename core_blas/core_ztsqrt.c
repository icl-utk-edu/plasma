/**
 *
 * @file core_ztsqrt.c
 *
 *  PLASMA core_blas kernel
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver and
 *  Univ. of Manchester.
 *
 * @version 3.0.0
 * @author Hatem Ltaief
 * @author Mathieu Faverge
 * @author Jakub Kurzak
 * @author Jakub Sistek
 * @date 2016-7-8
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

// this will be swapped during the automatic code generation
#undef REAL
#define COMPLEX

/***************************************************************************//**
 *
 * @ingroup CORE_PLASMA_Complex64_t
 *
 * Computes a QR factorization of a rectangular matrix
 * formed by coupling an n-by-n upper triangular tile A1
 * on top of an m-by-n tile A2:
 *
 *    | A1 | = Q * R
 *    | A2 |
 *
 *******************************************************************************
 *
 * @param[in] m
 *         The number of columns of the tile A2. m >= 0.
 *
 * @param[in] n
 *         The number of rows of the tile A1.
 *         The number of columns of the tiles A1 and A2. n >= 0.
 *
 * @param[in] ib
 *         The inner-blocking size.  ib >= 0.
 *
 * @param[in,out] A1
 *         On entry, the n-by-n tile A1.
 *         On exit, the elements on and above the diagonal of the array
 *         contain the n-by-n upper trapezoidal tile R;
 *         the elements below the diagonal are not referenced.
 *
 * @param[in] lda1
 *         The leading dimension of the array A1. LDA1 >= max(1,N).
 *
 * @param[in,out] A2
 *         On entry, the m-by-n tile A2.
 *         On exit, all the elements with the array TAU, represent
 *         the unitary tile Q as a product of elementary reflectors
 *         (see Further Details).
 *
 * @param[in] lda2
 *         The leading dimension of the tile A2. lda2 >= max(1,m).
 *
 * @param[out] T
 *         The ib-by-n triangular factor T of the block reflector.
 *         T is upper triangular by block (economic storage);
 *         The rest of the array is not referenced.
 *
 * @param[in] ldt
 *         The leading dimension of the array T. ldt >= ib.
 *
 ******************************************************************************/
void CORE_ztsqrt(int m, int n, int ib, 
                 PLASMA_Complex64_t *A1, int lda1,
                 PLASMA_Complex64_t *A2, int lda2,
                 PLASMA_Complex64_t *T, int ldt)
{
    // block size is assumed to be equal to n
    int nb = n;

    // Check input arguments
    if (m < 0) {
        plasma_error("Illegal value of m");
        return;
    }
    if (n < 0) {
        plasma_error("Illegal value of n");
        return;
    }
    if (ib < 0) {
        plasma_error("Illegal value of ib");
        return;
    }
    if ((lda2 < imax(1,m)) && (m > 0)) {
        plasma_error("Illegal value of lda2");
        return;
    }

    // Quick return
    if ((m == 0) || (n == 0) || (ib == 0))
        return;

    // prepare memory for auxiliary arrays
    int ltau  = nb;
    PLASMA_Complex64_t *TAU  = 
        (PLASMA_Complex64_t *) malloc(sizeof(PLASMA_Complex64_t) * ltau);
    if (TAU == NULL) {
        plasma_error("malloc() failed");
        return;
    }
    int lwork = ib*nb;
    PLASMA_Complex64_t *WORK = 
        (PLASMA_Complex64_t *) malloc(sizeof(PLASMA_Complex64_t) * lwork);
    if (WORK == NULL) {
        plasma_error("malloc() failed");
        return;
    }

    // variable storing 1 and 0 constants
    static PLASMA_Complex64_t zone  = 1.0;
    static PLASMA_Complex64_t zzero = 0.0;

    PLASMA_Complex64_t alpha;
    int i, ii, sb;

    for(ii = 0; ii < n; ii += ib) {
        sb = imin(n-ii, ib);
        for(i = 0; i < sb; i++) {
            // Generate elementary reflector H( II*IB+I ) to annihilate 
            // A( II*IB+I:M, II*IB+I )
            LAPACKE_zlarfg_work(m+1, &A1[lda1*(ii+i)+ii+i], &A2[lda2*(ii+i)], 1,
                                &TAU[ii+i]);

            if (ii+i+1 < n) {
                // Apply H( II*IB+I ) to A( II*IB+I:M, II*IB+I+1:II*IB+IB ) 
                // from the left
                alpha = -conj(TAU[ii+i]);
                cblas_zcopy(
                    sb-i-1,
                    &A1[lda1*(ii+i+1)+(ii+i)], lda1,
                    WORK, 1);
#ifdef COMPLEX
                LAPACKE_zlacgv_work(sb-i-1, WORK, 1);
#endif
                // Plasma_ConjTrans will be converted do PlasmaTrans in 
                // automatic datatype conversion, which is what we want here.
                // PlasmaConjTrans is protected from this conversion.
                cblas_zgemv(
                    CblasColMajor, (CBLAS_TRANSPOSE)Plasma_ConjTrans,
                    m, sb-i-1,
                    CBLAS_SADDR(zone), &A2[lda2*(ii+i+1)], lda2,
                    &A2[lda2*(ii+i)], 1,
                    CBLAS_SADDR(zone), WORK, 1);
#ifdef COMPLEX
                LAPACKE_zlacgv_work(sb-i-1, WORK, 1 );
#endif
                cblas_zaxpy(
                    sb-i-1, CBLAS_SADDR(alpha),
                    WORK, 1,
                    &A1[lda1*(ii+i+1)+ii+i], lda1);
#ifdef COMPLEX
                LAPACKE_zlacgv_work(sb-i-1, WORK, 1 );
#endif
                cblas_zgerc(
                    CblasColMajor, m, sb-i-1, CBLAS_SADDR(alpha),
                    &A2[lda2*(ii+i)], 1,
                    WORK, 1,
                    &A2[lda2*(ii+i+1)], lda2);
            }
            // Calculate T
            alpha = -TAU[ii+i];
            cblas_zgemv(
                CblasColMajor, (CBLAS_TRANSPOSE)Plasma_ConjTrans, m, i,
                CBLAS_SADDR(alpha), &A2[lda2*ii], lda2,
                &A2[lda2*(ii+i)], 1,
                CBLAS_SADDR(zzero), &T[ldt*(ii+i)], 1);

            cblas_ztrmv(
                CblasColMajor, (CBLAS_UPLO)PlasmaUpper,
                (CBLAS_TRANSPOSE)PlasmaNoTrans, (CBLAS_DIAG)PlasmaNonUnit, i,
                &T[ldt*ii], ldt,
                &T[ldt*(ii+i)], 1);

            T[ldt*(ii+i)+i] = TAU[ii+i];
        }
        if (n > ii+sb) {
            CORE_ztsmqr(
                PlasmaLeft, Plasma_ConjTrans,
                sb, n-(ii+sb), m, n-(ii+sb), ib, ib,
                &A1[lda1*(ii+sb)+ii], lda1,
                &A2[lda2*(ii+sb)], lda2,
                &A2[lda2*ii], lda2,
                &T[ldt*ii], ldt);
                //WORK, sb);
        }
    }

    // deallocate auxiliary arrays
    free(TAU);
    free(WORK);
}

/******************************************************************************/
void CORE_OMP_ztsqrt(int m, int n, int ib, int nb,
                     PLASMA_Complex64_t *A1, int lda1,
                     PLASMA_Complex64_t *A2, int lda2,
                     PLASMA_Complex64_t *T,  int ldt)
{
    // assuming m == nb, n == nb
    #pragma omp task depend(inout:A1[0:nb*nb]) \
                     depend(inout:A2[0:nb*nb]) \
                     depend(out:T[0:ib*nb])
    CORE_ztsqrt(m, n, ib, 
                A1, lda1,
                A2, lda2,
                T,  ldt);
}
