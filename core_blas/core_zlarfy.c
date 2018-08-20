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
#include "core_lapack.h"

#undef REAL
#define COMPLEX

/***************************************************************************//**
 *
 * @ingroup CORE_plasma_complex64_t
 *
 *  CORE_zlarfy applies an elementary reflector, or Householder matrix, H,
 *  to a N-by-N hermitian matrix C, from both the left and the right.
 *
 *  H is represented in the form
 *
 *     H = I - tau * v * v'
 *
 *  where  tau  is a scalar and  v  is a vector.
 *
 *  If tau is zero, then H is taken to be the unit matrix.
 *
 *******************************************************************************
 *
 * @param[in] N
 *          The number of rows and columns of the matrix C.  N >= 0.
 *
 * @param[in,out] A
 *          COMPLEX*16 array, dimension (LDA, N)
 *          On entry, the Hermetian matrix A.
 *          On exit, A is overwritten by H * A * H'.
 *
 * @param[in] LDA
 *         The leading dimension of the array A.  LDA >= max(1,N).
 *
 * @param[in] V
 *          The vector V that contains the Householder reflectors.
 *
 * @param[in] TAU
 *          The value tau.
 *
 * @param[out] WORK
 *          Workspace.
 *
 ******************************************************************************/
void core_zlarfy(int N,
            plasma_complex64_t *A, int LDA,
            const plasma_complex64_t *V,
            const plasma_complex64_t *TAU,
            plasma_complex64_t *WORK)
{
    static plasma_complex64_t zzero =  0.0;
    static plasma_complex64_t zmone = -1.0;

    int j;
    plasma_complex64_t dtmp;

    /* Compute dtmp = X'*V */
    /* X = AVtau */
    cblas_zhemv(CblasColMajor, CblasLower,
                N, CBLAS_SADDR(*TAU), A, LDA,
                V, 1, CBLAS_SADDR(zzero), WORK, 1);

    /* cblas_zdotc_sub(N, WORK, 1, V, 1, &dtmp);*/
    dtmp = 0.;
    for (j = 0; j < N ; j++)
        dtmp = dtmp + conj(WORK[j]) * V[j];

    /* Compute 1/2 X'*V*t = 1/2*dtmp*tau  */
    dtmp = -dtmp * 0.5 * (*TAU);

   /* Compute W=X-1/2VX'Vt = X - dtmp*V */
    cblas_zaxpy(N, CBLAS_SADDR(dtmp),
                V, 1, WORK, 1);

    /*
     * Performs the symmetric rank 2 operation
     *    A := alpha*x*y' + alpha*y*x' + A
     */
    cblas_zher2(CblasColMajor, CblasLower, N,
                CBLAS_SADDR(zmone), WORK, 1,
                                    V,    1,
                                    A,    LDA);

    return;
}
#undef COMPLEX
