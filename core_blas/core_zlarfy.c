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
#include "core_lapack.h"

#undef REAL
#define COMPLEX

/***************************************************************************//**
 *
 * @ingroup CORE_plasma_complex64_t
 *
 *  CORE_zlarfy applies an elementary reflector, or Householder matrix, H,
 *  to a n-by-n Hermitian matrix C, from both the left and the right.
 *
 *  H is represented in the form
 *
 *     H = I - tau v v^H
 *
 *  where tau is a scalar and v is a vector.
 *
 *  If tau is zero, then H is taken to be the unit matrix.
 *
 *******************************************************************************
 *
 * @param[in] n
 *          The number of rows and columns of the matrix C. n >= 0.
 *
 * @param[in,out] A
 *          COMPLEX*16 array, dimension (lda, n)
 *          On entry, the Hermetian matrix A.
 *          On exit, A is overwritten by H A H^H.
 *
 * @param[in] lda
 *         The leading dimension of the array A. lda >= max(1,n).
 *
 * @param[in] v
 *          The vector v that contains the Householder reflectors.
 *
 * @param[in] tau
 *          The value tau.
 *
 * @param[out] work
 *          Workspace of size n.
 *
 ******************************************************************************/
void plasma_core_zlarfy(
    int n,
    plasma_complex64_t *A, int lda,
    const plasma_complex64_t *v,
    const plasma_complex64_t *tau,
    plasma_complex64_t *work)
{
    const plasma_complex64_t zzero =  0.0;
    const plasma_complex64_t zmone = -1.0;

    int j;
    plasma_complex64_t dtmp;

    // Compute dtmp = x^H v
    // x = A v tau
    cblas_zhemv(CblasColMajor, CblasLower,
                n, CBLAS_SADDR(*tau), A, lda,
                v, 1, CBLAS_SADDR(zzero), work, 1);

    // cblas_zdotc_sub(n, work, 1, v, 1, &dtmp);
    dtmp = 0.;
    for (j = 0; j < n; ++j)
        dtmp = dtmp + conj(work[j]) * v[j];

    // Compute 1/2 x^H v tau = 1/2 dtmp tau
    dtmp = -dtmp * 0.5 * (*tau);

    // Compute w = x - 1/2 v x^H v t = x - dtmp v */
    cblas_zaxpy(n, CBLAS_SADDR(dtmp),
                v, 1, work, 1);

    // Performs the symmetric rank 2 operation
    // A := alpha x y^H + alpha y x^H + A
    cblas_zher2(CblasColMajor, CblasLower, n,
                CBLAS_SADDR(zmone), work, 1,
                                    v,    1,
                                    A,    lda);
}
#undef COMPLEX
