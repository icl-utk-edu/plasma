/**
 *
 * @file core_zgetrf_tile.c
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> c d s
 *
 **/

#include "core_blas.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_types.h"

#ifdef PLASMA_WITH_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

#define KUT ((A.ku+A.kl+A.nb-1)/A.nb)
#define   A(m,n) ((PLASMA_Complex64_t*)plasma_getaddr(A, KUT+((m)-(n)), (n)))
#define lda(m,n)  BLKLDD((A), KUT+(m)-(n))

/***************************************************************************//**
 *
 * @ingroup core_getrf_tile
 *
 *  Performs the Cholesky factorization of a Hermitian positive definite
 *  matrix A. The factorization has the form
 *
 *    \f[ A = L \times L^H, \f]
 *    or
 *    \f[ A = U^H \times U, \f]
 *
 *  where U is an upper triangular matrix and L is a lower triangular matrix.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of A is stored;
 *          - PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in,out] A
 *          On entry, the Hermitian positive definite matrix A.
 *          If uplo = PlasmaUpper, the leading N-by-N upper triangular part of A
 *          contains the upper triangular part of the matrix A, and the strictly lower triangular
 *          part of A is not referenced.
 *          If uplo = 'L', the leading N-by-N lower triangular part of A contains the lower
 *          triangular part of the matrix A, and the strictly upper triangular part of A is not
 *          referenced.
 *          On exit, if return value = 0, the factor U or L from the Cholesky factorization
 *          A = U^H*U or A = L*L^H.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,n).
 *
 ******************************************************************************/
void CORE_zgetrf_tile(PLASMA_desc A, 
                      int k, int m, int n,
                      int *ipiv,
                      int ku, int kn, int *prev_fill, int *fill,
                      int iinfo)
{
    int ii;
    PLASMA_Complex64_t *WORK = (PLASMA_Complex64_t*)malloc(m*n * sizeof(PLASMA_Complex64_t));
    /* Copy tiles into workspace */
    for (ii=0; ii<(m+A.nb-1)/A.nb; ii++) {
        int nb = imin(A.nb, m-A.nb*ii);
        LAPACKE_zlacpy_work(LAPACK_COL_MAJOR, 'F', nb, n, A(k+ii, k), lda(k+ii, k), &WORK[ii*A.nb], m);
    }

    /* Do LU in workspace */
    int info = LAPACKE_zgetrf_work(LAPACK_COL_MAJOR, m, n, WORK, m, ipiv );
    if (info != 0) printf( " zgetrf failed with info=%d\n",info );

    /* Copy result back into tiles */
    for (ii=0; ii<(m+A.nb-1)/A.nb; ii++) {
        int nb = imin(A.nb, m-A.nb*ii);
        LAPACKE_zlacpy_work(LAPACK_COL_MAJOR, 'F', nb, n, &WORK[ii*A.nb], m, A(k+ii, k), lda(k+ii, k));
    }

    /* Update fill */
    if (info == PLASMA_SUCCESS) {
        int mn = imin(m,n);
        fill[0] = imin(ipiv[0]-1,kn-1); // zero base
        for(ii=1; ii<mn; ii++) {
            fill[0] = imax(fill[0], imin(ii+ipiv[ii]-1,kn-1));
        }
        fill[0] = (k*A.nb+ku+fill[0])/A.nb;
        if (k > 0) {
            fill[0] = imax(fill[0], prev_fill[0]);
        }
    } else {
        if (info > 0) {
            for(ii=info-1; ii<imin(m,n); ii++)
                ipiv[ii] = ii+1;
        }
        //plasma_sequence_flush(quark, sequence, request, iinfo+info);
    }

    free(WORK);
}

/******************************************************************************/
void CORE_OMP_zgetrf_tile(PLASMA_desc A,
                          int k, int m, int n,
                          int *ipiv,
                          int ku, int kn, int *prev_fill, int *fill,
                          int iinfo, int *fake)
{
    // omp depends assume lda = n.
    PLASMA_Complex64_t *Akk = A(k,k);
    #pragma omp task depend(inout:Akk[0:m*n]) depend(out:fake[0])
    CORE_zgetrf_tile(A, k, m, n, ipiv,
                     ku, kn, prev_fill, fill,
                     iinfo);
}
