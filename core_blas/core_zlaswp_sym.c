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
#include "plasma_internal.h"
#include "core_lapack.h"

#define A(m,n) ((plasma_complex64_t*)plasma_tile_addr(A, m, n))
#define W2(j)  ((plasma_complex64_t*)plasma_tile_addr(W, j+A.mt, 0))   // 2mt x nb*nb

/***************************************************************************//**
 *
 * @ingroup core_lacpy
 *
 *  Copies all or part of a two-dimensional matrix A to another matrix B.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaGeneral: entire A,
 *          - PlasmaUpper:   upper triangle,
 *          - PlasmaLower:   lower triangle.
 *
 * @param[in] m
 *          The number of rows of the matrices A and B.
 *          m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrices A and B.
 *          n >= 0.
 *
 * @param[in] A
 *          The m-by-n matrix to copy.
 *
 * @param[in] lda
 *          The leading dimension of the array A.
 *          lda >= max(1,m).
 *
 * @param[out] B
 *          The m-by-n copy of the matrix A.
 *          On exit, B = A ONLY in the locations specified by uplo.
 *
 * @param[in] ldb
 *          The leading dimension of the array B.
 *          ldb >= max(1,m).
 *
 ******************************************************************************/
void core_zhetrf_pivot_out(int k, int i1, int nn, int uplo,
                           int ib,
                           plasma_desc_t A,
                           plasma_complex64_t *B, int ldb,
                           int *iperm, int *iperm2work) {

    plasma_complex64_t *Bi;
    int i, j, ii, kk, ldan, tempn;
    int *iperm_i = &iperm2work[i1*A.mb];

    if (uplo == PlasmaLower) {

        /* copy the offdiagonal in i1-th row */
        for( j=k; j<i1; j++ ) {
            iperm_i = &iperm2work[j*A.mb];
            ldan = plasma_tile_mmain(A, i1);
            tempn = j == A.mt-1 ? A.m-j*A.mb : A.mb;
            for( i=0; i<tempn; i++ ) {
                if( iperm_i[i] >= 0 ) {
                    ii = i1*A.mb;
                    Bi = &B[iperm_i[i]*ldb-k*A.mb];

                    plasma_complex64_t *Aij = A(i1, j);
                    for( kk=0; kk<nn; kk++ ) Bi[iperm[ii+kk]] = Aij[kk + i*ldan];
                }
            }
        }

        /* copy the diagonal */
        iperm_i = &iperm2work[i1*A.mb];
        ldan = plasma_tile_mmain(A, i1);
        for( i=0; i<nn; i++ ) {
            if( iperm_i[i] >= 0 ) {
                ii = i1*A.mb;
                Bi = &B[iperm_i[i]*ldb-k*A.mb];

                plasma_complex64_t *Aij = A(i1, j);
                for( kk=0; kk<i; kk++ ) Bi[iperm[ii+kk]] = conj(Aij[i+ kk*ldan]);
                for( kk=i; kk<nn; kk++ ) Bi[iperm[ii+kk]] = Aij[kk+i*ldan];
            }
        }
        /* copy the offdiagonal in i1-th col */
        for( j=i1+1; j<A.mt; j++ ) {
            iperm_i = &iperm2work[j*A.mb];
            ldan = plasma_tile_mmain(A, j);
            tempn = j == A.mt-1 ? A.m-j*A.mb : A.mb;
            for( i=0; i<tempn; i++ ) {
                if( iperm_i[i] >= 0 ) {
                    ii = i1*A.mb;
                    Bi = &B[iperm_i[i]*ldb-k*A.mb];

                    plasma_complex64_t *Aij = A(j, i1);
                    for( kk=0; kk<nn; kk++ ) Bi[iperm[ii+kk]] = conj(Aij[i+kk*ldan]);
                }
            }
        }
    } else {
    }
}

void core_zhetrf_pivot_in(int k, int i1, int nn, int uplo, int ib,
                          plasma_desc_t A,
                          plasma_complex64_t *B, int ldb,
                          int *perm2work ) {

    plasma_complex64_t *Bi;
    int i, j, ii, kk, kkk, ldan, tempn;
    int *perm2work_i = &perm2work[i1*A.mb];

    if( uplo == PlasmaLower ) {
        /* copy the offdiagonal in i1-th row */
        for( j=k; j<i1; j++ ) {
            /* column swap */
            perm2work_i = &perm2work[j*A.mb];
            tempn = j == A.mt-1 ? A.m-j*A.mb : A.mb;
            for( i=0; i<tempn; i++ ) {
                if( perm2work_i[i] >= 0 ) { // this is a pivot column
                    ii = (i1-k)*A.mb;
                    Bi = &B[perm2work_i[i]*ldb];

                    ldan = plasma_tile_mmain(A, i1);
                    plasma_complex64_t *Aij = A(i1, j);
                    for( kk=0; kk<nn; kk++ ) Aij[kk + i*ldan] = Bi[ii+kk];
                }
            }

            /* row swap,  trying inner-blocking */
            perm2work_i = &perm2work[i1*A.mb];
            ldan = plasma_tile_mmain(A, i1);
            tempn = j == A.mt-1 ? A.m-j*A.mb : A.mb;
            for( i=0; i<nn; ) {
                int istart = i;
                int iend   = imin(nn, i+ib);
                for( i=istart; i<iend; i++ ) {
                    if( perm2work_i[i] >= 0 ) { // this is a pivot row
                        ii = (j-k)*A.mb;
                        Bi = &B[perm2work_i[i]*ldb];

                        plasma_complex64_t *Aij = A(i1, j);
                        for( kk=0; kk<tempn; kk+=ib ) {
                            for( kkk=kk; kkk<imin(kk+ib, tempn); kkk++ ) {
                                Aij[i + kkk*ldan] = conj(Bi[ii+kkk]);
                            }
                        }
                    }
                }
            }
        }
        /* copy the diagonal */
        perm2work_i = &perm2work[i1*A.mb];
        for( i=0; i<nn; i++ ) {
            if( perm2work_i[i] >= 0 ) {
                ii = i1*A.mb;
                Bi = &B[perm2work_i[i]*ldb-k*A.mb];

                ldan = plasma_tile_mmain(A, i1);
                tempn = i1 == A.mt-1 ? A.m-i1*A.mb : A.mb;
                plasma_complex64_t *Aij = A(i1, i1);
                for( kk=0; kk<i; kk++ ) Aij[i + kk*ldan] = conj(Bi[ii+kk]);
                for( kk=i; kk<nn; kk++ ) Aij[kk + i*ldan] = Bi[ii+kk];
            }
        }
    } else {
    }
}
/******************************************************************************/
void core_omp_zlaswp_sym(plasma_enum_t uplo, int k, int tempm, int mvak, int ib, 
                         plasma_desc_t A, plasma_desc_t W,
                         int *ipiv, int *perm,
                         int *iperm, int *iperm2work, int *perm2work,
                         plasma_sequence_t *sequence, plasma_request_t *request)
{
    {
        if (sequence->status == PlasmaSuccess) {
            /* initialize permutation matrices */
            int tempi = (k+1)*A.mb;
            for (int i=0; i<tempm; i++) perm[tempi+i] = tempi+i;
            for (int i=0; i<imin(tempm, mvak); i++) {
                int piv = perm[ipiv[i]-1];
                perm[ipiv[i]-1] = perm[tempi+i];
                perm[tempi+i] = piv;
            }
            int npiv = 0;
            for (int i=0; i<tempm; i++) {
                if( perm[tempi+i] != tempi+i ) {
                    perm2work[tempi+i] = npiv;
                    npiv ++;
                } else {
                    perm2work[tempi+i] = -1;
                }
                iperm[perm[tempi+i]] = tempi+i;
            }
            for (int i=0; i<tempm; i++) iperm2work[tempi+i] = perm2work[iperm[tempi+i]];

            /* copy pivots columns into workspace */
            for (int n=k+1; n<A.nt; n ++) {
                int tempn = n == A.nt-1 ? A.n-n*A.nb : A.nb;
                core_zhetrf_pivot_out( k+1, n, tempn, PlasmaLower,
                                       ib,
                                       A,
                                       W2(0), A.n,
                                       iperm, iperm2work);
            }
            /* copy back pivots columns into A */
            for (int n=k+1; n<A.nt; n ++) {
                int tempn = n == A.nt-1 ? A.n-n*A.nb : A.nb;
                core_zhetrf_pivot_in(k+1, n, tempn, PlasmaLower,
                                     ib,
                                     A,
                                     W2(0), A.n,
                                     perm2work);
            }
        }
    }
}
