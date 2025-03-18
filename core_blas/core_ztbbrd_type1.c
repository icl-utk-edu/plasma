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
#include "bulge.h"

#include <string.h>

#define AL( i_, j_ ) (A + nb + lda*(j_) + ((i_)-(j_)))
#define AU( i_, j_ ) (A + nb + lda*(j_) + ((i_)-(j_)+nb))
#define VQ( i_ )     (VQ + (i_))
#define VP( i_ )     (VP + (i_))
#define tauQ( i_ )   (tauQ + (i_))
#define tauP( i_ )   (tauP + (i_))

/***************************************************************************//**
 *
 * @ingroup core_tbbrd_type1
 *
 *  core_ztbbrd_type1 is a kernel that will operate on a region (triangle) of data
 *  bounded by start and end. This kernel eliminate a column by an column-wise
 *  annihiliation, then it apply a left+right update on the hermitian triangle.
 *  Note that the column to be eliminated is located at start-1.
 *
 *  All details are available in the technical report or SC11 paper.
 *  Azzam Haidar, Hatem Ltaief, and Jack Dongarra. 2011.
 *  Parallel reduction to condensed forms for symmetric eigenvalue problems
 *  using aggregated fine-grained and memory-aware kernels. In Proceedings
 *  of 2011 International Conference for High Performance Computing,
 *  Networking, Storage and Analysis (SC '11). ACM, New York, NY, USA,
 *  Article 8, 11 pages.
 *  http://doi.acm.org/10.1145/2063384.2063394
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of A is stored;
 *          - PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] n
 *          The order of the matrix A.
 *
 * @param[in] nb
 *          The size of the band.
 *
 * @param[in,out] A
 *          A pointer to the matrix A of size (3*nb + 1)-by-n.
 *
 * @param[in] lda
 *          The leading dimension of the matrix A. lda >= max( 1, 3*nb + 1 )
 *
 * @param[out] VP
 *          TODO: Check and fix doc
 *          PLASMA_Complex64_t array, dimension n if eigenvalue only
 *          requested or (LDV*blkcnt*Vblksiz) if Eigenvectors requested
 *          The Householder reflectors are stored in this array.
 *
 * @param[out] tauP
 *          TODO: Check and fix doc
 *          PLASMA_Complex64_t array, dimension (n).
 *          The scalar factors of the Householder reflectors are stored
 *          in this array.
 *
 * @param[out] VQ
 *          TODO: Check and fix doc
 *          PLASMA_Complex64_t array, dimension n if eigenvalue only
 *          requested or (LDV*blkcnt*Vblksiz) if Eigenvectors requested
 *          The Householder reflectors are stored in this array.
 *
 * @param[out] tauQ
 *          TODO: Check and fix doc
 *          PLASMA_Complex64_t array, dimension (n).
 *          The scalar factors of the Householder reflectors are stored
 *          in this array.
 *
 * @param[in] first
 *          The first index to update, after eliminating row or column first-1.
 *
 * @param[in] last
 *          The last index to update, inclusive.
 *
 * @param[in] sweep
 *          The sweep number that is eliminated. It serves to calculate the
 *          pointer to the position where to store the Vs and Ts.
 *
 * @param[in] Vblksiz
 *          Constant that corresponds to the blocking used when applying the Vs.
 *          It serves to calculate the pointer to the position where to store
 *          the Vs and Ts.
 *
 * @param[in] wantz
 *          Specifies whether only eigenvalues are requested or both
 *          eigenvalue and eigenvectors.
 *
 * @param[in] work
 *          Workspace of size nb.
 *
 *******************************************************************************
 *
 * @retval PLASMA_SUCCESS successful exit
 * @retval < 0 if -i, the i-th argument had an illegal value
 *
 ******************************************************************************/
void plasma_core_ztbbrd_type1(
    plasma_enum_t uplo, int n, int nb,
    plasma_complex64_t *A, int lda,
    plasma_complex64_t *VQ, plasma_complex64_t *tauQ,
    plasma_complex64_t *VP, plasma_complex64_t *tauP,
    int first, int last, int sweep, int Vblksiz, int wantz,
    plasma_complex64_t *work)
{
    plasma_complex64_t ctau;
    int i, len, ldx;
    int blkid, vpos, taupos, tpos;

    // Find the pointer to the Vs and Ts as stored by the bulge chasing.
    // Note that in case no singular vectors are required, V and T are stored
    // in a vector of size 2*n.
    if (wantz == 0) {
        vpos   = ((sweep + 1)%2)*n + first;
        taupos = ((sweep + 1)%2)*n + first;
    }
    else {
        findVTpos( n, nb, Vblksiz, sweep, first,
                   &vpos, &taupos, &tpos, &blkid );
    }

    ldx = lda - 1;
    len = last - first + 1;

    if (uplo == PlasmaUpper) {
        //========================
        //      UPPER CASE
        //========================
        // Eliminate the row A( first-1, first:last ) right of bidiagonal,
        // storing reflector Pi as a Householder vector in VP and tauP.
        // todo: why is this conj?
        *VP( vpos ) = 1.;
        for (i = 1; i < len; ++i) {
            *VP( vpos+i ) = conj( *AU( first-1, first+i ) );
            *AU( first-1, first+i ) = 0.;
        }
        *AU( first-1, first ) = conj( *AU( first-1, first ) );
        LAPACKE_zlarfg_work(
            len, AU( first-1, first ), VP( vpos+1 ), 1, tauP( taupos ) );

        // Apply Pi on right to A( first:last, first:last ).
        LAPACKE_zlarfx_work(
            LAPACK_COL_MAJOR, 'R', len, len,
            VP( vpos ), *tauP( taupos ), AU( first, first ), ldx, work );

        // Eliminate the created col A( first:last, first ) below diagonal,
        // storing reflector Qi as a Householder vector in VQ and tauQ.
        *VQ( vpos ) = 1.;
        memcpy( VQ( vpos+1 ), AU( first+1, first ), (len-1)*sizeof( plasma_complex64_t ) );
        memset( AU( first+1, first ), 0,            (len-1)*sizeof( plasma_complex64_t ) );
        LAPACKE_zlarfg_work(
            len, AU( first, first ), VQ( vpos+1 ), 1, tauQ( taupos ) );

        // Apply Qi on left to A( first:last, first+1:last ).
        ctau = conj( *tauQ( taupos ) );
        LAPACKE_zlarfx_work(
            LAPACK_COL_MAJOR, 'L', len, len-1,
            VQ( vpos ), ctau, AU( first, first+1 ), ldx, work );
    }
    else {
        //========================
        //      LOWER CASE
        //========================
        // For nb = 3:          Apply left      Apply right
        //        b             b               b
        // first: b b          <B> {B F F}      b <B 0 0>
        //        x b b     => <0> {B B F}   =>   {B B F}
        // last:  x x b b      <0> {X B B}        {X B B}
        //          x x b b         x x b b        X X B b
        //            x x b           x x b        F X X b
        //              x x             x x        F F X x

        // Eliminate the col A( first:last, first-1) below bidiagonal,
        // storing Householder vector in VQ.
        *VQ( vpos ) = 1.;
        memcpy( VQ( vpos+1 ), AL( first+1, first-1 ), (len-1)*sizeof( plasma_complex64_t ) );
        memset( AL( first+1, first-1 ), 0,            (len-1)*sizeof( plasma_complex64_t ) );
        LAPACKE_zlarfg_work(
            len, AL( first, first-1 ), VQ( vpos+1 ), 1, tauQ( taupos ) );

        // Apply left on A( first:last, first:last ).
        ctau = conj( *tauQ( taupos ) );
        LAPACKE_zlarfx_work(
            LAPACK_COL_MAJOR, 'L',
            len, len, VQ( vpos ), ctau, AL( first, first ), ldx, work );

        // Eliminate the created row A( first, first:last ) right of diagonal,
        // storing the Householder vector in VP.
        *VP( vpos ) = 1.;
        for (i = 1; i < len; ++i) {
            *VP( vpos+i ) = conj( *AL( first, first+i ) );
            *AL( first, first+i ) = 0.;
        }
        *AL( first, first ) = conj( *AL( first, first ) );
        LAPACKE_zlarfg_work(
            len, AL( first, first ), VP( vpos+1 ), 1, tauP( taupos ) );

        // Apply right on A( first+1:last, first:last ).
        LAPACKE_zlarfx_work(
            LAPACK_COL_MAJOR, 'R', len-1, len,
            VP( vpos ), *tauP( taupos ), AL( first+1, first ), ldx, work );
    }
}

#undef AU
#undef AL
#undef VQ
#undef VP
#undef tauQ
#undef tauP
