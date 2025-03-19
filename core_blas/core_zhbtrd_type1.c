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
#include "bulge.h"

#include <string.h>

#define A( i_, j_ ) (A + lda*(j_) + ((i_) - (j_)))
#define V( i_ )     (V + (i_))
#define tau( i_ )   (tau + (i_))

/***************************************************************************//**
 *
 * @ingroup core_hbtrd_type1
 *
 *  First kernel for each sweep in the Hermitian eigenvalue
 *  bulge-chasing algorithm, which updates the first diagonal tile.
 *  Eliminates entries below the sub-diagonal of column first-1 using a
 *  Householder reflector. Then applies the reflector on the left and
 *  right to update the Hermitian matrix, represented by a lower
 *  triangular region bounded by [first, last] inclusive.
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
 * @param[in] n
 *          The order of the matrix A.
 *
 * @param[in] nb
 *          The size of the band.
 *
 * @param[in,out] A
 *          A pointer to the matrix A of size (2*nb + 1)-by-n.
 *
 * @param[in] lda
 *          The leading dimension of the matrix A. lda >= max( 1, 2*nb + 1 )
 *
 * @param[out] V
 *          Array of dimension 2*n if only eigenvalues are requested (wantz = 0),
 *          or (ldv*blkcnt*Vblksiz) if eigenvectors are requested (wantz != 0).
 *          Stores the Householder vectors.
 *          Adds one Householder vector to eliminate a column of the band.
 *
 * @param[out] tau
 *          Array of dimension 2*n.
 *          Stores the scalar factors of the Householder reflectors.
 *          Adds one scalar factor.
 *
 * @param[in] first
 *          The first index to update, after eliminating column first-1.
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
 ******************************************************************************/
void plasma_core_zhbtrd_type1(
    int n, int nb,
    plasma_complex64_t *A, int lda,
    plasma_complex64_t *V, plasma_complex64_t *tau,
    int first, int last, int sweep, int Vblksiz, int wantz,
    plasma_complex64_t *work)
{
    int len, ldx;
    int blkid, vpos, taupos, tpos;

    // Find the pointer to the Vs and Ts as stored by the bulge chasing.
    // Note that in case no eigenvectors are required, V and T are stored
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
    *V( vpos ) = 1.;

    assert( len > 0 );

    memcpy( V( vpos+1 ), A( first+1, first-1 ), (len-1)*sizeof(plasma_complex64_t) );
    memset( A( first+1, first-1 ), 0,           (len-1)*sizeof(plasma_complex64_t) );

    // Eliminate the col at first-1.
    LAPACKE_zlarfg_work( len, A( first, first-1 ),
                         V( vpos+1 ), 1, tau( taupos ) );

    // Apply left and right on A( first:last, first:last ).
    plasma_core_zlarfy( len, A( first, first ), ldx,
                        V( vpos ), tau( taupos ), work );
}

#undef A
#undef V
#undef tau
