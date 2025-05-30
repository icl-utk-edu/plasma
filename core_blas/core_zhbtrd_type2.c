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

#define A( i_, j_ ) (A + lda*(j_) + ((i_) - (j_)))
#define V( i_ )     (V + (i_))
#define tau( i_ )   (tau + (i_))

/***************************************************************************//**
 *
 * @ingroup core_hbtrd_type2
 *
 *  Updates the off-diagonal tiles in the Hermitian eigenvalue
 *  bulge-chasing algorithm. Applies the reflector from the previous
 *  type 1 or 2 kernel on the right to update an off-diagonal tile,
 *  represented by an upper triangular region bounded by [first, last] inclusive,
 *  to create a bulge. Then eliminates entries below the band in column first
 *  using a Householder reflector, and applies the reflector on the left.
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
 * @param[in,out] V
 *          Array of dimension 2*n if only eigenvalues are requested (wantz = 0),
 *          or (ldv*blkcnt*Vblksiz) if eigenvectors are requested (wantz != 0).
 *          Stores the Householder vectors.
 *          Uses one Householder reflector from the previous type 1 or 2
 *          kernel to continue an update.
 *          Adds one Householder vector to eliminate a column of the bulge.
 *
 * @param[in,out] tau
 *          Array of dimension 2*n.
 *          Stores the scalar factors of the Householder reflectors.
 *          Uses one scalar factor to continue an update.
 *          Adds one scalar factor.
 *
 * @param[in] first
 *          The first index to update.
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
void plasma_core_zhbtrd_type2(
    int n, int nb,
    plasma_complex64_t *A, int lda,
    plasma_complex64_t *V, plasma_complex64_t *tau,
    int first, int last, int sweep, int Vblksiz, int wantz,
    plasma_complex64_t *work)
{
    plasma_complex64_t ctmp;
    int J1, J2, len, lem, ldx;
    int blkid, vpos, taupos, tpos;

    if (wantz == 0) {
        vpos   = ((sweep + 1)%2)*n + first;
        taupos = ((sweep + 1)%2)*n + first;
    }
    else {
        findVTpos( n, nb, Vblksiz, sweep, first,
                   &vpos, &taupos, &tpos, &blkid );
    }

    ldx = lda - 1;
    J1  = last + 1;
    J2  = imin( last + nb, n - 1 );
    len = last - first + 1;
    lem = J2 - J1 + 1;

    //assert( len > 0 );
    //if (lem == 0) {
    //    printf( "%s: begin %d, last %d, len %d, lem %d\n",
    //            __func__, first, last, len, lem );
    //}
    //assert( lem > 0 );

    if (lem > 0) {
        // Apply remaining right coming from the top block.
        LAPACKE_zlarfx_work( LAPACK_COL_MAJOR, lapack_const( PlasmaRight ),
                             lem, len, V( vpos ), *(tau( taupos )),
                             A( J1, first ), ldx, work );
    }

    if (lem > 1) {
        if (wantz == 0 ) {
            vpos   = ((sweep + 1)%2)*n + J1;
            taupos = ((sweep + 1)%2)*n + J1;
        }
        else {
            findVTpos( n, nb, Vblksiz, sweep, J1,
                       &vpos, &taupos, &tpos, &blkid );
        }

        // Eliminate the first column of the created bulge.
        *V( vpos ) = 1.;

        memcpy( V( vpos+1 ), A( J1+1, first ), (lem-1)*sizeof(plasma_complex64_t) );
        memset( A( J1+1, first ), 0,           (lem-1)*sizeof(plasma_complex64_t) );

        // Eliminate first col.
        LAPACKE_zlarfg_work( lem, A( J1, first ), V( vpos+1 ), 1, tau( taupos ) );

        // Apply left on A( J1:J2, first+1:last )
        // We decrease len because we start at col first+1 instead of first.
        // Col first is the col that has been eliminated.
        len = len - 1;

        ctmp = conj( *tau( taupos ) );
        LAPACKE_zlarfx_work( LAPACK_COL_MAJOR, lapack_const( PlasmaLeft ),
                             lem, len, V( vpos ), ctmp,
                             A( J1, first+1 ), ldx, work );
    }
}

#undef A
#undef V
#undef tau
