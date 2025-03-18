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

#define AL( i_, j_ ) (A + nb + lda * (j_) + ((i_)-(j_)))
#define AU( i_, j_ ) (A + nb + lda * (j_) + ((i_)-(j_)+nb))
#define VQ( i_ )     (VQ + (i_))
#define VP( i_ )     (VP + (i_))
#define tauQ( i_ )   (tauQ + (i_))
#define tauP( i_ )   (tauP + (i_))

/***************************************************************************//**
 *
 * @ingroup core_tbbrd_type3
 *
 *  core_ztbbrd_type3 is a kernel that will operate on a region (triangle) of data
 *  bounded by start and end. This kernel apply a left+right update on the hermitian
 *  triangle.  Note that this kernel is very similar to type1 but does not do an
 *  elimination.
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
 * @param[in] VP
 *          TODO: Check and fix doc
 *          PLASMA_Complex64_t array, dimension n if eigenvalue only
 *          requested or (LDV*blkcnt*Vblksiz) if Eigenvectors requested
 *          The Householder reflectors are stored in this array.
 *
 * @param[in] tauP
 *          TODO: Check and fix doc
 *          PLASMA_Complex64_t array, dimension (n).
 *          The scalar factors of the Householder reflectors are stored
 *          in this array.
 *
 * @param[in] VQ
 *          TODO: Check and fix doc
 *          PLASMA_Complex64_t array, dimension n if eigenvalue only
 *          requested or (LDV*blkcnt*Vblksiz) if Eigenvectors requested
 *          The Householder reflectors are stored in this array.
 *
 * @param[in] tauQ
 *          TODO: Check and fix doc
 *          PLASMA_Complex64_t array, dimension (n).
 *          The scalar factors of the Householder reflectors are stored
 *          in this array.
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
void plasma_core_ztbbrd_type3(
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
        // Apply P on right to A( first:last, first:last ).
        LAPACKE_zlarfx_work(
            LAPACK_COL_MAJOR, 'R', len, len,
            VP( vpos ), *tauP( taupos ), AU( first, first ), ldx, work );

        // Eliminate the created col at first
        *VQ( vpos ) = 1.;
        memcpy( VQ( vpos+1 ), AU( first+1, first ), (len-1)*sizeof( plasma_complex64_t ) );
        memset( AU( first+1, first ), 0, (len-1)*sizeof( plasma_complex64_t ) );
        LAPACKE_zlarfg_work(
            len, AU( first, first ), VQ( vpos+1 ), 1, tauQ( taupos ) );

        // Apply Q on left to A( ).
        ctau = conj( *tauQ( taupos ) );
        LAPACKE_zlarfx_work(
            LAPACK_COL_MAJOR, 'L', len, len-1,
            VQ( vpos ), ctau, AU( first, first+1 ), ldx, work );
    }
    else {
        //========================
        //      LOWER CASE
        //========================
        // Apply Q on left to A( first:last, first:last ).
        ctau = conj( *tauQ( taupos ) );
        LAPACKE_zlarfx_work(
            LAPACK_COL_MAJOR, 'L', len, len,
            VQ( vpos ), ctau, AL( first, first ), ldx, work );

        // Eliminate the created row at first
        *VP( vpos ) = 1.;
        for (i = 1; i < len; ++i) {
            *VP( vpos+i ) = conj( *AL( first, first+i ) );
            *AL( first, first+i ) = 0.;
        }
        *AL( first, first ) = conj( *AL( first, first ) );
        LAPACKE_zlarfg_work(
            len, AL( first, first ), VP( vpos+1 ), 1, tauP( taupos ) );

        // Apply P on right to A( ).
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
