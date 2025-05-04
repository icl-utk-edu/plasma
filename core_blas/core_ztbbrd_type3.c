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
#include "plasma_bulge.h"

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
 *  Updates diagonal tiles after the first one of each sweep in the SVD
 *  bulge-chasing algorithm. For an upper band matrix, applies the
 *  reflector from the previous type 2 kernel on the right to update the
 *  diagonal tile in rows [first, last] inclusive, creating a bulge
 *  below the diagonal. Then applies another Householder reflector on
 *  the left to eliminate entries below the diagonal in the first column
 *  of the bulge, limiting the update to columns [first, last]
 *  inclusive, as shown below. This kernel is very similar to type 1 but
 *  does not do an elimination of row first-1.
 *
 *      For nb = 3:                 Apply right          Apply left
 *             b b                  b b                  b b
 *               b b x x f f          b b x X 0 0          b b x x
 *                 b b x x f            b b X X F            b b x   x f
 *                 f b b x x            f b B X X            f b b   x x
 *      first:        b b x x              {B B X} x            <B> {B X} X F F
 *                      b b x x   =>       {F B B} x x          <0> {B B} X X F
 *      last:             b b x x          {F F B} b x x        <0> {F B} B X X
 *
 *  For a lower band matrix, the symmetric process is used.
 *
 *  @see plasma_core_ztbbrd_type1 for legend.
 *  @see plasma_core_ztbbrd_type2
 *
 *  All details are available in the SC13 paper:
 *  Azzam Haidar, Jakub Kurzak, Piotr Luszczek. 2013.
 *  An improved parallel singular value algorithm and its implementation
 *  for multicore hardware. In Proceedings of the International
 *  Conference on High Performance Computing, Networking, Storage and
 *  Analysis (SC '13).
 *  https://doi.org/10.1145/2503210.2503292
 *
 *  and the technical report:
 *  SLATE Working Note 13:
 *  Implementing Singular Value and Symmetric/Hermitian Eigenvalue Solvers.
 *  Mark Gates, Kadir Akbudak, Mohammed Al Farhan, Ali Charara, Jakub Kurzak,
 *  Dalal Sukkari, Asim YarKhan, Jack Dongarra. 2023.
 *  https://icl.utk.edu/publications/swan-013
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
 *          PLASMA_Complex64_t array, dimension n if singular value only
 *          requested or (LDV*blkcnt*Vblksiz) if singular vectors requested
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
 *          PLASMA_Complex64_t array, dimension n if singular value only
 *          requested or (LDV*blkcnt*Vblksiz) if singular vectors requested
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
 *          Specifies whether only singular values are requested or both
 *          singular values and singular vectors.
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
