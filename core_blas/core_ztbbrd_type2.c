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
 * @ingroup core_tbbrd_type2
 *
 *  Updates the off-diagonal tiles in the SVD bulge-chasing algorithm.
 *  For an upper band matrix, continues to apply the reflector from the
 *  previous type 1 or 3 kernel on the left to update an off-diagonal
 *  tile, creating a bulge above the band. Then applies another
 *  Householder reflector on the right to eliminate entries outside the
 *  band in row first, limiting the update to columns [first, last]
 *  inclusive, as shown below.
 *
 *      For nb = 3:                Apply left           Apply right
 *                                         J1   J2              J1   J2
 *             b b                 b b                  b b
 *      first:   b b x x             B B X {X F F}        b b x <X 0 0>
 *               f b b x x     =>    0 B B {X X F}  =>      b b {X X F}
 *      last:    f f b b x x         0 F B {B X X}          f b {B X X}
 *                     b b x x              b b x x              B B X x
 *                       b b x x              b b x x            F B B x x
 *                         b b x x              b b x x          F F B b x x
 *
 *  For a lower band matrix, the symmetric process is used.
 *
 *  @see plasma_core_ztbbrd_type1 for legend.
 *  @see plasma_core_ztbbrd_type3
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
 * @param[in,out] VP
 *          TODO: Check and fix doc
 *          PLASMA_Complex64_t array, dimension n if singular value only
 *          requested or (LDV*blkcnt*Vblksiz) if singular vectors requested
 *          The Householder reflectors of the previous type 1 are used here
 *          to continue update then new one are generated to eliminate the
 *          bulge and stored in this array.
 *
 * @param[in,out] tauP
 *          TODO: Check and fix doc
 *          PLASMA_Complex64_t array, dimension (n).
 *          The scalar factors of the Householder reflectors of the previous
 *          type 1 are used here to continue update then new one are generated
 *          to eliminate the bulge and stored in this array.
 *
 * @param[in,out] VQ
 *          TODO: Check and fix doc
 *          PLASMA_Complex64_t array, dimension n if singular value only
 *          requested or (LDV*blkcnt*Vblksiz) if singular vectors requested
 *          The Householder reflectors of the previous type 1 are used here
 *          to continue update then new one are generated to eliminate the
 *          bulge and stored in this array.
 *
 * @param[in,out] tauQ
 *          TODO: Check and fix doc
 *          PLASMA_Complex64_t array, dimension (n).
 *          The scalar factors of the Householder reflectors of the previous
 *          type 1 are used here to continue update then new one are generated
 *          to eliminate the bulge and stored in this array.
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
 *          singular value and singular vectors.
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
void plasma_core_ztbbrd_type2(
    plasma_enum_t uplo, int n, int nb,
    plasma_complex64_t *A, int lda,
    plasma_complex64_t *VQ, plasma_complex64_t *tauQ,
    plasma_complex64_t *VP, plasma_complex64_t *tauP,
    int first, int last, int sweep, int Vblksiz, int wantz,
    plasma_complex64_t *work)
{
    plasma_complex64_t ctau;
    int i, J1, J2, lenj, len, ldx;
    int blkid, vpos, taupos, tpos;

    // [J1, J2] are columns this kernel updates.
    ldx = lda - 1;
    J1  = last + 1;
    J2  = imin( last + nb, n - 1 );
    len = last - first + 1;
    lenj = J2 - J1 + 1;

    if (uplo == PlasmaUpper) {
        //========================
        //      UPPER CASE
        //========================
        if (lenj > 0) {
            if (wantz == 0) {
                vpos   = ((sweep + 1)%2)*n + first;
                taupos = ((sweep + 1)%2)*n + first;
            }
            else {
                findVTpos( n, nb, Vblksiz, sweep, first,
                           &vpos, &taupos, &tpos, &blkid );
            }
            // Apply remaining left coming from the previous type 1 or 3 kernel.
            ctau = conj( *tauQ( taupos ) );
            LAPACKE_zlarfx_work(
                LAPACK_COL_MAJOR, 'L', len, lenj,
                VQ( vpos ), ctau, AU( first, J1 ), ldx, work);
        }
        if (lenj > 1) {
            if (wantz == 0) {
                vpos   = ((sweep + 1)%2)*n + J1;
                taupos = ((sweep + 1)%2)*n + J1;
            }
            else {
                findVTpos( n, nb, Vblksiz, sweep, J1,
                           &vpos, &taupos, &tpos, &blkid );
            }

            // Remove the top row of the created bulge
            *VP( vpos ) = 1.;
            for (i = 1; i < lenj; ++i) {
                *VP( vpos+i )     = conj( *AU( first, J1+i ) );
                *AU( first, J1+i ) = 0.;
            }

            // Eliminate first row.
            *AU( first, J1 ) = conj( *AU( first, J1 ) );
            LAPACKE_zlarfg_work(
                lenj, AU( first, J1 ), VP( vpos+1 ), 1, tauP( taupos ) );

            // Apply Right on A( J1:J2, first+1:last )
            LAPACKE_zlarfx_work(
                LAPACK_COL_MAJOR, 'R', len-1, lenj,
                VP( vpos ), *tauP( taupos ), AU( first+1, J1 ), ldx, work );
        }
    }
    else {
        //========================
        //      LOWER CASE
        //========================
        if (lenj > 0) {
            if (wantz == 0) {
                vpos   = ((sweep + 1)%2)*n + first;
                taupos = ((sweep + 1)%2)*n + first;
            }
            else {
                findVTpos( n, nb, Vblksiz, sweep, first,
                           &vpos, &taupos, &tpos, &blkid );
            }
            // Apply remaining right coming from previous type 1 or 3 kernel.
            LAPACKE_zlarfx_work(
                LAPACK_COL_MAJOR, 'R', lenj, len,
                VP( vpos ), *tauP( taupos ), AL( J1, first ), ldx, work );
        }
        if (lenj > 1) {
            if (wantz == 0) {
                vpos   = ((sweep + 1)%2)*n + J1;
                taupos = ((sweep + 1)%2)*n + J1;
            }
            else {
                findVTpos( n, nb, Vblksiz, sweep, J1,
                           &vpos, &taupos, &tpos, &blkid );
            }

            // Remove the first column of the created bulge
            *VQ( vpos ) = 1.;
            memcpy( VQ( vpos+1 ), AL( J1+1, first ), (lenj-1)*sizeof( plasma_complex64_t ) );
            memset( AL( J1+1, first ), 0,            (lenj-1)*sizeof( plasma_complex64_t ) );

            // Eliminate first col.
            LAPACKE_zlarfg_work(
                lenj, AL( J1, first ), VQ( vpos+1 ), 1, tauQ( taupos ) );

            // Apply left on A( J1:J2, first+1:last )
            ctau = conj( *tauQ( taupos ) );
            LAPACKE_zlarfx_work(
                LAPACK_COL_MAJOR, 'L', lenj, len-1,
                VQ( vpos ), ctau, AL( J1, first+1 ), ldx, work );
        }
    }
}

#undef AU
#undef AL
#undef VQ
#undef VP
#undef tauQ
#undef tauP
