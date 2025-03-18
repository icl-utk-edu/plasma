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
 * @ingroup core_tbbrd_type2
 *
 *  core_ztbbrd_type2 is a kernel that will operate on a region (triangle) of data
 *  bounded by start and end. This kernel apply the right update remaining from the
 *  type1 and this later will create a bulge so it eliminate the first column of
 *  the created bulge and do the corresponding Left update.
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
 * @param[in,out] VP
 *          TODO: Check and fix doc
 *          PLASMA_Complex64_t array, dimension n if eigenvalue only
 *          requested or (LDV*blkcnt*Vblksiz) if Eigenvectors requested
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
 *          PLASMA_Complex64_t array, dimension n if eigenvalue only
 *          requested or (LDV*blkcnt*Vblksiz) if Eigenvectors requested
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
