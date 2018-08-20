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
#include "core_lapack.h"
#include "bulge.h"
#include <string.h>

#define A(m,n)   (A + LDA * (n) + ((m)-(n)))
#define V(m)     (V + (m))
#define TAU(m)   (TAU + (m))

/***************************************************************************//**
 *
 * @ingroup core_hbtype1cb
 *
 *  Is a kernel that will operate on a region (triangle) of data
 *  bounded by st and ed. This kernel eliminate a column by an column-wise
 *  annihiliation, then it apply a left+right update on the hermitian triangle.
 *  Note that the column to be eliminated is located at st-1.
 *
 *  All detail are available on technical report or SC11 paper.
 *  Azzam Haidar, Hatem Ltaief, and Jack Dongarra. 2011.
 *  Parallel reduction to condensed forms for symmetric eigenvalue problems
 *  using aggregated fine-grained and memory-aware kernels. In Proceedings
 *  of 2011 International Conference for High Performance Computing,
 *  Networking, Storage and Analysis (SC '11). ACM, New York, NY, USA, ,
 *  Article 8 , 11 pages.
 *  http://doi.acm.org/10.1145/2063384.2063394
 *
 *******************************************************************************
 *
 * @param[in] N
 *          The order of the matrix A.
 *
 * @param[in] NB
 *          The size of the band.
 *
 * @param[in, out] A
 *          A pointer to the matrix A of size (2*NB+1)-by-N.
 *
 * @param[in] LDA
 *          The leading dimension of the matrix A. LDA >= max(1,2*NB+1)
 *
 * @param[out] V
 *          PLASMA_Complex64_t array, dimension N if eigenvalue only
 *          requested or (LDV*blkcnt*Vblksiz) if Eigenvectors requested
 *          The Householder reflectors are stored in this array.
 *
 * @param[out] TAU
 *          PLASMA_Complex64_t array, dimension (N).
 *          The scalar factors of the Householder reflectors are stored
 *          in this array.
 *
 * @param[in] st
 *          A pointer to the start index where this kernel will operate.
 *
 * @param[in] ed
 *          A pointer to the end index where this kernel will operate.
 *
 * @param[in] sweep
 *          The sweep number that is eliminated. it serve to calculate the
 *          pointer to the position where to store the Vs and Ts.
 *
 * @param[in] Vblksiz
 *          constant which correspond to the blocking used when applying the Vs.
 *          it serve to calculate the pointer to the position where to store the
 *          Vs and Ts.
 *
 * @param[in] WANTZ
 *          constant which indicate if Eigenvalue are requested or both
 *          Eigenvalue/Eigenvectors.
 *
 * @param[in] WORK
 *          Workspace of size nb.
 *
 *******************************************************************************
 *
 * @return
 *          \retval PLASMA_SUCCESS successful exit
 *          \retval <0 if -i, the i-th argument had an illegal value
 *
 ******************************************************************************/
/***************************************************************************
 *          TYPE 1-BAND Lower-columnwise-Householder
 ***************************************************************************/
void core_zhbtype1cb(int N, int NB,
                     plasma_complex64_t *A, int LDA,
                     plasma_complex64_t *V, plasma_complex64_t *TAU,
                     int st, int ed, int sweep, int Vblksiz, int WANTZ,
                     plasma_complex64_t *WORK)
{
    int len, LDX;
    int blkid, vpos, taupos, tpos;

    /* find the pointer to the Vs and Ts as stored by the bulgechasing
     * note that in case no eigenvector required V and T are stored
     * on a vector of size N
     * */
     if( WANTZ == 0 ) {
         vpos   = ((sweep+1)%2)*N + st;
         taupos = ((sweep+1)%2)*N + st;
     } else {
         findVTpos(N, NB, Vblksiz, sweep, st,
                   &vpos, &taupos, &tpos, &blkid);
     }

    LDX = LDA-1;
    len = ed-st+1;
    *V(vpos) = 1.;

    memcpy( V(vpos+1), A(st+1, st-1), (len-1)*sizeof(plasma_complex64_t) );
    memset( A(st+1, st-1), 0, (len-1)*sizeof(plasma_complex64_t) );

    /* Eliminate the col  at st-1 */
    LAPACKE_zlarfg_work(len, A(st, st-1), V(vpos+1), 1, TAU(taupos) );

    /* Apply left and right on A(st:ed,st:ed) */
    core_zlarfy(len, A(st,st), LDX, V(vpos), TAU(taupos), WORK);

    return;
}
/***************************************************************************/
#undef A
#undef V
#undef TAU


