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

#define A(m,n)   (A + LDA * (n) + ((m)-(n)))
#define V(m)     (V + (m))
#define TAU(m)   (TAU + (m))

/***************************************************************************//**
 *
 * @ingroup CORE_plasma_complex64_t
 *
 *  CORE_zhbtype3cb is a kernel that will operate on a region (triangle) of data
 *  bounded by st and ed. This kernel apply a left+right update on the hermitian
 *  triangle.  Note that this kernel is very similar to type1 but does not do an
 *  elimination.
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
 * @param[in] V
 *          plasma_complex64_t array, dimension N if eigenvalue only
 *          requested or (LDV*blkcnt*Vblksiz) if Eigenvectors requested
 *          The Householder reflectors are stored in this array.
 *
 * @param[in] TAU
 *          plasma_complex64_t array, dimension (N).
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

/***************************************************************************//**
 *          TYPE 3-BAND Lower-columnwise-Householder
 ***************************************************************************/
void core_zhbtype3cb(int N, int NB,
                     plasma_complex64_t *A, int LDA,
                     const plasma_complex64_t *V, const plasma_complex64_t *TAU,
                     int st, int ed, int sweep, int Vblksiz, int WANTZ,
                     plasma_complex64_t *WORK)
{
    int len, LDX;
    int blkid, vpos, taupos, tpos;

    if( WANTZ == 0 ) {
        vpos   = ((sweep+1)%2)*N + st;
        taupos = ((sweep+1)%2)*N + st;
    } else {
        findVTpos(N, NB, Vblksiz, sweep, st,
                  &vpos, &taupos, &tpos, &blkid);
    }

    LDX = LDA-1;
    len = ed-st+1;

    /* Apply left and right on A(st:ed,st:ed)*/
    core_zlarfy(len, A(st,st), LDX, V(vpos), TAU(taupos), WORK);
    return;
}
/***************************************************************************/
#undef A
#undef V
#undef TAU
