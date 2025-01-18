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

#define A(m, n)  (A + lda*(n) + ((m) - (n)))
#define V(m)     (V + (m))
#define tau(m)   (tau + (m))

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
 * @param[in] n
 *          The order of the matrix A.
 *
 * @param[in] nb
 *          The size of the band.
 *
 * @param[in, out] A
 *          A pointer to the matrix A of size (2*nb + 1)-by-n.
 *
 * @param[in] lda
 *          The leading dimension of the matrix A. lda >= max(1, 2*nb + 1)
 *
 * @param[out] V
 *          PLASMA_Complex64_t array, dimension n if eigenvalue only
 *          requested or (ldv*blkcnt*Vblksiz) if Eigenvectors requested
 *          The Householder reflectors are stored in this array.
 *
 * @param[out] tau
 *          PLASMA_Complex64_t array, dimension (n).
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
 * @param[in] wantz
 *          constant which indicate if Eigenvalue are requested or both
 *          Eigenvalue/Eigenvectors.
 *
 * @param[in] work
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
void plasma_core_zhbtype1cb(
    int n, int nb,
    plasma_complex64_t *A, int lda,
    plasma_complex64_t *V, plasma_complex64_t *tau,
    int st, int ed, int sweep, int Vblksiz, int wantz,
    plasma_complex64_t *work)
{
    int len, ldx;
    int blkid, vpos, taupos, tpos;

    // Find the pointer to the Vs and Ts as stored by the bulge chasing.
    // Note that in case no eigenvector required V and T are stored
    // on a vector of size n
    if (wantz == 0) {
        vpos   = ((sweep + 1)%2)*n + st;
        taupos = ((sweep + 1)%2)*n + st;
    }
    else {
        findVTpos(n, nb, Vblksiz, sweep, st,
                  &vpos, &taupos, &tpos, &blkid);
    }

    ldx = lda-1;
    len = ed-st+1;
    *V(vpos) = 1.;

    memcpy( V(vpos+1), A(st+1, st-1), (len-1)*sizeof(plasma_complex64_t) );
    memset( A(st+1, st-1), 0, (len-1)*sizeof(plasma_complex64_t) );

    // Eliminate the col at st-1.
    LAPACKE_zlarfg_work(len, A(st, st-1), V(vpos+1), 1, tau(taupos) );

    // Apply left and right on A(st:ed, st:ed).
    plasma_core_zlarfy(len, A(st, st), ldx, V(vpos), tau(taupos), work);
}
/***************************************************************************/
#undef A
#undef V
#undef tau


