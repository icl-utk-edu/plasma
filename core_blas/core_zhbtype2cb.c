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
#include "bulge.h"
#include <string.h>
#include "core_blas.h"
#include "plasma_types.h"
#include "plasma_internal.h"
#include "core_lapack.h"

#define A(m,n)   (A + LDA * (n) + ((m)-(n)))
#define V(m)     (V + (m))
#define TAU(m)   (TAU + (m))

/***************************************************************************//**
 *
 * @ingroup CORE_PLASMA_Complex64_t
 *
 *  CORE_zhbtype2cb is a kernel that will operate on a region (triangle) of data
 *  bounded by st and ed. This kernel apply the right update remaining from the
 *  type1 and this later will create a bulge so it eliminate the first column of
 *  the created bulge and do the corresponding Left update.
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
 * @param[in, out] V
 *          PLASMA_Complex64_t array, dimension N if eigenvalue only
 *          requested or (LDV*blkcnt*Vblksiz) if Eigenvectors requested
 *          The Householder reflectors of the previous type 1 are used here
 *          to continue update then new one are generated to eliminate the
 *          bulge and stored in this array.
 *
 * @param[in, out] TAU
 *          PLASMA_Complex64_t array, dimension (N).
 *          The scalar factors of the Householder reflectors of the previous
 *          type 1 are used here to continue update then new one are generated
 *          to eliminate the bulge and stored in this array.
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
 *          TYPE 2-BAND Lower-columnwise-Householder
 ***************************************************************************/
void core_zhbtype2cb(int N, int NB,
                     plasma_complex64_t *A, int LDA,
                     plasma_complex64_t *V, plasma_complex64_t *TAU,
                     int st, int ed, int sweep, int Vblksiz, int WANTZ,
                     plasma_complex64_t *WORK)
{
    plasma_complex64_t ctmp;
    int J1, J2, len, lem, LDX;
    int blkid, vpos, taupos, tpos;

    if( WANTZ == 0 ) {
        vpos   = ((sweep+1)%2)*N + st;
        taupos = ((sweep+1)%2)*N + st;
    } else {
        findVTpos(N, NB, Vblksiz, sweep, st,
                  &vpos, &taupos, &tpos, &blkid);
    }

    LDX = LDA-1;
    J1  = ed+1;
    J2  = imin(ed+NB,N-1);
    len = ed-st+1;
    lem = J2-J1+1;

    if( lem > 0 ) {
        /* Apply remaining right commming from the top block */
        LAPACKE_zlarfx_work(LAPACK_COL_MAJOR, lapack_const(PlasmaRight),
                            lem, len, V(vpos), *(TAU(taupos)), A(J1, st), LDX, WORK);
    }

    if( lem > 1 ) {
        if( WANTZ == 0 ) {
            vpos   = ((sweep+1)%2)*N + J1;
            taupos = ((sweep+1)%2)*N + J1;
        } else {
            findVTpos(N,NB,Vblksiz,sweep,J1, &vpos, &taupos, &tpos, &blkid);
        }

        /* Remove the first column of the created bulge */
        *V(vpos) = 1.;

        memcpy(V(vpos+1), A(J1+1, st), (lem-1)*sizeof(plasma_complex64_t));
        memset(A(J1+1, st), 0, (lem-1)*sizeof(plasma_complex64_t));

        /* Eliminate the col at st */
        LAPACKE_zlarfg_work( lem, A(J1, st), V(vpos+1), 1, TAU(taupos) );

        /*
         * Apply left on A(J1:J2,st+1:ed)
         * We decrease len because we start at col st+1 instead of st.
         * col st is the col that has been revomved;
         */
        len = len-1;

        ctmp = conj(*TAU(taupos));
        LAPACKE_zlarfx_work(LAPACK_COL_MAJOR, lapack_const(PlasmaLeft),
                            lem, len, V(vpos), ctmp, A(J1, st+1), LDX, WORK);
    }
    return;
}
/***************************************************************************/
#undef A
#undef V
#undef TAU
