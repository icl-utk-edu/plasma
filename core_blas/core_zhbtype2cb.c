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
#include "plasma_core_blas.h"
#include "plasma_types.h"
#include "plasma_internal.h"
#include "core_lapack.h"

#include <string.h>

#define A(m, n)  (A + lda*(n) + ((m) - (n)))
#define V(m)     (V + (m))
#define tau(m)   (tau + (m))

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
 *          The leading dimension of the matrix A. lda >= max(1, 2*nb + 1)
 *
 * @param[in, out] V
 *          plasma_complex64_t array, dimension n if eigenvalue only
 *          requested or (ldv*blkcnt*Vblksiz) if Eigenvectors requested
 *          The Householder reflectors of the previous type 1 are used here
 *          to continue update then new one are generated to eliminate the
 *          bulge and stored in this array.
 *
 * @param[in, out] tau
 *          plasma_complex64_t array, dimension (n).
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
 *          TYPE 2-BAND Lower-columnwise-Householder
 ***************************************************************************/
void plasma_core_zhbtype2cb(
    int n, int nb,
    plasma_complex64_t *A, int lda,
    plasma_complex64_t *V, plasma_complex64_t *tau,
    int st, int ed, int sweep, int Vblksiz, int wantz,
    plasma_complex64_t *work)
{
    plasma_complex64_t ctmp;
    int J1, J2, len, lem, ldx;
    int blkid, vpos, taupos, tpos;

    if (wantz == 0) {
        vpos   = ((sweep + 1)%2)*n + st;
        taupos = ((sweep + 1)%2)*n + st;
    }
    else {
        findVTpos(n, nb, Vblksiz, sweep, st,
                  &vpos, &taupos, &tpos, &blkid);
    }

    ldx = lda-1;
    J1  = ed+1;
    J2  = imin(ed+nb, n-1);
    len = ed-st+1;
    lem = J2-J1+1;

    if (lem > 0) {
        // Apply remaining right commming from the top block.
        LAPACKE_zlarfx_work(LAPACK_COL_MAJOR, lapack_const(PlasmaRight),
                            lem, len, V(vpos), *(tau(taupos)),
                            A(J1, st), ldx, work);
    }

    if (lem > 1) {
        if (wantz == 0 ) {
            vpos   = ((sweep+1)%2)*n + J1;
            taupos = ((sweep+1)%2)*n + J1;
        }
        else {
            findVTpos(n, nb, Vblksiz, sweep, J1, &vpos, &taupos, &tpos, &blkid);
        }

        /* Remove the first column of the created bulge */
        *V(vpos) = 1.;

        memcpy(V(vpos+1), A(J1+1, st), (lem-1)*sizeof(plasma_complex64_t));
        memset(A(J1+1, st), 0, (lem-1)*sizeof(plasma_complex64_t));

        /* Eliminate the col at st */
        LAPACKE_zlarfg_work( lem, A(J1, st), V(vpos+1), 1, tau(taupos) );

        /*
         * Apply left on A(J1:J2, st+1:ed)
         * We decrease len because we start at col st+1 instead of st.
         * col st is the col that has been revomved;
         */
        len = len-1;

        ctmp = conj(*tau(taupos));
        LAPACKE_zlarfx_work(LAPACK_COL_MAJOR, lapack_const(PlasmaLeft),
                            lem, len, V(vpos), ctmp, A(J1, st+1), ldx, work);
    }
}
/***************************************************************************/
#undef A
#undef V
#undef tau
