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


#define AL(m_, n_) (A + nb + lda * (n_) + ((m_)-(n_)))
#define AU(m_, n_) (A + nb + lda * (n_) + ((m_)-(n_)+nb))
#define VQ(m)     (VQ + (m))
#define VP(m)     (VP + (m))
#define TAUQ(m)   (TAUQ + (m))
#define TAUP(m)   (TAUP + (m))

/***************************************************************************//**
 *
 * @ingroup core_zgbtype2cb
 *
 *  core_zgbtype2cb is a kernel that will operate on a region (triangle) of data
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
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of A is stored;
 *          = PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] N
 *          The order of the matrix A.
 *
 * @param[in] nb
 *          The size of the band.
 *
 * @param[in, out] A
 *          A pointer to the matrix A of size (3*nb+1)-by-N.
 *
 * @param[in] lda
 *          The leading dimension of the matrix A. lda >= max(1,3*nb+1)
 *
 * @param[in, out] VP
 *          TODO: Check and fix doc
 *          PLASMA_Complex64_t array, dimension n if eigenvalue only
 *          requested or (LDV*blkcnt*Vblksiz) if Eigenvectors requested
 *          The Householder reflectors of the previous type 1 are used here
 *          to continue update then new one are generated to eliminate the
 *          bulge and stored in this array.
 *
 * @param[in, out] TAUP
 *          TODO: Check and fix doc
 *          PLASMA_Complex64_t array, dimension (n).
 *          The scalar factors of the Householder reflectors of the previous
 *          type 1 are used here to continue update then new one are generated
 *          to eliminate the bulge and stored in this array.
 *
 * @param[in, out] VQ
 *          TODO: Check and fix doc
 *          PLASMA_Complex64_t array, dimension n if eigenvalue only
 *          requested or (LDV*blkcnt*Vblksiz) if Eigenvectors requested
 *          The Householder reflectors of the previous type 1 are used here
 *          to continue update then new one are generated to eliminate the
 *          bulge and stored in this array.
 *
 * @param[in, out] TAUQ
 *          TODO: Check and fix doc
 *          PLASMA_Complex64_t array, dimension (n).
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
 *          TYPE 2-BAND-bidiag Lower/Upper columnwise-Householder
 ***************************************************************************/

void plasma_core_zgbtype2cb (plasma_enum_t uplo, int n, int nb,
                     plasma_complex64_t *A, int lda,
                     plasma_complex64_t *VQ, plasma_complex64_t *TAUQ,
                     plasma_complex64_t *VP, plasma_complex64_t *TAUP,
                     int st, int ed, int sweep, int Vblksiz, int wantz,
                     plasma_complex64_t *WORK)
{
    plasma_complex64_t ctmp;
    int i, J1, J2, len, lem, LDX;
    int blkid, vpos, taupos, tpos;

    LDX = lda-1;
    J1  = ed+1;
    J2  = imin(ed+nb,n-1);
    lem = ed-st+1;
    len = J2-J1+1;

    if( uplo == PlasmaUpper ) {
        /* ========================
        *       UPPER CASE
        * ========================*/
        if( len > 0 ) {
            if( wantz == 0 ) {
                vpos   = ((sweep+1)%2)*n + st;
                taupos = ((sweep+1)%2)*n + st;
            } else {
                findVTpos(n, nb, Vblksiz, sweep, st,
                          &vpos, &taupos, &tpos, &blkid);
            }
            // Apply remaining Left commming from type1/3_upper 
            ctmp = conj(*TAUQ(taupos));
            LAPACKE_zlarfx_work(LAPACK_COL_MAJOR, 'L',
                                lem, len, VQ(vpos), ctmp, AU(st, J1), LDX, WORK);
        }

        if( len > 1 ) {
            if( wantz == 0 ) {
                vpos   = ((sweep+1)%2)*n + J1;
                taupos = ((sweep+1)%2)*n + J1;
            } else {
                findVTpos(n,nb,Vblksiz,sweep,J1, &vpos, &taupos, &tpos, &blkid);
            }

            // Remove the top row of the created bulge 
            *VP(vpos) = 1.;
            for(i=1; i<len; i++){
                *VP(vpos+i)     = conj(*AU(st, J1+i));
                *AU(st, J1+i) = 0.;
            }
            // Eliminate the row  at st 
            ctmp = conj(*AU(st, J1));
            LAPACKE_zlarfg_work(len, &ctmp, VP(vpos+1), 1, TAUP(taupos) );
            *AU(st, J1) = ctmp;
            /*
             * Apply Right on A(J1:J2,st+1:ed)
             * We decrease len because we start at row st+1 instead of st.
             * row st is the row that has been revomved;
             */
            lem = lem-1;
            ctmp = *TAUP(taupos);
            LAPACKE_zlarfx_work(LAPACK_COL_MAJOR, 'R',
                                lem, len, VP(vpos), ctmp, AU(st+1, J1), LDX, WORK);
        }
    }else{
        /* ========================
         *       LOWER CASE
         * ========================*/
        if( len > 0 ) {
            if( wantz == 0 ) {
                vpos   = ((sweep+1)%2)*n + st;
                taupos = ((sweep+1)%2)*n + st;
            } else {
                findVTpos(n, nb, Vblksiz, sweep, st,
                          &vpos, &taupos, &tpos, &blkid);
            }
            // Apply remaining Right commming from type1/3_lower 
            ctmp = (*TAUP(taupos));
            LAPACKE_zlarfx_work(LAPACK_COL_MAJOR, 'R',
                                len, lem, VP(vpos), ctmp, AL(J1, st), LDX, WORK);
        }
        if( len > 1 ) {
            if( wantz == 0 ) {
                vpos   = ((sweep+1)%2)*n + J1;
                taupos = ((sweep+1)%2)*n + J1;
            } else {
                findVTpos(n,nb,Vblksiz,sweep,J1, &vpos, &taupos, &tpos, &blkid);
            }

            // Remove the first column of the created bulge
            *VQ(vpos) = 1.;
            memcpy(VQ(vpos+1), AL(J1+1, st), (len-1)*sizeof(plasma_complex64_t));
            memset(AL(J1+1, st), 0, (len-1)*sizeof(plasma_complex64_t));
            // Eliminate the col  at st 
            LAPACKE_zlarfg_work(len, AL(J1, st), VQ(vpos+1), 1, TAUQ(taupos) );
            /*
             * Apply left on A(J1:J2,st+1:ed)
             * We decrease len because we start at col st+1 instead of st.
             * col st is the col that has been revomved;
             */
            lem = lem-1;
            ctmp = conj(*TAUQ(taupos));
            LAPACKE_zlarfx_work(LAPACK_COL_MAJOR, 'L',
                                len, lem, VQ(vpos), ctmp, AL(J1, st+1), LDX, WORK);
        }
    }
    // end of uplo case
    return;
}
/***************************************************************************/
#undef AU
#undef AL
#undef VQ
#undef VP
#undef TAUQ
#undef TAUP
