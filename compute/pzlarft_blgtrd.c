/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> s d c
 *
 **/

#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_types.h"
#include "plasma_workspace.h"
#include "plasma_core_blas.h"
#include "core_lapack.h"
#include "bulge.h"

#include <omp.h>

#define V(m)     &(V[(m)])
#define TAU(m)   &(TAU[(m)])
#define T(m)   &(T[(m)])
/***************************************************************************/
/**
 *  Parallel compute T2 from bulgechasing of Symetric matrix 
 *  Lower case is supported
 **/
/***************************************************************************/
void plasma_pzlarft_blgtrd(int N, int NB, int Vblksiz,
                           plasma_complex64_t *V, plasma_complex64_t *T,
                           plasma_complex64_t *TAU,
                           plasma_sequence_t *sequence,
                           plasma_request_t *request)
{
    int my_core_id = omp_get_thread_num();
    int cores_num  = omp_get_num_threads();

    //===========================
    //   local variables
    //===========================
    int LDT, LDV;
    int Vm, Vn, mt, nt;
    int myrow, mycol, blkj, blki;
    int firstrow;
    int blkid,vpos,taupos,tpos;
    int blkpercore,blkcnt, myid;

    if (sequence->status != PlasmaSuccess)
        return;

    // Quick return */
    if (N == 0){
        return;
    }
    if (NB == 0){
        return;
    }
    if (NB == 1){
        return;
    }

    findVTsiz(N, NB, Vblksiz, &blkcnt, &LDV);
    blkpercore = blkcnt/cores_num;
    blkpercore = blkpercore==0 ? 1:blkpercore;
    LDT        = Vblksiz;
    LDV        = NB+Vblksiz-1;

    /*========================================
     * compute the T's in parallel.
     * The Ts are independent so each core pick
     * a T and compute it. The loop is based on
     * the version 113 of the pzunmqr_blgtrd.c
     * which go over the losange block_column
     * by block column. but it is not important
     * here the order because Ts are independent.
     * ========================================
     */
    nt  = plasma_ceildiv((N-1),Vblksiz);
    for (blkj=nt-1; blkj>=0; blkj--) {
        /* the index of the first row on the top of block (blkj) */
        firstrow = blkj * Vblksiz + 1;
        /*find the number of tile for this block */
        if( blkj == nt-1 )
            mt = plasma_ceildiv( N -  firstrow,    NB);
        else
            mt = plasma_ceildiv( N - (firstrow+1), NB);
        /*loop over the tiles find the size of the Vs and apply it */
        for (blki=mt; blki>0; blki--) {
            /*calculate the size of each losange of Vs= (Vm,Vn)*/
            myrow     = firstrow + (mt-blki)*NB;
            mycol     = blkj*Vblksiz;
            Vm = imin( NB+Vblksiz-1, N-myrow);
            if( ( blkj == nt-1 ) && ( blki == mt ) ){
                Vn = imin (Vblksiz, Vm);
            } else {
                Vn = imin (Vblksiz, Vm-1);
            }
            /*calculate the pointer to the Vs and the Ts.
             * Note that Vs and Ts have special storage done
             * by the bulgechasing function*/
            findVTpos(N,NB,Vblksiz,mycol,myrow, &vpos, &taupos, &tpos, &blkid);
            myid = blkid/blkpercore;
            if( my_core_id==(myid%cores_num) ){
                if( ( Vm > 0 ) && ( Vn > 0 ) ){
                    LAPACKE_zlarft_work(LAPACK_COL_MAJOR,
                                        lapack_const(PlasmaForward),
                                        lapack_const(PlasmaColumnwise),
                                        Vm, Vn, V(vpos), LDV, TAU(taupos), T(tpos), LDT);
                }
            }
        }
    }
}
#undef V
#undef TAU
#undef T
/***************************************************************************/
