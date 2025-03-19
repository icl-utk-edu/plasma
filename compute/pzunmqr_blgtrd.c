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
#include "plasma_bulge.h"

#include <omp.h>
#include <string.h>

#undef REAL
#define COMPLEX

#define E(_m, _n) (E   + (_m) + LDE * (_n) )
#define V(_m)     (V   + (_m))
#define T(_m)     (T   + (_m))
#define TAU(_m)   (TAU + (_m))
/***************************************************************************
 *  Parallel apply Q2 from bulgechasing matrices - static scheduling
 *  Lower case is treated
 **/
    /*
     * side == PlasmaLeft:
     *     meaning apply E = Q*E = (q_1*q_2*.....*q_n) * E ==> so
     *     traverse Vs in reverse order (forward) from q_n to q_1 Also
     *     E is splitten by block of col over cores because each apply
     *     consist in a block of row (horizontal block)
     */
    /*
     * side == PlasmaRight:
     *     meaning apply E = E*Q = E * (q_1*q_2*.....*q_n) ==> so
     *     traverse Vs in normal order (forward) from q_1 to q_n Also
     *     E is splitten by block of row over core because each apply
     *     consist in a block of col (vertical block)
     */
/***************************************************************************/
void plasma_pzunmqr_blgtrd(plasma_enum_t side, plasma_enum_t trans,
                           int N, int NB, int NE,
                           int Vblksiz, int WANTZ, plasma_complex64_t *V,
                           plasma_complex64_t *T, plasma_complex64_t *TAU,
                           plasma_complex64_t *E, int LDE,
                           plasma_workspace_t work,
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
    int firstrow, nbcolinvolvd;
    int blkid, vpos, taupos, tpos;
    int chunkid, nbchunk, colpercore, corest, corelen, len, col;
    int coreid, allcoresnb, maxrequiredcores;
    int lchunkid, rchunkid, halfchunk, nbportion, sw;
    int standalonework = 0 ;
    int versionL, versionR;


    // Quick return
    if ( N == 0 ) {
        return;
    }
    if ( NB == 0 ) {
        return;
    }
    if ( NE == 0 ) {
        return;
    }
    /* ==========================================
     * some infor for developer
     * Initialisation and checking nb of cores
     * ==========================================*/
    /* we have to 2 algo for left (113 114) and 2 algo for right (91 92)
     * which correspond to versionL versionR.
     * They ae very similar (detail explained in tech report and matlab code)
     * however version 114 and 92 improve locality.
     * while version 113 is used in case WNATZ=1 (construct Q2) which allow
     * the construction to be done in an optimized way taking into
     * consideration that the matrix is Identity so making less flops.
     *
     */
    versionL = 113;
    versionR = 92;
    LDT      = Vblksiz;
    LDV      = NB+Vblksiz-1;
    /* use colpercore =  N/cores_num; :if i want to split E into
     * cores_num chunk so I will have large chunk for each core.
     * However I prefer to split E into chunk of small size where
     * I guarantee that blas3 occur and the size of chunk remain into
     * cache, I think it is better. than I will have different chunk of
     * small size per core and i will do each chunk till the end and
     * then move to the second one for data locality
     *
     * version v1: for each chunck it apply all the V's then move to
     *                    the other chunck. the locality here inside each
     *                    chunck meaning that thread "t" apply V_k then move
     *                    to V_k+1 which overlap with V_k meaning that the
     *                    E_k+1 overlap with E_k. so here is the
     *                    locality however thread t had to read V_k+1 and
     *                    T_k+1 at each apply. note that all thread if they
     *                    run at same speed they might reading the same V_k
     *                    and T_k at the same time.
     *
     * version v2: for each V_k and T_k thread t apply those V_k and
     *                    T_k to E_k for all its chunck, then move to V_k+1
     *                    T_k+1 and apply them to E_k+1 on all the chunck,
     *                    and so on. the difference is that, thread keep V_k
     *                    and T_K while move over the E_k.
     *                    both version look like similar in perf.
     *
     *  THIS IS v1 CODE
     * */

    if(WANTZ==1)
        colpercore =  plasma_ceildiv(NE,2*cores_num);
    else
        colpercore =  plasma_ceildiv(NE,cores_num);

    if(colpercore>1000)
        colpercore =  plasma_ceildiv(colpercore,10);
    else if(colpercore>800)
        colpercore =  plasma_ceildiv(colpercore,8);
    else if(colpercore>600)
        colpercore =  plasma_ceildiv(colpercore,6);
    else if(colpercore>400)
        colpercore =  plasma_ceildiv(colpercore,4);
    else if(colpercore>200)
        colpercore =  plasma_ceildiv(colpercore,2);
    if(colpercore>200)
        colpercore=120;
    if(colpercore<30)
        colpercore=32;
    /*colpercore = N make the code sequential running on thread=0;*/
    nbchunk          =  plasma_ceildiv(NE, colpercore);
    allcoresnb       = cores_num;
    maxrequiredcores = nbchunk;
    if( maxrequiredcores < 1 )
        maxrequiredcores = 1;
    if(allcoresnb > maxrequiredcores)
        allcoresnb = maxrequiredcores;

    /* =========================================
     * case NB = 1  special case.
     * just make the element real for complex
     * =========================================
     */
    if ( NB == 1 ) {
        /* NOTE in CASE USED STANDALONE FUNCTION
         * In COMPLEX matrix Z need to be scaled by the TAU
         * generated during the make off-diagonal elements real
         */
        /*
         * In case where the first stage has been done we are sure
         * that all element of E are real so no need to go through
         * NB=1.  However in case where this function need to be used
         * for only band matrices meaning only stage2 has been called
         * then it require to make all off-diagonal elements real and
         * so remove the return from here and from the bulgechasing
         * function
         */
        if(WANTZ==1){
            plasma_complex64_t zone = 1.0;
            memset(E, 0, LDE*N*sizeof(plasma_complex64_t));
            for(sw=0; sw<NE; sw++)
                E[sw+sw*LDE] = zone;
        }
        if(standalonework==0){
            return;
        }
        else{
#ifdef COMPLEX
            for (chunkid = 0; chunkid<nbchunk; chunkid++) {
                coreid  = chunkid%allcoresnb;
                corest  = chunkid*colpercore;
                corelen = imin(colpercore, (NE-(chunkid*colpercore)));
                if( my_core_id==coreid ){
                    if( side==PlasmaLeft ){
                        for (mycol =1; mycol<NE; mycol++){
                            cblas_zscal(corelen, TAU(mycol), E(mycol, corest), LDE);
                        }
                    }else{
                        plasma_complex64_t ztmp;
                        /*Upper case need to be checked*/
                        for (mycol = corest; mycol<corest+corelen; mycol++){
                            ztmp = conj(*TAU(mycol));
                            cblas_zscal(N, &ztmp, E(0,mycol), 1);
                        }
                    }
                }
            }
#endif
            return;
        }
    }
    /* =========================================
     *       case NB >1  main code
     * =========================================
     */

    /* WANTZ = 1 meaning E is IDENTITY so form Q using optimized update.
     *         So we use the reverse order from small q to large one,
     *         so from q_n to q_1 so Left update to Identity.
     *         Use version 113 because in 114 we need to update the whole matrix and not in icreasing order.
     * WANTZ = 2 meaning E is a full matrix and need to be updated from Left or Right so use normal update
     * */
    if ( WANTZ==1 ) {
        versionL = 113;
        halfchunk = plasma_ceildiv(nbchunk,2);
        for (lchunkid = 0; lchunkid<halfchunk; lchunkid++) {
            rchunkid  = (nbchunk-1) - lchunkid;
            nbportion = lchunkid == rchunkid ? 1 : 2;
            /* sw is the switch between left and right side chunk
             * it is used to have same load distribution of work */

            for (sw = 0; sw<nbportion; sw++) {
                chunkid = sw == 0 ? lchunkid : rchunkid;
                coreid  = lchunkid%allcoresnb;
                corest  = chunkid*colpercore;
                corelen = imin(colpercore, (NE-(chunkid*colpercore)));
                if( my_core_id == coreid ) {
                    /*
                     * Version 113:
                     * loop over the block_col (nt) and for each find the
                     * number of tiles (mt) in this block_col. then loop over mt, find
                     * the size of the V's(Vm,Vn) and apply it to the corresponding
                     * portion of E.
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
                            if((Vm>0)&&(Vn>0)){
                                col = imax(mycol,corest);
                                len = corelen - (col-corest);
                                if(side==PlasmaLeft){
                                    if( len > 0 )
                                        plasma_core_zlarfb_gemm(
                                            side, trans,
                                            PlasmaForward, PlasmaColumnwise,
                                            Vm, len, Vn, V(vpos), LDV, T(tpos),
                                            LDT, E(myrow,col), LDE,
                                            ((plasma_complex64_t*)work.spaces[my_core_id]),
                                            len);
                                }
                                else{
                                    if( len > 0 )
                                        plasma_core_zlarfb_gemm(
                                            side, trans,
                                            PlasmaForward, PlasmaColumnwise,
                                            len, Vm, Vn, V(vpos), LDV, T(tpos), LDT, E(col, myrow), LDE,
                                            ((plasma_complex64_t*)work.spaces[my_core_id]),
                                            len);
                                }
                            }
                        }
                    }
                } /* END my_core_id=coreid */
            } /* END of sw  */
        } /* END loop over the chunk */
    } /* END if WANTZ=1 */
    else{
        /*
         * WANTZ != 1
         */
        for (chunkid = 0; chunkid<nbchunk; chunkid++) {
            coreid  = chunkid%allcoresnb;
            corest  = chunkid*colpercore;
            corelen = imin(colpercore, (NE-(chunkid*colpercore)));
            if(corelen < 0)
                corelen = 0;
            if( my_core_id == coreid ) {
                /*
                 * PlasmaLeft
                 */
                //if( side == PlasmaLeft ) {
                if( versionL == 113 ) {
                    /*
                     * Version 113:
                     * loop over the block_col (nt) and for each find the
                     * number of tiles (mt) in this block_col. then loop over mt, find
                     * the size of the V's(Vm,Vn) and apply it to the corresponding
                     * portion of E.
                     */
                    if( versionL == 113 ) {
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
                                if((Vm>0)&&(Vn>0)){
                                    if( side == PlasmaLeft ){
                                        plasma_core_zlarfb_gemm(
                                            PlasmaLeft, trans,
                                            PlasmaForward, PlasmaColumnwise,
                                            Vm, corelen, Vn, V(vpos), LDV, T(tpos), LDT, E(myrow,corest), LDE,
                                            ((plasma_complex64_t*)work.spaces[my_core_id]),
                                            corelen);

                                    }else{
                                        plasma_core_zlarfb_gemm(
                                            PlasmaRight, trans,
                                            PlasmaForward, PlasmaColumnwise,
                                            corelen, Vm, Vn, V(vpos), LDV, T(tpos), LDT, E(corest, myrow), LDE,
                                            ((plasma_complex64_t*)work.spaces[my_core_id]),
                                            corelen);
                                    }
                                }
                            }
                        }
                    }
                    /*
                     * Version 114:
                     * loop over the block_row (mt) and for each find diagonally the
                     * number of tiles (nt) in this block_row. then loop over nt, find
                     * the size of the V's(Vm,Vn) and apply it to the corresponding
                     * portion of E.
                     */
                    else {
                        mt    = plasma_ceildiv((N-1),NB);
                        for (blki = mt; blki>0; blki--) {
                            /* nbcolinvolvd = number of column corresponding to this block_row (blki) */
                            nbcolinvolvd = imin(N-1, blki*NB);
                            /*find the number of tile for this block (diagonal row of tiles) */
                            nt = plasma_ceildiv(nbcolinvolvd,Vblksiz);
                            /*loop over the tiles find the size of the Vs and apply it */
                            for (blkj = nt-1; blkj>=0; blkj--) {
                                /* the index of the first row of the first col meaning
                                 * the block on the top left (blki) */
                                firstrow = (mt-blki)*NB+1;
                                /*calculate the size of each losange of Vs= (Vm,Vn)*/
                                myrow    = firstrow + blkj*Vblksiz;
                                mycol    = blkj*Vblksiz;
                                Vm = imin( NB+Vblksiz-1, N-myrow);
                                if( ( blkj == nt-1 ) && ( blki == mt ) ){
                                    Vn = imin (Vblksiz, Vm);
                                }else{
                                    Vn = imin (Vblksiz, Vm-1);
                                }
                                /*calculate the pointer to the Vs and the Ts.
                                 * Note that Vs and Ts have special storage done
                                 * by the bulgechasing function*/
                                findVTpos(N,NB,Vblksiz,mycol,myrow, &vpos, &taupos, &tpos, &blkid);
                                if((Vm>0)&&(Vn>0)) {
                                    plasma_core_zlarfb_gemm(
                                        PlasmaLeft, trans,
                                        PlasmaForward, PlasmaColumnwise,
                                        Vm, corelen, Vn, V(vpos), LDV, T(tpos), LDT, E(myrow,corest), LDE,
                                        ((plasma_complex64_t*)work.spaces[my_core_id]),
                                        corelen);
                                }
                            }
                        }
                    }
                }
                /*
                 * PlasmaRight
                 */
                else {
                    /*
                     * Version 91:
                     */
                    if( versionR == 91 ) {
                        nt  = plasma_ceildiv((N-1),Vblksiz);
                        for (blkj=0; blkj<nt; blkj++) {
                            /* the index of the first myrow on the top of block (blkj) */
                            firstrow = blkj * Vblksiz + 1;
                            /*find the number of tile for this block */
                            if( blkj == nt-1 )
                                mt = plasma_ceildiv( N -  firstrow,    NB);
                            else
                                mt = plasma_ceildiv( N - (firstrow+1), NB);
                            /*loop over the tiles find the size of the Vs and apply it */
                            for (blki=1; blki<=mt; blki++) {
                                /*calculate the size of each losange of Vs= (Vm,Vn)*/
                                myrow  = firstrow + (mt-blki)*NB;
                                Vm = imin( NB+Vblksiz-1, N-myrow);
                                if( ( blkj == nt-1 ) && ( blki == mt ) )
                                {
                                    Vn = imin (Vblksiz, Vm);
                                }else{
                                    Vn = imin (Vblksiz, Vm-1);
                                }
                                /*calculate the pointer to the Vs and the Ts.
                                 * Note that Vs and Ts have special storage done
                                 * by the bulgechasing function*/
                                mycol     = blkj*Vblksiz;
                                findVTpos(N,NB,Vblksiz,mycol,myrow, &vpos, &taupos, &tpos, &blkid);
                                if((Vm>0)&&(Vn>0)){
                                    plasma_core_zlarfb_gemm(
                                        PlasmaRight,
                                        trans,
                                        PlasmaForward,
                                        PlasmaColumnwise,
                                        corelen, Vm, Vn, V(vpos), LDV, T(tpos), LDT, E(corest,myrow), LDE,
                                        ((plasma_complex64_t*)work.spaces[my_core_id]),
                                        corelen);
                                }
                            }
                        }
                    }
                    /*trans
                     * Version 92:
                     */
                    else {
                        mt    = plasma_ceildiv((N-1),NB);
                        for (blki = 1; blki<=mt; blki++) {
                            /* nbcolinvolvd = number of column corresponding to this block_row (blki) */
                            nbcolinvolvd = imin(N-1, blki*NB);
                            /*find the number of tile for this block (diagonal row of tiles) */
                            nt = plasma_ceildiv(nbcolinvolvd,Vblksiz);
                            /*loop over the tiles find the size of the Vs and apply it */
                            for (blkj = 0; blkj<nt; blkj++) {
                                /* the index of the first row of the first col meaning
                                 * the block on the top left (blki) */
                                firstrow = (mt-blki)*NB+1;
                                /*calculate the size of each losange of Vs= (Vm,Vn)*/
                                myrow    = firstrow + blkj*Vblksiz;
                                mycol    = blkj*Vblksiz;
                                Vm = imin( NB+Vblksiz-1, N-myrow);
                                if( ( blkj == nt-1 ) && ( blki == mt ) ){
                                    Vn = imin (Vblksiz, Vm);
                                }else{
                                    Vn = imin (Vblksiz, Vm-1);
                                }
                                /*calculate the pointer to the Vs and the Ts.
                                 * Note that Vs and Ts have special storage done
                                 * by the bulgechasing function*/
                                findVTpos(N,NB,Vblksiz,mycol,myrow, &vpos, &taupos, &tpos, &blkid);
                                if((Vm>0)&&(Vn>0)) {
                                    plasma_core_zlarfb_gemm(
                                        PlasmaRight, trans,
                                        PlasmaForward, PlasmaColumnwise,
                                        corelen, Vm, Vn, V(vpos), LDV, T(tpos), LDT, E(corest,myrow), LDE,
                                        ((plasma_complex64_t*)work.spaces[my_core_id]),
                                        corelen);
                                }
                            }
                        }
                    }
                }
            } /* END my_core_id=coreid */
        } /* END loop over the chunk */
    } /* END ELSE of WANTZ == 1 */
}
#undef E
#undef V
#undef T
#undef TAU
/***************************************************************************/
#undef COMPLEX
