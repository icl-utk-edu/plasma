/**
 *
 * @file
 *
 *  Bulge chasing global definitions.
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#ifndef PLASMA_BULGE_H
#define PLASMA_BULGE_H

/***************************************************************************//**
 *  For integers x >= 0, y > 0, returns ceil( x/y ).
 **/
static inline int plasma_ceildiv( int x, int y )
{
    return (x + y - 1)/y;
}

/***************************************************************************//**
 *  Finds the position for a block in the V and tau arrays for bulge chasing.
 *
 *  @param[in] n
 *  @param[in] nb
 *  @param[in] Vblksiz
 *  @param[in] sweep
 *  @param[in] st
 *  @param[out] Vpos
 *  @param[out] TAUpos
 *  @param[out] Tpos
 *  @param[out] myblkid
 *
 **/
static inline void findVTpos(
    int n, int nb, int Vblksiz, int sweep, int st,
    int *Vpos, int *TAUpos, int *Tpos, int *myblkid)
{
    int mastersweep, blkid, locj, ldv;

    int prevcolblknb = 0;
    int prevblkcnt   = 0;
    int curcolblknb  = 0;

    int nbprevcolblk = sweep/Vblksiz;
    for (int prevcolblkid = 0; prevcolblkid < nbprevcolblk; ++prevcolblkid) {
        mastersweep  = prevcolblkid * Vblksiz;
        prevcolblknb = plasma_ceildiv( n - (mastersweep + 2), nb );
        prevblkcnt   = prevblkcnt + prevcolblknb;
    }
    curcolblknb = plasma_ceildiv( st - sweep, nb );
    blkid       = prevblkcnt + curcolblknb - 1;
    locj        = sweep % Vblksiz;
    ldv         = nb + Vblksiz - 1;

    *myblkid= blkid;
    *Vpos   = blkid*Vblksiz*ldv  + locj*ldv + locj;
    *TAUpos = blkid*Vblksiz + locj;
    *Tpos   = blkid*Vblksiz*Vblksiz + locj*Vblksiz + locj;
}

/***************************************************************************//**
 *  Finds the size of the V and tau arrays for bulge chasing.
 *
 *  @param[in] n
 *  @param[in] nb
 *  @param[in] Vblksiz
 *  @param[out] blkcnt
 *  @param[out] ldv
 *
 **/
static inline void findVTsiz(
    int n, int nb, int Vblksiz, int *blkcnt, int *ldv )
{
    int curcolblknb, mastersweep;

    *blkcnt = 0;
    int nbcolblk = plasma_ceildiv( n - 1, Vblksiz );
    for (int colblk = 0; colblk < nbcolblk; ++colblk) {
        mastersweep = colblk * Vblksiz;
        curcolblknb = plasma_ceildiv( n - (mastersweep + 2), nb );
        *blkcnt     = *blkcnt + curcolblknb;
    }
    *blkcnt = *blkcnt + 1;
    *ldv = nb + Vblksiz - 1;
}

#endif // PLASMA_BULGE_H
