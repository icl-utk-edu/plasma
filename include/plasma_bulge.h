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

extern int s_blkcnt;

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
 *  @param[in] first
 *  @param[out] Vpos
 *  @param[out] TAUpos
 *  @param[out] Tpos
 *  @param[out] myblkid
 *
 **/
static inline void findVTpos(
    int n, int nb, int Vblksiz, int sweep, int first,
    int *Vpos, int *TAUpos, int *Tpos, int *blkid)
{
    // printf( "%s: n %3d, nb %3d, vb %3d, sweep %3d, first %3d \n",
    //         __func__, n, nb, Vblksiz, sweep, first );

    // Count the number of V blocks in previous block columns.
    // The (- Vblksiz + 1) term excludes the current block column.
    // The left column of a V block has nb rows, except the bottom V block can
    // have between 2 and nb rows. If it has only 1 row, there's nothing to
    // eliminate. The (n-2) term excludes the top and bottom rows.
    int blkcnt = 0;
    for (int j = 0; j < sweep - Vblksiz + 1; j += Vblksiz) {
        blkcnt += plasma_ceildiv( n - 2 - j, nb );
    }
    // Add the number of whole V blocks above the current V block
    // in the current block column.
    blkcnt += plasma_ceildiv( first - sweep, nb ) - 1;

    int locj = sweep % Vblksiz;   // col and row within this V block.
    int ldv  = nb + Vblksiz - 1;  // height of V block

    *blkid  = blkcnt;
    *Vpos   = blkcnt*Vblksiz*ldv + locj*ldv + locj;
    *TAUpos = blkcnt*Vblksiz + locj;
    *Tpos   = blkcnt*Vblksiz*Vblksiz + locj*Vblksiz + locj;

    // todo: not thread safe
    static int s_n = 0, s_max_blk = 0;
    if (s_n != n) {
        s_n = n;
        s_max_blk = *blkid + 1;
    }
    else if (s_max_blk < *blkid + 1) {
        s_max_blk = *blkid + 1;
        printf( "%s: n %3d, sweep %3d, first %3d, max blkid %3d, s_blkcnt %3d %s\n",
                __func__, n, sweep, first, s_max_blk, s_blkcnt,
                s_max_blk == s_blkcnt ? "tight" : "" );
    }

    if (*blkid + 1 > s_blkcnt) {
        printf( "%s: n %d, nb %d, vb %d, sweep %d, first %d, blkid %d + 1 > s_blkcnt %d outside bounds!\n",
                __func__, n, nb, Vblksiz, sweep, first, *blkid, s_blkcnt );
        assert( 0 );
    }
    //printf( "blkid  %d\n", *blkid );
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
    *blkcnt = *blkcnt + 1;  // why +1?
    *ldv = nb + Vblksiz - 1;


    // Count the number of V blocks in all block columns. See findVTpos.
    int blkcnt_ = 0;
    for (int j = 0; j < n - 2; j += Vblksiz) {
        blkcnt_ += plasma_ceildiv( n - 2 - j, nb );
    }
    // If the last block has 1 element, in complex we need one extra block to
    // make that entry real. (Currently the real case does an extra block, too.)
    if ((n-1) % Vblksiz == 1)
        blkcnt_ += 1;

    if (blkcnt_ != *blkcnt) {
        printf( "%s: n %3d, nb %3d, blkcnt %3d, new blkcnt %3d\n",
                __func__, n, nb, *blkcnt, blkcnt_ );
    }
    //*ldv  = nb + Vblksiz - 1;  // height of V block

    s_blkcnt = blkcnt_;
    //printf( "blkcnt %d\n", *blkcnt );
}

#endif // PLASMA_BULGE_H
