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
    // Count the number of V blocks in previous block columns.
    // The left column of a V block has nb rows, except the bottom V block can
    // have between 2 and nb rows. If there's only 1 row, there's nothing to
    // eliminate. The n-2 excludes the top and bottom rows.
    int blkcnt = 0;
    for (int j = 0; j < sweep - Vblksiz + 1; j += Vblksiz) {
        blkcnt += plasma_ceildiv( n - 2 - j, nb );
    }
    // Add the number of whole V blocks above the current V block in the current
    // block column.
    blkcnt += plasma_ceildiv( first - sweep, nb ) - 1;

    int locj = sweep % Vblksiz;   // col and row within this V block.
    int ldv  = nb + Vblksiz - 1;  // height of V block

    *blkid  = blkcnt;
    *Vpos   = blkcnt*Vblksiz*ldv + locj*ldv + locj;
    *TAUpos = blkcnt*Vblksiz + locj;
    *Tpos   = blkcnt*Vblksiz*Vblksiz + locj*Vblksiz + locj;
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
    // Count the number of V blocks in all block columns. See findVTpos.
    int blkcnt_ = 0;
    for (int j = 0; j < n - 1; j += Vblksiz) {
        blkcnt_ += plasma_ceildiv( n - 2 - j, nb );
    }
    *blkcnt = blkcnt_ + 1;
    *ldv  = nb + Vblksiz - 1;  // height of V block
}

#endif // PLASMA_BULGE_H
