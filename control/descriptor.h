/**
 *
 * @file context.h
 *
 *  PLASMA control routines.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver.
 *
 * @version 3.0.0
 * @author Jakub Kurzak
 * @date 2016-01-01
 *
 **/
#ifndef DESCRIPTOR_H
#define DESCRIPTOR_H

/***************************************************************************//**
 *  Tile matrix descriptor.
 *
 *           n1      n2
 *      +----------+---+
 *      |          |   |    m1 = lm - (lm%mb)
 *      |          |   |    m2 = lm%mb
 *  m1  |    A11   |A12|    n1 = ln - (ln%nb)
 *      |          |   |    n2 = ln%nb
 *      |          |   |
 *      +----------+---+
 *  m2  |    A21   |A22|
 *      +----------+---+
 *
 **/
typedef struct {
    void *mat;        ///< pointer to the beginning of the matrix
    size_t A21;       ///< pointer to the beginning of A21
    size_t A12;       ///< pointer to the beginning of A12
    size_t A22;       ///< pointer to the beginning of A22
    PLASMA_enum dtyp; ///< precision of the matrix
    int mb;           ///< number of rows in a tile
    int nb;           ///< number of columns in a tile
    int bsiz;         ///< size in elements
    int lm;           ///< number of rows of the entire matrix
    int ln;           ///< number of columns of the entire matrix
    int lm1;          ///< number of tile rows of A11
    int ln1;          ///< number of tile columns of A11
    int lmt;          ///< number of tile rows of the entire matrix
    int lnt;          ///< number of tile columns of the entire matrix
    int i;            ///< row index to the beginning of the submatrix
    int j;            ///< column index to the beginning of the submatrix
    int m;            ///< number of rows of the submatrix
    int n;            ///< number of columns of the submatrix
    int mt;           ///< number of tile rows of the submatrix
    int nt;           ///< number of tile columns of the submatrix
} PLASMA_desc;

#endif DESCRIPTOR_H
