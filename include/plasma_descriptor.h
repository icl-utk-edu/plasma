/**
 *
 * @file plasma_descriptor.h
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
#ifndef ICL_PLASMA_DESCRIPTOR_H
#define ICL_PLASMA_DESCRIPTOR_H

#include "plasma_types.h"

#include <stdlib.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

/***************************************************************************//**
 * @ingroup plasma_descriptor
 *
 * Tile matrix descriptor.
 *
 *              n1      n2
 *         +----------+---+
 *         |          |   |    m1 = lm - (lm%mb)
 *         |          |   |    m2 = lm%mb
 *     m1  |    A11   |A12|    n1 = ln - (ln%nb)
 *         |          |   |    n2 = ln%nb
 *         |          |   |
 *         +----------+---+
 *     m2  |    A21   |A22|
 *         +----------+---+
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

/******************************************************************************/
static inline int plasma_element_size(int type)
{
    switch(type) {
    case PlasmaByte:          return          1;
    case PlasmaInteger:       return   sizeof(int);
    case PlasmaRealFloat:     return   sizeof(float);
    case PlasmaRealDouble:    return   sizeof(double);
    case PlasmaComplexFloat:  return 2*sizeof(float);
    case PlasmaComplexDouble: return 2*sizeof(double);
    default: assert(0);
    }
}

/******************************************************************************/
static inline int BLKLDD(PLASMA_desc A, int k)
{
    if (k+A.i/A.mb < A.lm1)
        return A.mb;
    else
        return A.lm%A.mb;
}

/******************************************************************************/
static inline void *plasma_getaddr(PLASMA_desc A, int m, int n)
{
    int mm = m + A.i/A.mb;
    int nn = n + A.j/A.nb;
    size_t eltsize = plasma_element_size(A.dtyp);
    size_t offset = 0;

    if (mm < A.lm1)
        if (nn < A.ln1)
            offset = A.bsiz*(mm + (size_t)A.lm1 * nn);
        else
            offset = A.A12 + ((size_t)A.mb * (A.ln%A.nb) * mm);
    else
        if (nn < A.ln1)
            offset = A.A21 + ((size_t)A.nb * (A.lm%A.mb) * nn);
        else
            offset = A.A22;

    return (void*)((char*)A.mat + (offset*eltsize));
}

/******************************************************************************/
int PLASMA_Desc_Create(PLASMA_desc **desc, void *mat, PLASMA_enum dtyp,
                       int mb, int nb, int bsiz, int lm, int ln, int i,
                       int j, int m, int n);

int PLASMA_Desc_Destroy(PLASMA_desc **desc);

PLASMA_desc plasma_desc_init(PLASMA_enum dtyp, int mb, int nb, int bsiz,
                             int lm, int ln, int i, int j, int m, int n);

PLASMA_desc plasma_desc_submatrix(PLASMA_desc descA, int i, int j, int m, int n);

int plasma_desc_check(PLASMA_desc *desc);
int plasma_desc_mat_alloc(PLASMA_desc *desc);
int plasma_desc_mat_free(PLASMA_desc *desc);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_PLASMA_DESCRIPTOR_H
