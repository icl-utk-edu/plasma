/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
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
    plasma_enum_t dtyp; ///< precision of the matrix
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
    int kl;           ///< number of rows below the diagonal
    int ku;           ///< number of rows above the diagonal
    int klt;          ///< number of tile rows below the diagonal tile
    int kut;          ///< number of tile rows above the diagonal tile
                      ///  includes the space for potential fills, i.e., kl+ku
    plasma_enum_t status;
} plasma_desc_t;

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
static inline int BLKLDD(plasma_desc_t A, int k)
{
    if (k+A.i/A.mb < A.lm1)
        return A.mb;
    else
        return A.lm%A.mb;
}

/******************************************************************************/
static inline void *plasma_getaddr(plasma_desc_t A, int m, int n)
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
static inline int BLKLDD_BAND(plasma_enum_t uplo,
                              plasma_desc_t A, int m, int n)
{
    int kut;
    if (uplo == PlasmaFull) {
        kut = (A.kl+A.kl+A.nb-1)/A.nb;
    }
    else if (uplo == PlasmaUpper) {
        kut = (A.ku+A.nb-1)/A.nb;
    }
    else {
        kut = 0;
    }
    return BLKLDD(A, kut+m-n);
}

/******************************************************************************/
static inline void *plasma_getaddr_band(plasma_enum_t uplo,
                                        plasma_desc_t A, int m, int n)
{
    int kut;
    if (uplo == PlasmaFull) {
        kut = (A.kl+A.kl+A.nb-1)/A.nb;
    }
    else if (uplo == PlasmaUpper) {
        kut = (A.ku+A.nb-1)/A.nb;
    }
    else {
        kut = 0;
    }
    return plasma_getaddr(A, kut+m-n, n);
}

/******************************************************************************/
int plasma_desc_create(plasma_enum_t dtyp, int mb, int nb, int lm, int ln,
                       int i, int j, int m, int n, plasma_desc_t *desc);

int plasma_desc_band_create(plasma_enum_t dtyp, plasma_enum_t uplo,
                            int mb, int nb, int lm, int ln, int i, int j,
                            int m, int n, int kl, int ku, plasma_desc_t *desc);

int plasma_desc_destroy(plasma_desc_t *desc);

plasma_desc_t plasma_desc_init(plasma_enum_t dtyp, int mb, int nb, int bsiz,
                             int lm, int ln, int i, int j, int m, int n);


plasma_desc_t plasma_desc_band_init(plasma_enum_t dtyp, plasma_enum_t uplo,
                                  int mb, int nb, int bsiz,
                                  int lm, int ln, int i, int j, int m, int n,
                                  int kl, int ku);

int plasma_desc_check(plasma_desc_t *desc);
int plasma_desc_band_check(plasma_enum_t uplo, plasma_desc_t *desc);

plasma_desc_t plasma_desc_view(plasma_desc_t descA, int i, int j, int m, int n);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_PLASMA_DESCRIPTOR_H
