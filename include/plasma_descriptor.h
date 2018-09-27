/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#ifndef PLASMA_DESCRIPTOR_H
#define PLASMA_DESCRIPTOR_H

#include "plasma_types.h"
#include "plasma_error.h"

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
    // matrix properties
    plasma_enum_t type;      ///< general, general band, etc.
    plasma_enum_t uplo;      ///< upper, lower, etc.
    plasma_enum_t precision; ///< precision of the matrix

    // pointer and offsets
    void *matrix; ///< pointer to the beginning of the matrix
    size_t A21;   ///< pointer to the beginning of A21
    size_t A12;   ///< pointer to the beginning of A12
    size_t A22;   ///< pointer to the beginning of A22

    // tile parameters
    int mb; ///< number of rows in a tile
    int nb; ///< number of columns in a tile

    // main matrix parameters
    int gm;  ///< number of rows of the entire matrix
    int gn;  ///< number of columns of the entire matrix
    int gmt; ///< number of tile rows of the entire matrix
    int gnt; ///< number of tile columns of the entire matrix

    // submatrix parameters
    int i;  ///< row index to the beginning of the submatrix
    int j;  ///< column index to the beginning of the submatrix
    int m;  ///< number of rows of the submatrix
    int n;  ///< number of columns of the submatrix
    int mt; ///< number of tile rows of the submatrix
    int nt; ///< number of tile columns of the submatrix

    // submatrix parameters for a band matrix
    int kl;  ///< number of rows below the diagonal
    int ku;  ///< number of rows above the diagonal
    int klt; ///< number of tile rows below the diagonal tile
    int kut; ///< number of tile rows above the diagonal tile
             ///  includes the space for potential fills, i.e., kl+ku
} plasma_desc_t;

/******************************************************************************/
static inline size_t plasma_element_size(int type)
{
    switch (type) {
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
static inline void *plasma_tile_addr_general(plasma_desc_t A, int m, int n)
{
    int mm = m + A.i/A.mb;
    int nn = n + A.j/A.nb;
    size_t eltsize = plasma_element_size(A.precision);
    size_t offset = 0;

    int lm1 = A.gm/A.mb;
    int ln1 = A.gn/A.nb;

    if (mm < lm1)
        if (nn < ln1)
            offset = A.mb*A.nb*(mm + (size_t)lm1 * nn);
        else
            offset = A.A12 + ((size_t)A.mb * (A.gn%A.nb) * mm);
    else
        if (nn < ln1)
            offset = A.A21 + ((size_t)A.nb * (A.gm%A.mb) * nn);
        else
            offset = A.A22;

    return (void*)((char*)A.matrix + (offset*eltsize));
}

/******************************************************************************/
static inline void *plasma_tile_addr_triangle(plasma_desc_t A, int m, int n)
{
    int mm = m + A.i/A.mb;
    int nn = n + A.j/A.nb;
    size_t eltsize = plasma_element_size(A.precision);
    size_t offset = 0;

    int lm1 = A.gm/A.mb;
    int ln1 = A.gn/A.nb;

    if (mm < lm1) {
        if (nn < ln1) {
            if (A.type == PlasmaUpper) {
                offset = A.mb*A.nb*(mm + (nn * (nn + 1))/2);
            }
            else {
                offset = A.mb*A.nb*((mm - nn) + (nn * (2*lm1 - (nn-1)))/2);
            }
        }
        else {
            offset = A.A12 + ((size_t)A.mb * (A.gn%A.nb) * mm);
        }
    }
    else {
        if (nn < ln1) {
            offset = A.A21 + ((size_t)A.nb * (A.gm%A.mb) * nn);
        }
        else {
            offset = A.A22;
        }
    }

    return (void*)((char*)A.matrix + (offset*eltsize));
}

/******************************************************************************/
static inline void *plasma_tile_addr_general_band(plasma_desc_t A, int m, int n)
{
    return plasma_tile_addr_general(A, (A.kut-1)+m-n, n);
}

/******************************************************************************/
static inline void *plasma_tile_addr(plasma_desc_t A, int m, int n)
{
    if (A.type == PlasmaGeneral) {
        return plasma_tile_addr_general(A, m, n);
    }
    else if (A.type == PlasmaGeneralBand) {
        return plasma_tile_addr_general_band(A, m, n);
    }
    else if (A.type == PlasmaUpper || A.type == PlasmaLower) {
        return plasma_tile_addr_triangle(A, m, n);
    }
    else {
        plasma_fatal_error("invalid matrix type");
        return NULL;
    }
}

/***************************************************************************//**
 *
 *  Returns the height of the tile with vertical position k.
 *
 */
static inline int plasma_tile_mmain(plasma_desc_t A, int k)
{
    if (A.type == PlasmaGeneralBand) {
        return A.mb;
    }
    else {
        if (A.i/A.mb+k < A.gm/A.mb)
            return A.mb;
        else
            return A.gm%A.mb;
    }
}

/***************************************************************************//**
 *
 *  Returns the width of the tile with horizontal position k.
 *
 */
static inline int plasma_tile_nmain(plasma_desc_t A, int k)
{
    if (A.j/A.nb+k < A.gn/A.nb)
        return A.nb;
    else
        return A.gn%A.nb;
}

/***************************************************************************//**
 *
 *  Returns the height of the portion of the submatrix occupying the tile
 *  at vertical position k.
 *
 */
static inline int plasma_tile_mview(plasma_desc_t A, int k)
{
    if (k < A.mt-1)
        return A.mb;
    else
        if ((A.i+A.m)%A.mb == 0)
            return A.mb;
        else
            return (A.i+A.m)%A.mb;
}

/***************************************************************************//**
 *
 *  Returns the width of the portion of the submatrix occupying the tile
 *  at horizontal position k.
 *
 */
static inline int plasma_tile_nview(plasma_desc_t A, int k)
{
    if (k < A.nt-1)
        return A.nb;
    else
        if ((A.j+A.n)%A.nb == 0)
            return A.nb;
        else
            return (A.j+A.n)%A.nb;
}

/******************************************************************************/
static inline int plasma_tile_mmain_band(plasma_desc_t A, int m, int n)
{
    return plasma_tile_mmain(A, (A.kut-1)+m-n);
}

/******************************************************************************/
int plasma_desc_general_create(plasma_enum_t dtyp, int mb, int nb,
                               int lm, int ln, int i, int j, int m, int n,
                               plasma_desc_t *A);

int plasma_desc_general_band_create(plasma_enum_t dtyp, plasma_enum_t uplo,
                                    int mb, int nb, int lm, int ln,
                                    int i, int j, int m, int n, int kl, int ku,
                                    plasma_desc_t *A);

int plasma_desc_triangular_create(plasma_enum_t dtyp, plasma_enum_t uplo, int mb, int nb,
                                  int lm, int ln, int i, int j, int m, int n,
                                  plasma_desc_t *A);

int plasma_desc_destroy(plasma_desc_t *A);

int plasma_desc_general_init(plasma_enum_t precision, void *matrix,
                             int mb, int nb, int lm, int ln, int i, int j,
                             int m, int n, plasma_desc_t *A);

int plasma_desc_general_band_init(plasma_enum_t precision, plasma_enum_t uplo,
                                  void *matrix, int mb, int nb, int lm, int ln,
                                  int i, int j, int m, int n, int kl, int ku,
                                  plasma_desc_t *A);

int plasma_desc_triangular_init(plasma_enum_t precision, plasma_enum_t uplo, void *matrix,
                                int mb, int nb, int lm, int ln, int i, int j,
                                int m, int n, plasma_desc_t *A);

int plasma_desc_check(plasma_desc_t A);
int plasma_desc_general_check(plasma_desc_t A);
int plasma_desc_general_band_check(plasma_desc_t A);

plasma_desc_t plasma_desc_view(plasma_desc_t A, int i, int j, int m, int n);

int plasma_descT_create(plasma_desc_t A, int ib, plasma_enum_t householder_mode,
                        plasma_desc_t *T);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // PLASMA_DESCRIPTOR_H
