/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#ifndef ICL_PLASMA_TYPES_H
#define ICL_PLASMA_TYPES_H

#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************/
#ifdef PLASMA_WITH_MKL
#define lapack_complex_float plasma_complex32_t
#define lapack_complex_double plasma_complex64_t
#endif

/******************************************************************************/
#ifndef CBLAS_SADDR
#define CBLAS_SADDR(var) &(var)
#endif

/******************************************************************************/
enum {
    PlasmaByte          = 0,
    PlasmaInteger       = 1,
    PlasmaRealFloat     = 2,
    PlasmaRealDouble    = 3,
    PlasmaComplexFloat  = 4,
    PlasmaComplexDouble = 5
};

/***************************************************************************//**
 *
 *  PLASMA constants - CBLAS & LAPACK
 *  The naming and numbering is consistent with:
 *
 *    1) CBLAS from Netlib (http://www.netlib.org/blas/blast-forum/cblas.tgz),
 *    2) C Interface to LAPACK from Netlib (http://www.netlib.org/lapack/lapwrapc/).
 *
 **/
enum {
    PlasmaInvalid    = -1,

    PlasmaNoTrans    = 111,
    PlasmaTrans      = 112,
    PlasmaConjTrans  = 113,
    Plasma_ConjTrans = PlasmaConjTrans,

    PlasmaUpper      = 121,
    PlasmaLower      = 122,
    PlasmaFull       = 123,  // formerly PlasmaUpperLower

    PlasmaNonUnit    = 131,
    PlasmaUnit       = 132,

    PlasmaLeft       = 141,
    PlasmaRight      = 142,

    PlasmaForward    = 391,
    PlasmaBackward   = 392,

    PlasmaColumnwise = 401,
    PlasmaRowwise    = 402,

    PlasmaW          = 501,
    PlasmaA2         = 502
};

enum {
    PLASMA_SUCCESS              =    0,
    PLASMA_ERR_NOT_INITIALIZED  = -101,
    PLASMA_ERR_REINITIALIZED    = -102,
    PLASMA_ERR_NOT_SUPPORTED    = -103,
    PLASMA_ERR_ILLEGAL_VALUE    = -104,
    PLASMA_ERR_NOT_FOUND        = -105,
    PLASMA_ERR_OUT_OF_RESOURCES = -106,
    PLASMA_ERR_INTERNAL_LIMIT   = -107,
    PLASMA_ERR_UNALLOCATED      = -108,
    PLASMA_ERR_FILESYSTEM       = -109,
    PLASMA_ERR_UNEXPECTED       = -110,
    PLASMA_ERR_SEQUENCE_FLUSHED = -111
};

enum {
    PLASMA_INPLACE,
    PLASMA_OUTOFPLACE
};

enum {
    PLASMA_TILE_SIZE,
    PLASMA_INNER_BLOCK_SIZE,
    PLASMA_TRANSLATION_MODE
};

/******************************************************************************/
typedef int PLASMA_enum;
typedef int PLASMA_bool;

typedef float  _Complex plasma_complex32_t;
typedef double _Complex plasma_complex64_t;

/******************************************************************************/
PLASMA_enum PLASMA_trans_const(char lapack_char);
PLASMA_enum PLASMA_uplo_const(char lapack_char);
PLASMA_enum PLASMA_diag_const(char lapack_char);
PLASMA_enum PLASMA_side_const(char lapack_char);
PLASMA_enum PLASMA_direct_const(char lapack_char);
PLASMA_enum PLASMA_storev_const(char lapack_char);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_PLASMA_TYPES_H
