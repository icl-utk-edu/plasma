/**
 *
 * @file plasmatypes.h
 *
 *  PLASMA header.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver.
 *
 * @version 3.0.0
 * @author Jakub Kurzak
 * @date 2016-01-01
 *
 **/
#ifndef PLASMATYPES_H
#define PLASMATYPES_H

#include <complex.h>

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

enum {
    PlasmaNoTrans    = 111,
    PlasmaTrans      = 112,
    PlasmaConjTrans  = 113,

    PlasmaUpper      = 121,
    PlasmaLower      = 122,
    PlasmaUpperLower = 123,

    PlasmaNonUnit    = 131,
    PlasmaUnit       = 132,

    PlasmaLeft       = 141,
    PlasmaRight      = 142
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

/******************************************************************************/
typedef int PLASMA_enum;
typedef int PLASMA_bool;

typedef float  complex PLASMA_Complex32_t;
typedef double complex PLASMA_Complex64_t;

/******************************************************************************/
typedef struct {
    PLASMA_bool status; ///< error code
} PLASMA_request;

static const PLASMA_request PLASMA_REQUEST_INITIALIZER = {PLASMA_SUCCESS};

typedef struct {
    PLASMA_bool status;      ///< error code
    PLASMA_request *request; ///< failed request
} PLASMA_sequence;

#endif // PLASMATYPES_H
