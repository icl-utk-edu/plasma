/**
 *
 * @file plasma.h
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
#ifndef PLASMA_H
#define PLASMA_H

#include <complex.h>

#ifdef PLASMA_WITH_MKL
    #include <mkl_cblas.h>
#else
    #include <cblas.h>
#endif

/******************************************************************************/
typedef int PLASMA_enum;
typedef int PLASMA_bool;

typedef float  complex PLASMA_Complex32_t;
typedef double complex PLASMA_Complex64_t;

/******************************************************************************/
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
    PLASMA_SUCCESS = 0,
    PLASMA_ERR_ILLEGAL_VALUE,
    PLASMA_ERR_NOT_INITIALIZED,
    PLASMA_ERR_SEQUENCE_FLUSHED,
    PLASMA_ERR_UNALLOCATED
};

enum {
    PLASMA_INPLACE,
    PLASMA_OUTOFPLACE
};

/******************************************************************************/
#ifndef CBLAS_SADDR
#define CBLAS_SADDR(var) &(var)
#endif

/******************************************************************************/
typedef struct {
    PLASMA_bool status; ///< error code
} PLASMA_request;

static const PLASMA_request PLASMA_REQUEST_INITIALIZER = {PLASMA_SUCCESS};

typedef struct {
    PLASMA_bool status;      ///< error code
    PLASMA_request *request; ///< failed request
} PLASMA_sequence;

/******************************************************************************/

#include "../control/descriptor.h"

#include "plasma_s.h"
#include "plasma_d.h"
#include "plasma_c.h"
#include "plasma_z.h"

#endif //  #ifndef PLASMA_H
