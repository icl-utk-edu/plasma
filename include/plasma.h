#ifndef PLASMA_H
#define PLASMA_H

/**
 *
 * @file plasma.h
 *
 *  PLASMA headers.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver.
 *
 * @version 3.0.0
 * @author Jakub Kurzak
 * @date 2016-01-01
 *
 **/
#include <complex.h>
#include <cblas.h>

/******************************************************************************/
typedef int PLASMA_enum;

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

/******************************************************************************/
#ifndef CBLAS_SADDR
#define CBLAS_SADDR(var) &(var)
#endif

typedef enum CBLAS_SIDE      CBLAS_SIDE;
typedef enum CBLAS_TRANSPOSE CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO      CBLAS_UPLO;

/******************************************************************************/
typedef struct {
    
} PLASMA_desc;

typedef struct {
    
} PLASMA_sequence;

typedef struct {
    
} PLASMA_request;

#endif        //  #ifndef PLASMA_H
