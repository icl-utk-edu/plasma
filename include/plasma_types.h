/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#ifndef PLASMA_TYPES_H
#define PLASMA_TYPES_H

#include <complex.h>
#include <limits.h>

/*
 * RELEASE is a, b, c
 */
#define PLASMA_VERSION_MAJOR 25
#define PLASMA_VERSION_MINOR 5
#define PLASMA_VERSION_PATCH 27

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************/
#if defined(PLASMA_HAVE_MKL) || defined(PLASMA_WITH_MKL)
#define lapack_complex_float plasma_complex32_t
#define lapack_complex_double plasma_complex64_t
#endif

/***************************************************************************//**
 *
 *  Some CBLAS routines take scalars by value in real arithmetic
 *  and by pointer in complex arithmetic.
 *  In precision generation, CBLAS_SADDR is removed from real arithmetic files.
 *
 **/
#ifndef CBLAS_SADDR
#if defined(PLASMA_WITH_OPENBLAS)
#define CBLAS_SADDR(var) ((void*)&(var))
#else
#define CBLAS_SADDR(var) &(var)
#endif
#endif

/******************************************************************************/
enum {
    PlasmaByte          = 0,
    PlasmaInteger       = 1,
    PlasmaRealFloat     = 2,
    PlasmaRealDouble    = 3,
    PlasmaComplexFloat  = 4,
    PlasmaComplexDouble = 5,
    PlasmaDatatypeUnknown = INT_MIN // ensure int storage type in C++
};

/***************************************************************************//**
 *
 *  PLASMA constants - CBLAS & LAPACK.
 *  The naming and numbering is consistent with:
 *
 *    - CBLAS - http://www.netlib.org/blas/blast-forum/cblas.tgz,
 *    - LAPACKE - http://www.netlib.org/lapack/lapwrapc/.
 *
 *  During precision generation, Plasma_ConjTrans is conveted to PlasmaTrans,
 *  while PlasmaConjTrans is preserved.
 *
 **/
enum {
    PlasmaInvalid       = -1,

    PlasmaNoTrans       = 111,
    PlasmaTrans         = 112,
    PlasmaConjTrans     = 113,
    Plasma_ConjTrans    = PlasmaConjTrans,

    PlasmaUpper         = 121,
    PlasmaLower         = 122,
    PlasmaGeneral       = 123,
    PlasmaGeneralBand   = 124,

    PlasmaNonUnit       = 131,
    PlasmaUnit          = 132,

    PlasmaLeft          = 141,
    PlasmaRight         = 142,

    PlasmaOneNorm       = 171,
    PlasmaRealOneNorm   = 172,
    PlasmaTwoNorm       = 173,
    PlasmaFrobeniusNorm = 174,
    PlasmaInfNorm       = 175,
    PlasmaRealInfNorm   = 176,
    PlasmaMaxNorm       = 177,
    PlasmaRealMaxNorm   = 178,

    PlasmaNoVec         = 301,
    PlasmaVec           = 302,
    PlasmaCount         = 303,
    PlasmaIVec          = 304,
    PlasmaAllVec        = 305,
    PlasmaSomeVec       = 306,

    PlasmaRangeAll      = 351,
    PlasmaRangeV        = 352,
    PlasmaRangeI        = 353,

    PlasmaForward       = 391,
    PlasmaBackward      = 392,

    PlasmaColumnwise    = 401,
    PlasmaRowwise       = 402,

    PlasmaW             = 501,
    PlasmaA2            = 502,
    Plasma_Const_Limit,  // Ensure always last.

    PlasmaConstUnknown = INT_MIN // ensure int storage type in C++
};

enum {
    PlasmaSuccess = 0,
    /* 20 negative values reserved for errors with invalid parameters,
       and additional integer ranges may be needed in the future */
    PlasmaErrorNotInitialized = -101,
    PlasmaErrorNotSupported   = -102,
    PlasmaErrorIllegalValue   = -103,
    PlasmaErrorOutOfMemory    = -104,
    PlasmaErrorNullParameter  = -105,
    PlasmaErrorInternal       = -106,
    PlasmaErrorSequence       = -107,
    PlasmaErrorComponent      = -108,
    PlasmaErrorEnvironment    = -109,
    PlasmaErrorUnknown = INT_MIN // ensure int storage type in C++
};

enum {
    PlasmaInplace,
    PlasmaOutplace,
    PlasmaInoutUnknown = INT_MAX // ensure int storage type in C++
};

enum {
    PlasmaFlatHouseholder,
    PlasmaTreeHouseholder,
    PlasmaHouseholderUnknown = INT_MAX // ensure int storage type in C++
};

enum {
    PlasmaDisabled = 0,
    PlasmaEnabled = 1,
    PlasmaAbledUnknown = INT_MAX // ensure int storage type in C++
};

enum {
    PlasmaTuning,
    PlasmaNb,
    PlasmaIb,
    PlasmaInplaceOutplace,
    PlasmaNumPanelThreads,
    PlasmaHouseholderMode,
    PlasmaParamUnknown = INT_MAX // ensure int storage type in C++
};

/******************************************************************************/
typedef int plasma_enum_t;

typedef float  _Complex plasma_complex32_t;
typedef double _Complex plasma_complex64_t;

// Define type that the precision generator will not change, for timing.
typedef double plasma_time_t;

/******************************************************************************/
plasma_enum_t plasma_eigt_const(char lapack_char);
plasma_enum_t plasma_job_const(char lapack_char);
plasma_enum_t plasma_range_const(char lapack_char);
plasma_enum_t plasma_diag_const(char lapack_char);
plasma_enum_t plasma_direct_const(char lapack_char);
plasma_enum_t plasma_norm_const(char lapack_char);
plasma_enum_t plasma_side_const(char lapack_char);
plasma_enum_t plasma_storev_const(char lapack_char);
plasma_enum_t plasma_trans_const(char lapack_char);
plasma_enum_t plasma_uplo_const(char lapack_char);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // PLASMA_TYPES_H
