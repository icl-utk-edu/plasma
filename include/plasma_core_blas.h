/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#ifndef PLASMA_CORE_BLAS_H
#define PLASMA_CORE_BLAS_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

static const char *lapack_constants[] = {
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",

    "", "", "", "", "", "", "", "", "", "",
    "",
    "NoTrans",                              ///< 111: PlasmaNoTrans
    "Trans",                                ///< 112: PlasmaTrans
    "ConjTrans",                            ///< 113: PlasmaConjTrans

    "", "", "", "", "", "", "",
    "Upper",                                ///< 121: PlasmaUpper
    "Lower",                                ///< 122: PlasmaLower
    "General",                              ///< 123: PlasmaGeneral

    "", "", "", "", "", "", "",
    "NonUnit",                              ///< 131: PlasmaNonUnit
    "Unit",                                 ///< 132: PlasmaUnit

    "", "", "", "", "", "", "", "",
    "Left",                                 ///< 141: PlasmaLeft
    "Right",                                ///< 142: PlasmaRight

    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "",
    "One",                                  ///< 171: PlasmaOneNorm
    "",                                     ///< 172: PlasmaRealOneNorm
    "Two",                                  ///< 173: PlasmaTwoNorm
    "Frobenius",                            ///< 174: PlasmaFrobeniusNorm
    "Infinity",                             ///< 175: PlasmaInfNorm
    "",                                     ///< 176: PlasmaRealInfNorm
    "Maximum",                              ///< 177: PlasmaMaxNorm
    "",                                     ///< 178: PlasmaRealMaxNorm

    "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "",
    "Forward",                             ///< 391: PlasmaForward
    "Backward",                            ///< 392: PlasmaBackward
    "", "", "", "", "", "", "", "",
    "Columnwise",                          ///< 401: PlasmaColumnwise
    "Rowwise"                              ///< 402: PlasmaRowwise
};

/***************************************************************************//**
 * @retval LAPACK character constant corresponding to PLASMA constant
 * @ingroup plasma_const
 ******************************************************************************/
static inline char lapack_const(int plasma_const) {
    return lapack_constants[plasma_const][0];
}

#define plasma_coreblas_error(msg) \
        plasma_coreblas_error_func_line_file(__func__, __LINE__, __FILE__, msg)

static inline void plasma_coreblas_error_func_line_file(
    char const *func, int line, const char *file, const char *msg)
{
    fprintf(stderr,
            "COREBLAS ERROR at %d of %s() in %s: %s\n",
            line, func, file, msg);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#include "plasma_core_blas_s.h"
#include "plasma_core_blas_d.h"
#include "plasma_core_blas_ds.h"
#include "plasma_core_blas_c.h"
#include "plasma_core_blas_z.h"
#include "plasma_core_blas_zc.h"

#endif // PLASMA_CORE_BLAS_H
