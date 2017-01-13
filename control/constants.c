/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/

#include "plasma_types.h"

#include <assert.h>
#include <stdbool.h>

/***************************************************************************//**
 * @addtogroup plasma_const
 * Convert LAPACK character constants to PLASMA constants.
 * This is a one-to-many mapping, requiring multiple translators
 * (e.g., "N" can be NoTrans or NonUnit or NoVec).
 * Matching is case-insensitive.
 * @{
 ******************************************************************************/

// These functions and cases are in the same order as the constants are
// declared in plasma_types.h

/***************************************************************************//**
 * @retval PlasmaNonUnit if lapack_char = 'N'
 * @retval PlasmaUnit    if lapack_char = 'U'
 ******************************************************************************/
plasma_enum_t plasma_diag_const(char lapack_char)
{
    switch (lapack_char) {
    case 'N': case 'n': return PlasmaNonUnit;
    case 'U': case 'u': return PlasmaUnit;
    default:            return PlasmaInvalid;
    }
}

/***************************************************************************//**
 * @retval PlasmaForward  if lapack_char = 'F'
 * @retval PlasmaBackward if lapack_char = 'B'
 ******************************************************************************/
plasma_enum_t plasma_direct_const(char lapack_char)
{
    switch (lapack_char) {
    case 'F': case 'f': return PlasmaForward;
    case 'B': case 'b': return PlasmaBackward;
    default:            return PlasmaInvalid;
    }
}

/***************************************************************************//**
 * @retval PlasmaOneNorm       if lapack_char = 'O|o|1'
 * @retval PlasmaTwoNorm       if lapack_char = '2'
 * @retval PlasmaFrobeniusNorm if lapack_char = 'F|f|E|e'
 * @retval PlasmaInfNorm       if lapack_char = 'I|i'
 * @retval PlasmaMaxNorm       if lapack_char = 'M|m'
 ******************************************************************************/
plasma_enum_t plasma_norm_const(char lapack_char)
{
    switch (lapack_char) {
    case 'O': case 'o': case '1':           return PlasmaOneNorm;
    case '2':                               return PlasmaTwoNorm;
    case 'F': case 'f': case 'E': case 'e': return PlasmaFrobeniusNorm;
    case 'I': case 'i':                     return PlasmaInfNorm;
    case 'M': case 'm':                     return PlasmaMaxNorm;
    // MagmaRealOneNorm
    // MagmaRealInfNorm
    // MagmaRealMaxNorm
    default:                                return PlasmaInvalid;
    }
}

/***************************************************************************//**
 * @retval PlasmaLeft      if lapack_char = 'L'
 * @retval PlasmaRight     if lapack_char = 'R'
 ******************************************************************************/
// @retval PlasmaBothSides if lapack_char = 'B'  // for trevc
plasma_enum_t plasma_side_const(char lapack_char)
{
    switch (lapack_char) {
    case 'L': case 'l': return PlasmaLeft;
    case 'R': case 'r': return PlasmaRight;
    //case 'B': case 'b': return PlasmaBothSides;  // for trevc
    default:            return PlasmaInvalid;
    }
}

/***************************************************************************//**
 * @retval PlasmaColumnwise if lapack_char = 'C'
 * @retval PlasmaRowwise    if lapack_char = 'R'
 ******************************************************************************/
plasma_enum_t plasma_storev_const(char lapack_char)
{
    switch (lapack_char) {
    case 'C': case 'c': return PlasmaColumnwise;
    case 'R': case 'r': return PlasmaRowwise;
    default:            return PlasmaInvalid;
    }
}

/***************************************************************************//**
 * @retval PlasmaNoTrans   if lapack_char = 'N'
 * @retval PlasmaTrans     if lapack_char = 'T'
 * @retval PlasmaConjTrans if lapack_char = 'C'
 ******************************************************************************/
plasma_enum_t plasma_trans_const(char lapack_char)
{
    switch (lapack_char) {
    case 'N': case 'n': return PlasmaNoTrans;
    case 'T': case 't': return PlasmaTrans;
    case 'C': case 'c': return PlasmaConjTrans;
    default:            return PlasmaInvalid;
    }
}

/***************************************************************************//**
 * @retval PlasmaUpper   if lapack_char = 'U'
 * @retval PlasmaLower   if lapack_char = 'L'
 * @retval PlasmaGeneral otherwise
 ******************************************************************************/
plasma_enum_t plasma_uplo_const(char lapack_char)
{
    switch (lapack_char) {
    case 'U': case 'u': return PlasmaUpper;
    case 'L': case 'l': return PlasmaLower;
    default:            return PlasmaGeneral;
    }
}

/***************************************************************************//**
 * @}
 * end group plasma_const
 ******************************************************************************/
