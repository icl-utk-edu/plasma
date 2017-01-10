/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> c
 *
 **/

#include "core_blas.h"

#include <math.h>

/***************************************************************************//**
 *
 * @ingroup core_cabs1
 *
 *******************************************************************************
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 *******************************************************************************
 *
 * @retval Complex 1-norm absolute value: abs(real(alpha)) + abs(imag(alpha)).
 *
 *******************************************************************************
 *
 * @sa core_scabs1
 *
 ******************************************************************************/
double core_dcabs1(plasma_complex64_t alpha)
{
    return fabs(creal(alpha)) + fabs(cimag(alpha));
}
