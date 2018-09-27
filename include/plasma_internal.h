/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#ifndef PLASMA_INTERNAL_H
#define PLASMA_INTERNAL_H

#if ((__GNUC__ == 6) && (__GNUC_MINOR__ < 1)) || (__GNUC__ < 6)
  #define priority(p)
#endif

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************/
static inline int imin(int a, int b)
{
    if (a < b)
        return a;
    else
        return b;
}

/******************************************************************************/
static inline int imax(int a, int b)
{
    if (a > b)
        return a;
    else
        return b;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#include "plasma_internal_s.h"
#include "plasma_internal_d.h"
#include "plasma_internal_ds.h"
#include "plasma_internal_c.h"
#include "plasma_internal_z.h"
#include "plasma_internal_zc.h"

#endif // PLASMA_INTERNAL_H
