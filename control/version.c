/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/

#include "plasma.h"

/******************************************************************************/
void plasma_version(int *major, int *minor, int *patch)
{
    if (major) *major = PLASMA_VERSION_MAJOR;
    if (minor) *minor = PLASMA_VERSION_MINOR;
    if (patch) *patch = PLASMA_VERSION_PATCH;
}
