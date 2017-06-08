/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/

#include "plasma_async.h"
#include "plasma_internal.h"

#include <stdlib.h>

/******************************************************************************/
int plasma_request_fail(plasma_sequence_t *sequence,
                        plasma_request_t *request,
                        int status)
{
    sequence->request = request;
    sequence->status = status;
    request->status = status;
    return status;
}

/******************************************************************************/
int plasma_request_init(plasma_request_t *request)
{
    request->status = PlasmaSuccess;
    return PlasmaSuccess;
}

/******************************************************************************/
int plasma_sequence_init(plasma_sequence_t *sequence)
{
    sequence->status = PlasmaSuccess;
    sequence->request = NULL;
    return PlasmaSuccess;
}
