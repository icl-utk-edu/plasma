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
int plasma_request_fail(PLASMA_sequence *sequence,
                        PLASMA_request *request,
                        int status)
{
    sequence->request = request;
    sequence->status = status;
    request->status = status;
    return status;
}

/******************************************************************************/
int plasma_sequence_create(PLASMA_sequence **sequence)
{
    *sequence = (PLASMA_sequence*)malloc(sizeof(PLASMA_sequence));
    if (*sequence == NULL) {
        plasma_error("malloc() failed");
        return PLASMA_ERR_OUT_OF_RESOURCES;
    }
    (*sequence)->status = PLASMA_SUCCESS;
    return PLASMA_SUCCESS;
}

/******************************************************************************/
int plasma_sequence_destroy(PLASMA_sequence *sequence)
{
    free(sequence);
    return PLASMA_SUCCESS;
}
