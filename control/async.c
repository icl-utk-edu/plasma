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
                        PLASMA_request *request,
                        int status)
{
    sequence->request = request;
    sequence->status = status;
    request->status = status;
    return status;
}

/******************************************************************************/
int plasma_sequence_create(plasma_sequence_t **sequence)
{
    *sequence = (plasma_sequence_t*)malloc(sizeof(plasma_sequence_t));
    if (*sequence == NULL) {
        plasma_error("malloc() failed");
        return PLASMA_ERR_OUT_OF_RESOURCES;
    }
    (*sequence)->status = PLASMA_SUCCESS;
    return PLASMA_SUCCESS;
}

/******************************************************************************/
int plasma_sequence_destroy(plasma_sequence_t *sequence)
{
    free(sequence);
    return PLASMA_SUCCESS;
}
