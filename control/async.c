/**
 *
 * @file async.c
 *
 *  PLASMA control routines.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver.
 *
 * @version 3.0.0
 * @author Jakub Kurzak
 * @date 2016-01-01
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
    *sequence = (PLASMA_sequence*) malloc(sizeof(PLASMA_sequence));
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
