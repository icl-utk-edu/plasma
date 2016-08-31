/**
 *
 * @file plasma_async.h
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#ifndef ICL_PLASMA_ASYNC_H
#define ICL_PLASMA_ASYNC_H

#include "plasma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************/
typedef struct {
    PLASMA_bool status; ///< error code
} PLASMA_request;

static const PLASMA_request PLASMA_REQUEST_INITIALIZER = {PLASMA_SUCCESS};

typedef struct {
    PLASMA_bool status;      ///< error code
    PLASMA_request *request; ///< failed request
} PLASMA_sequence;

/******************************************************************************/
int plasma_request_fail(PLASMA_sequence *sequence,
                        PLASMA_request *request,
                        int status);

int plasma_sequence_create(PLASMA_sequence **sequence);
int plasma_sequence_destroy(PLASMA_sequence *sequence);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_PLASMA_ASYNC_H
