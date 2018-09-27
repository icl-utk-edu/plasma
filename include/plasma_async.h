/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#ifndef PLASMA_ASYNC_H
#define PLASMA_ASYNC_H

#include "plasma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************/
typedef struct {
    plasma_enum_t status; ///< error code
} plasma_request_t;

typedef struct {
    plasma_enum_t status;      ///< error code
    plasma_request_t *request; ///< failed request
} plasma_sequence_t;

/******************************************************************************/
int plasma_request_fail(plasma_sequence_t *sequence,
                        plasma_request_t *request,
                        int status);

int plasma_request_init(plasma_request_t *request);

int plasma_sequence_init(plasma_sequence_t *sequence);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // PLASMA_ASYNC_H
