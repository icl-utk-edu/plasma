/**
 *
 * @file async.h
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
#ifndef ASYNC_H
#define ASYNC_H

#include "plasmatypes.h"

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
int plasma_request_fail(
    PLASMA_sequence *sequence, PLASMA_request *request, int status);

int plasma_sequence_create(PLASMA_sequence **sequence);
int plasma_sequence_destroy(PLASMA_sequence *sequence);

#endif // ASYNC_H
