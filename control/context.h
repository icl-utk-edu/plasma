/**
 *
 * @file context.h
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
#ifndef CONTEXT_H
#define CONTEXT_H

#include "../include/plasmatypes.h"

#include <pthread.h>

/******************************************************************************/
typedef struct {
    int nb;                  ///< tile size
    PLASMA_enum translation; ///< in-place or out-of-place layout translation
} plasma_context_t;

typedef struct {
    pthread_t thread_id;       ///< thread id
    plasma_context_t *context; ///< pointer to associated context
} plasma_context_map_t;

/******************************************************************************/
int PLASMA_Init();
int PLASMA_Finalize();

int plasma_context_attach();
int plasma_context_detach();
plasma_context_t *plasma_context_self();

#endif // CONTEXT_H
