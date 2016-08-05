/**
 *
 * @file plasma_context.h
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#ifndef ICL_PLASMA_CONTEXT_H
#define ICL_PLASMA_CONTEXT_H

#include "plasma_types.h"

#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************/
typedef struct {
    int nb;                  ///< PLASMA_TILE_SIZE
    PLASMA_enum translation; ///< in-place or out-of-place PLASMA_TRANSLATION_MODE
} plasma_context_t;

typedef struct {
    pthread_t thread_id;       ///< thread id
    plasma_context_t *context; ///< pointer to associated context
} plasma_context_map_t;

/******************************************************************************/
int PLASMA_Init();
int PLASMA_Finalize();
int PLASMA_Set(PLASMA_enum param, int value);
int PLASMA_Get(PLASMA_enum param, int *value);

int plasma_context_attach();
int plasma_context_detach();
plasma_context_t *plasma_context_self();
void plasma_context_init(plasma_context_t *context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_PLASMA_CONTEXT_H
