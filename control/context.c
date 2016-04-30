/**
 *
 * @file context.c
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

#include <stdlib.h>

#include "plasma_context.h"
#include "plasma_internal.h"

static int max_contexts = 1024;
static int num_contexts = 0;

plasma_context_map_t *context_map = NULL;
pthread_mutex_t context_map_lock = PTHREAD_MUTEX_INITIALIZER;

/***************************************************************************//**
    @ingroup plasma_init
    Initializes PLASMA, allocating its context.
*/
int PLASMA_Init()
{
    pthread_mutex_lock(&context_map_lock);

    // Allocate context map if NULL.
    if (context_map == NULL) {
        context_map =
            (plasma_context_map_t*)calloc(max_contexts,
                                          sizeof(plasma_context_map_t));
        if (context_map == NULL) {
            pthread_mutex_unlock(&context_map_lock);
            plasma_error("calloc() failed");
            return PLASMA_ERR_OUT_OF_RESOURCES;
        }
    }
    pthread_mutex_unlock(&context_map_lock);

    plasma_context_attach();
    return PLASMA_SUCCESS;
}

/***************************************************************************//**
    @ingroup plasma_init
    Finalizes PLASMA, freeing its context.
*/
int PLASMA_Finalize()
{
    plasma_context_detach();
    return PLASMA_SUCCESS;
}

/******************************************************************************/
int PLASMA_Set(PLASMA_enum param, int value)
{
    plasma_context_t *plasma;

    plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }
    switch (param) {
    case PLASMA_TILE_SIZE:
        if (value <= 0) {
            plasma_error("Negative tile size");
            return PLASMA_ERR_ILLEGAL_VALUE;
        }
        plasma->nb = value;
        break;
    default:
        plasma_error("Unknown parameter");
        return PLASMA_ERR_ILLEGAL_VALUE;
    }
    return PLASMA_SUCCESS;
}

/******************************************************************************/
int PLASMA_Get(PLASMA_enum param, int *value)
{
    plasma_context_t *plasma;

    plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }
    switch (param) {
    case PLASMA_TILE_SIZE:
        *value = plasma->nb;
        return PLASMA_SUCCESS;
        break;
    default:
        plasma_error("Unknown parameter");
        return PLASMA_ERR_ILLEGAL_VALUE;
    }
    return PLASMA_SUCCESS;
}

/******************************************************************************/
int plasma_context_attach()
{
    pthread_mutex_lock(&context_map_lock);

    // Reallocate context map if out of space.
    if (num_contexts == max_contexts-1) {
        max_contexts *= 2;
        context_map = (plasma_context_map_t*) realloc(
            &context_map, max_contexts*sizeof(plasma_context_map_t));
        if (context_map == NULL) {
            pthread_mutex_unlock(&context_map_lock);
            plasma_error("realloc() failed");
            return PLASMA_ERR_OUT_OF_RESOURCES;
        }
    }
    // Create the context.
    plasma_context_t *context;
    context = (plasma_context_t*)malloc(sizeof(plasma_context_t));
    if (context == NULL) {
        pthread_mutex_unlock(&context_map_lock);
        plasma_error("malloc() failed");
        return PLASMA_ERR_OUT_OF_RESOURCES;
    }
    // Initialize the context.
    plasma_context_init(context);

    // Find and empty slot and insert the context.
    for (int i = 0; i < max_contexts; i++) {
        if (context_map[i].context == NULL) {
            context_map[i].context = context;
            context_map[i].thread_id = pthread_self();
            num_contexts++;
            pthread_mutex_unlock(&context_map_lock);
            return PLASMA_SUCCESS;
        }
    }
    // This should never happen.
    pthread_mutex_unlock(&context_map_lock);
    plasma_error("empty slot not found");
    return PLASMA_ERR_UNEXPECTED;
}

/******************************************************************************/
int plasma_context_detach()
{
    pthread_mutex_lock(&context_map_lock);

    // Find the thread and remove its context.
    for (int i = 0; i < max_contexts; i++) {
        if (context_map[i].context != NULL &&
            pthread_equal(context_map[i].thread_id, pthread_self())) {

            free(context_map[i].context);
            context_map[i].context = NULL;
            num_contexts--;
            pthread_mutex_unlock(&context_map_lock);
            return PLASMA_SUCCESS;
        }
    }
    pthread_mutex_unlock(&context_map_lock);
    plasma_error("context not found");
    return PLASMA_ERR_UNEXPECTED;
}

/******************************************************************************/
plasma_context_t *plasma_context_self()
{
    pthread_mutex_lock(&context_map_lock);

    // Find the thread and return its context.
    for (int i = 0; i < max_contexts; i++) {
        if (context_map[i].context != NULL &&
            pthread_equal(context_map[i].thread_id, pthread_self())) {

            pthread_mutex_unlock(&context_map_lock);
            return context_map[i].context;
        }
    }
    pthread_mutex_unlock(&context_map_lock);
    plasma_error("context not found");
    return NULL;
}

/******************************************************************************/
void plasma_context_init(plasma_context_t *context)
{
    context->nb = 256;
    context->translation = PLASMA_OUTOFPLACE;
}
