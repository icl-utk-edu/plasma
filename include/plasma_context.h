/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#ifndef ICL_PLASMA_CONTEXT_H
#define ICL_PLASMA_CONTEXT_H

#include "plasma_types.h"
#include "plasma_barrier.h"

#include <pthread.h>
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************/
typedef struct {
    lua_State *L;                   ///< Lua state
    int tuning;                     ///< PlasmaEnabled or PlasmaDisabled
    int nb;                         ///< PlasmaNb
    int ib;                         ///< PlasmaIb
    plasma_enum_t inplace_outplace; ///< PlasmaInplaceOutplace
    int max_threads;                ///< the value of OMP_NUM_THREADS
    int max_panel_threads;          ///< max threads for panel factorization
    plasma_barrier_t barrier;       ///< thread barrier for multithreaded tasks
    plasma_enum_t householder_mode; ///< PlasmaHouseholderMode
} plasma_context_t;

typedef struct {
    pthread_t thread_id;       ///< thread id
    plasma_context_t *context; ///< pointer to associated context
} plasma_context_map_t;

/******************************************************************************/
int plasma_init();
int plasma_finalize();
int plasma_set(plasma_enum_t param, int value);
int plasma_get(plasma_enum_t param, int *value);

int plasma_context_attach();
int plasma_context_detach();
plasma_context_t *plasma_context_self();
void plasma_context_init(plasma_context_t *context);
void plasma_context_finalize(plasma_context_t *context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_PLASMA_CONTEXT_H
