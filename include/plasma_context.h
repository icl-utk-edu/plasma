/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#ifndef PLASMA_CONTEXT_H
#define PLASMA_CONTEXT_H

#include "plasma_types.h"
#include "plasma_barrier.h"

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************/
typedef struct {
    int tuning;                     ///< PlasmaEnabled or PlasmaDisabled
    int nb;                         ///< PlasmaNb
    int ib;                         ///< PlasmaIb
    plasma_enum_t inplace_outplace; ///< PlasmaInplaceOutplace
    int max_threads;                ///< the value of OMP_NUM_THREADS
    int max_panel_threads;          ///< max threads for panel factorization
    plasma_barrier_t barrier;       ///< thread barrier for multithreaded tasks
    plasma_enum_t householder_mode; ///< PlasmaHouseholderMode
    void *L;                        ///< Lua state pointer; unusued when Lua is missing
} plasma_context_t;

/******************************************************************************/
int plasma_init();
int plasma_finalize();
int plasma_set(plasma_enum_t param, int value);
int plasma_get(plasma_enum_t param, int *value);

plasma_context_t *plasma_context_self();
void plasma_context_init(plasma_context_t *context);
void plasma_context_finalize(plasma_context_t *context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // PLASMA_CONTEXT_H
