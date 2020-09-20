/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/

#include "plasma_context.h"
#include "plasma_internal.h"
#include "plasma_tuning.h"

#include <stdlib.h>
#include <omp.h>

#if defined(PLASMA_USE_MAGMA)
#include <magma.h>
#endif

static int plasma_initialized_g = 0;
static plasma_context_t plasma_context_g;

/***************************************************************************//**
    @ingroup plasma_init
    Initializes PLASMA, allocating its context.
    This function must be called outside of any parallel region.
*/
int plasma_init()
{
    if (plasma_initialized_g)
        return PlasmaErrorNotInitialized;

    if (omp_in_parallel())
        return PlasmaErrorNotInitialized;

    plasma_initialized_g = 1;

    plasma_context_init(&plasma_context_g);

#if defined(PLASMA_USE_MAGMA)
    magma_init();
#endif

    return PlasmaSuccess;
}

/***************************************************************************//**
    @ingroup plasma_init
    Finalizes PLASMA, freeing its context.
    This function must be called outside of any parallel region.
*/
int plasma_finalize()
{
    if (! plasma_initialized_g)
        return PlasmaErrorNotInitialized;

    if (omp_in_parallel())
        return PlasmaErrorEnvironment;

#if defined(PLASMA_USE_MAGMA)
    magma_finalize();
#endif

    plasma_context_finalize(&plasma_context_g);

    plasma_initialized_g = 0;

    return PlasmaSuccess;
}

/***************************************************************************//**
    @ingroup plasma_init
    Sets one of PLASMA's internal state variables.
    This function must be called outside of any parallel region.
*/
int plasma_set(plasma_enum_t param, int value)
{
    if (! plasma_initialized_g)
        return PlasmaErrorNotInitialized;

    if (omp_in_parallel())
        return PlasmaErrorEnvironment;

    switch (param) {
    case PlasmaTuning:
        if (value != PlasmaEnabled && value != PlasmaDisabled) {
            plasma_error("invalid tuning flag");
            return PlasmaErrorIllegalValue;
        }
        plasma_context_g.tuning = value;
        break;
    case PlasmaNb:
        if (value <= 0) {
            plasma_error("invalid tile size");
            return PlasmaErrorIllegalValue;
        }
        plasma_context_g.nb = value;
        break;
    case PlasmaIb:
        if (value <= 0) {
            plasma_error("invalid inner block size");
            return PlasmaErrorIllegalValue;
        }
        plasma_context_g.ib = value;
        break;
    case PlasmaNumPanelThreads:
        if (value <= 0) {
            plasma_error("invalid number of panel threads");
            return PlasmaErrorIllegalValue;
        }
        plasma_context_g.max_panel_threads = value;
        break;
    case PlasmaHouseholderMode:
        if (value != PlasmaFlatHouseholder && value != PlasmaTreeHouseholder) {
            plasma_error("invalid Householder mode");
            return PlasmaErrorIllegalValue;
        }
        plasma_context_g.householder_mode = value;
        break;
    default:
        plasma_error("unknown parameter");
        return PlasmaErrorIllegalValue;
    }
    return PlasmaSuccess;
}

/***************************************************************************//**
    @ingroup plasma_init
    Gets one of PLASMA's internal state variables.
*/
int plasma_get(plasma_enum_t param, int *value)
{
    if (! plasma_initialized_g)
        return PlasmaErrorNotInitialized;

    switch (param) {
    case PlasmaTuning:
        *value = plasma_context_g.tuning;
        return PlasmaSuccess;
    case PlasmaNb:
        *value = plasma_context_g.nb;
        return PlasmaSuccess;
    case PlasmaIb:
        *value = plasma_context_g.ib;
        return PlasmaSuccess;
    case PlasmaNumPanelThreads:
        *value = plasma_context_g.max_panel_threads;
        return PlasmaSuccess;
    case PlasmaHouseholderMode:
        *value = plasma_context_g.householder_mode;
        return PlasmaSuccess;
    default:
        plasma_error("Unknown parameter");
        return PlasmaErrorIllegalValue;
    }
    return PlasmaSuccess;
}

/***************************************************************************//**
    @ingroup plasma_init
    Initializes PLASMA's execution context to default values.
*/
void plasma_context_init(plasma_context_t *context)
{
    if (! context)
        return;

    // Set defaults.
    context->tuning = PlasmaEnabled;
    context->nb = 256;
    context->ib = 64;
    context->inplace_outplace = PlasmaOutplace;
    context->max_threads = omp_get_max_threads();
    context->max_panel_threads = 1;
    context->householder_mode = PlasmaFlatHouseholder;

    plasma_tuning_init(context);
}

/***************************************************************************//**
    @ingroup plasma_init
    Resets PLASMA's execution context.
*/
void plasma_context_finalize(plasma_context_t *context)
{
    plasma_tuning_finalize(context);
}

/***************************************************************************//**
    @ingroup plasma_init
    Returns PLASMA's default execution context or NULL if PLASMA was not
    initialized.
*/
plasma_context_t *plasma_context_self()
{
    if (plasma_initialized_g)
        return &plasma_context_g;
    else
        return NULL;
}
