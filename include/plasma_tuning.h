/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#ifndef ICL_PLASMA_TUNING_H
#define ICL_PLASMA_TUNING_H

#include "plasma_context.h"

#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************/
lua_State *plasma_tuning_init();
void plasma_tuning_finalize(lua_State *L);
void plasma_tune_getrf(plasma_context_t *plasma, plasma_enum_t dtyp,
					   int m, int n);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_PLASMA_TUNING_H
