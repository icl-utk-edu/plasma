/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#ifndef ICL_PLASMA_CONFIG_H
#define ICL_PLASMA_CONFIG_H

#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************/
void plasma_config_init(lua_State *L);
void plasma_config_finalize(lua_State *L);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_PLASMA_CONFIG_H
