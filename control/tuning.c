/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/

#include "plasma_tuning.h"
#include "plasma_error.h"
#include "plasma_types.h"
#include "plasma_context.h"
#include "plasma_types.h"

#include <stdio.h>
#include <assert.h>
#include <stdarg.h>
#include <omp.h>

/******************************************************************************/
lua_State *plasma_tuning_init()
{
    // Initiaize Lua.
    lua_State *L = luaL_newstate();
    if (L == NULL) {
        plasma_error("luaL_newstate() failed");
        return NULL;
    }
    luaL_openlibs(L);

    // Get the config file name.
    char *config_filename = getenv("PLASMA_TUNING_FILENAME");
    if (config_filename == NULL) {
        plasma_error("PLASMA_TUNING_FILENAME not set");
        lua_close(L);
        return NULL;
    }

    // Check if config file exists.
    FILE *file;
    file = fopen(config_filename, "r");
    if (file == NULL) {
        plasma_error("config file not found");
        lua_close(L);
        return NULL;
    }
    fclose(file);

    // Execute the config file.
    int retval = luaL_dofile(L, config_filename);
    if (retval != 0) {
        plasma_error("error executing tuning file");
        lua_close(L);
        return NULL;
    }

    return L;
}

/******************************************************************************/
void plasma_tuning_finalize(lua_State *L)
{
    if (L != NULL)
        lua_close(L);
}

/******************************************************************************/
static void plasma_tune(lua_State *L, plasma_enum_t dtyp,
                       char *func_name, int *out, int count, ...)
{
    int retval;
    retval = lua_getglobal(L, func_name);
    if (retval != LUA_TFUNCTION) {
        plasma_error("lua_getglobal() failed");
        return;
    }
    switch (dtyp){
        case PlasmaComplexDouble: lua_pushstring(L, "Z"); break;
        case PlasmaComplexFloat:  lua_pushstring(L, "C"); break;
        case PlasmaRealDouble:    lua_pushstring(L, "D"); break;
        case PlasmaRealFloat:     lua_pushstring(L, "S"); break;
        default: plasma_error("invalid type"); return;
    }

    lua_pushinteger(L, omp_get_max_threads());

    va_list ap;
    va_start(ap, count);
    for (int i = 0; i < count; i++)
        lua_pushinteger(L, va_arg(ap, int));
    va_end(ap);

    retval = lua_pcall(L, 2+count, 1, 0);
    if (retval != LUA_OK) {
        plasma_error("lua_pcall() failed");
        return;
    }
    retval = lua_tonumber(L, -1);
    if (retval == 0) {
        plasma_error("lua_tonumber() failed");
        return;
    }
    *out = retval;
    lua_pop(L, 1);
}

/******************************************************************************/
void plasma_tune_getrf(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n)
{
    if (plasma->L == NULL)
        return;

    plasma_tune(plasma->L, dtyp, "getrf_nb", &plasma->nb, 2, m, n);
    plasma_tune(plasma->L, dtyp, "getrf_ib", &plasma->ib, 2, m, n);
    plasma_tune(plasma->L, dtyp, "getrf_max_panel_threads",
                &plasma->max_panel_threads, 2, m, n);
}
