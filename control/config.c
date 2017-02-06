/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/

#include "plasma_config.h"
#include "plasma_error.h"

#include <stdio.h>

/******************************************************************************/
void plasma_config_init(lua_State *L) {

    // Initiaize Lua.
    L = luaL_newstate();
    if (L == NULL) 
        plasma_fatal_error("luaL_newstate() failed");

    luaL_openlibs(L);

    // Get the config file name.
    char *config_filename = getenv("PLASMA_CONFIG_FILENAME");
    if (config_filename == NULL)
        plasma_error("PLASMA_CONFIG_FILENAME not set");

    // Check if config file exists.
    FILE *file;
    file = fopen(config_filename, "r");
    if (file == NULL)
        plasma_error("config file not found");
    else
        fclose(file);

    // Execute the config file.
    int retval = luaL_dofile (L, config_filename);
    if (retval != 0)
        plasma_error("error executing config file");
}

/******************************************************************************/
void plasma_config_finalize(lua_State *L)
{
    // Close Lua.
    // lua_close(L);
}
