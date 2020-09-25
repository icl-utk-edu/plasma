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

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#if defined(PLASMA_USE_LUA)
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

/******************************************************************************/
void
plasma_tuning_init(plasma_context_t *plasma)
{
    // Initiaize Lua.
    lua_State *L = luaL_newstate();
    if (L == NULL) {
        plasma_error("luaL_newstate() failed");
        return;
    }
    luaL_openlibs(L);

    // Get the config file name.
    char *config_filename = getenv("PLASMA_TUNING_FILENAME");
    if (config_filename == NULL) {
        plasma_error("PLASMA_TUNING_FILENAME not set");
        lua_close(L);
        return;
    }

    // Check if config file exists.
    FILE *file;
    file = fopen(config_filename, "r");
    if (file == NULL) {
        plasma_error("config file not found");
        lua_close(L);
        return;
    }
    fclose(file);

    // Execute the config file.
    int retval = luaL_dofile(L, config_filename);
    if (retval != 0) {
        plasma_error("error executing tuning file");
        lua_close(L);
        return;
    }
}

/******************************************************************************/
void plasma_tuning_finalize(plasma_context_t *plasma)
{
    lua_State *L = (lua_State *)plasma->L;
    if (L != NULL)
        lua_close(L);
}

/******************************************************************************/
static void plasma_tune(plasma_context_t *plasma, plasma_enum_t dtyp,
                        const char *func_name, int *out, int count, ...)
{
    lua_State *L = (lua_State *)plasma->L;
    int retval;
    retval = lua_getglobal(L, func_name);
    if (retval != LUA_TFUNCTION) {
        plasma_error("lua_getglobal() failed");
        return;
    }
    switch (dtyp) {
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

#else
void
plasma_tuning_init(plasma_context_t *plasma)
{
    if (plasma) return;
}

void plasma_tuning_finalize(plasma_context_t *plasma)
{
    if (plasma) return;
}


static void plasma_tune(plasma_context_t *plasma, plasma_enum_t dtyp,
                        const char *func_name, int *out, int count, ...)
{
    va_list ap;
    va_start(ap, count);
    /* drain variable arguments to prevent stack leaks */
    for (int i = 0; i < count; i++)
        va_arg(ap, int);
    va_end(ap);

    /* When Lua is missing use tile size 100 with inner blocking 50 and one
     * panel thread. */
    if (strstr(func_name, "_nb"))
        *out = 100;
    else if (strstr(func_name, "_ib"))
        *out = 50;
    else if (strstr(func_name, "_threads"))
        *out = 1;
    else {
        plasma_error("plasma_tune() unknown routine");
        *out = 64;
    }
}
#endif

/******************************************************************************/
void plasma_tune_gbtrf(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int n, int bw)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "gbtrf_nb", &plasma->nb, 2, n, bw);
    plasma_tune(plasma, dtyp, "gbtrf_ib", &plasma->ib, 2, n, bw);
    plasma_tune(plasma, dtyp, "gbtrf_max_panel_threads",
                &plasma->max_panel_threads, 2, n, bw);
}

/******************************************************************************/
void plasma_tune_geadd(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "geadd_nb", &plasma->nb, 2, m, n);
}

/******************************************************************************/
void plasma_tune_geinv(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "geinv_nb", &plasma->nb, 2, m, n);
    plasma_tune(plasma, dtyp, "geinv_ib", &plasma->ib, 2, m, n);
    plasma_tune(plasma, dtyp, "geinv_max_panel_threads",
                &plasma->max_panel_threads, 2, m, n);
}

/******************************************************************************/
void plasma_tune_gelqf(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "gelqf_nb", &plasma->nb, 2, m, n);
    plasma_tune(plasma, dtyp, "gelqf_ib", &plasma->ib, 2, m, n);
}

/******************************************************************************/
void plasma_tune_gemm(plasma_context_t *plasma, plasma_enum_t dtyp,
                      int m, int n, int k)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "gemm_nb", &plasma->nb, 3, m, n, k);
}

/******************************************************************************/
void plasma_tune_geqrf(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "geqrf_nb", &plasma->nb, 2, m, n);
    plasma_tune(plasma, dtyp, "geqrf_ib", &plasma->ib, 2, m, n);
}

/******************************************************************************/
void plasma_tune_geswp(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "geswp_nb", &plasma->nb, 2, m, n);
}

/******************************************************************************/
void plasma_tune_getrf(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "getrf_nb", &plasma->nb, 2, m, n);
    plasma_tune(plasma, dtyp, "getrf_ib", &plasma->ib, 2, m, n);
    plasma_tune(plasma, dtyp, "getrf_max_panel_threads",
                &plasma->max_panel_threads, 2, m, n);
}

/******************************************************************************/
void plasma_tune_hetrf(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int n)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "hetrf_nb", &plasma->nb, 1, n);
}

/******************************************************************************/
void plasma_tune_lacpy(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "lacpy_nb", &plasma->nb, 2, m, n);
}

/******************************************************************************/
void plasma_tune_lag2c(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "lag2c_nb", &plasma->nb, 2, m, n);
}

/******************************************************************************/
void plasma_tune_lange(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "lange_nb", &plasma->nb, 2, m, n);
}

/******************************************************************************/
void plasma_tune_lansy(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int n)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "lansy_nb", &plasma->nb, 1, n);
}

/******************************************************************************/
void plasma_tune_lantr(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "lantr_nb", &plasma->nb, 2, m, n);
}

/******************************************************************************/
void plasma_tune_lascl(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "lascl_nb", &plasma->nb, 2, m, n);
}

/******************************************************************************/
void plasma_tune_laset(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "laset_nb", &plasma->nb, 2, m, n);
}

/******************************************************************************/
void plasma_tune_lauum(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int n)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "lauum_nb", &plasma->nb, 1, n);
}

/******************************************************************************/
void plasma_tune_pbtrf(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int n)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "pbtrf_nb", &plasma->nb, 1, n);
}

/******************************************************************************/
void plasma_tune_poinv(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int n)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "poinv_nb", &plasma->nb, 1, n);
}

/******************************************************************************/
void plasma_tune_potrf(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int n)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "potrf_nb", &plasma->nb, 1, n);
}

/******************************************************************************/
void plasma_tune_symm(plasma_context_t *plasma, plasma_enum_t dtyp,
                      int m, int n)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "symm_nb", &plasma->nb, 2, m, n);
}

/******************************************************************************/
void plasma_tune_syr2k(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int n, int k)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "syr2k_nb", &plasma->nb, 2, n, k);
}

/******************************************************************************/
void plasma_tune_syrk(plasma_context_t *plasma, plasma_enum_t dtyp,
                      int n, int k)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "syrk_nb", &plasma->nb, 2, n, k);
}

/******************************************************************************/
void plasma_tune_tradd(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "tradd_nb", &plasma->nb, 2, m, n);
}

/******************************************************************************/
void plasma_tune_trmm(plasma_context_t *plasma, plasma_enum_t dtyp,
                      int m, int n)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "trmm_nb", &plasma->nb, 2, m, n);
}

/******************************************************************************/
void plasma_tune_trsm(plasma_context_t *plasma, plasma_enum_t dtyp,
                      int m, int n)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "trsm_nb", &plasma->nb, 2, m, n);
}

/******************************************************************************/
void plasma_tune_trtri(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int n)
{
    if (NULL == plasma->L)
        return;

    plasma_tune(plasma, dtyp, "trtri_nb", &plasma->nb, 1, n);
}
