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

void plasma_tune_gbtrf(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int n, int bw);
void plasma_tune_geadd(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n);
void plasma_tune_geinv(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n);
void plasma_tune_gelqf(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n);
void plasma_tune_gemm(plasma_context_t *plasma, plasma_enum_t dtyp,
                      int m, int n, int k);
void plasma_tune_geqrf(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n);
void plasma_tune_geswp(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n);
void plasma_tune_getrf(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n);
void plasma_tune_hetrf(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int n);
void plasma_tune_lacpy(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n);
void plasma_tune_lag2c(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n);
void plasma_tune_lange(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n);
void plasma_tune_lansy(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int n);
void plasma_tune_lantr(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n);
void plasma_tune_lascl(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n);
void plasma_tune_laset(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n);
void plasma_tune_lauum(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int n);
void plasma_tune_pbtrf(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int n);
void plasma_tune_poinv(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int n);
void plasma_tune_potrf(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int n);
void plasma_tune_symm(plasma_context_t *plasma, plasma_enum_t dtyp,
                      int m, int n);
void plasma_tune_syr2k(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int n, int k);
void plasma_tune_syrk(plasma_context_t *plasma, plasma_enum_t dtyp,
                      int n, int k);
void plasma_tune_tradd(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int m, int n);
void plasma_tune_trmm(plasma_context_t *plasma, plasma_enum_t dtyp,
                      int m, int n);
void plasma_tune_trsm(plasma_context_t *plasma, plasma_enum_t dtyp,
                      int m, int n);
void plasma_tune_trtri(plasma_context_t *plasma, plasma_enum_t dtyp,
                       int n);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_PLASMA_TUNING_H
