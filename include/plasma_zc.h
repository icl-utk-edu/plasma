/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions mixed zc -> ds
 *
 **/

#ifndef ICL_PLASMA_ZC_H
#define ICL_PLASMA_ZC_H

#include "plasma_async.h"
#include "plasma_descriptor.h"
#include "plasma_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

/***************************************************************************//**
 *  Standard interface.
 **/

int PLASMA_zcposv(PLASMA_enum uplo, int n, int nrhs,
                  PLASMA_Complex64_t *A, int lda,
                  PLASMA_Complex64_t *B, int ldb,
                  PLASMA_Complex64_t *X, int ldx, int *iter);

/***************************************************************************//**
 *  Tile asynchronous interface.
 **/

void PLASMA_zcposv_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B,
                              PLASMA_desc *X, int *iter,
                              PLASMA_sequence *sequence,
                              PLASMA_request  *request);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_PLASMA_ZC_H
