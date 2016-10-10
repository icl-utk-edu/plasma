/**
 *
 * @file
 *
 *  PLASMA header.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of Manchester, Univ. of California Berkeley and
 *  Univ. of Colorado Denver.
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
 *  Standard interface
 **/
int PLASMA_zlag2c(int m, int n,
                  plasma_complex64_t *A,  int lda,
                  plasma_complex32_t *As, int ldas);

int PLASMA_clag2z(int m, int n,
                  plasma_complex32_t *As, int ldas,
                  plasma_complex64_t *A,  int lda);

/***************************************************************************//**
 *  Tile asynchronous interface
 **/

void plasma_omp_zlag2c(plasma_desc_t *A, plasma_desc_t *As,
                       plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_omp_clag2z(plasma_desc_t *As, plasma_desc_t *A,
                       plasma_sequence_t *sequence, plasma_request_t *request);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_PLASMA_ZC_H
