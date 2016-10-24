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
#ifndef ICL_CORE_BLAS_ZC_H
#define ICL_CORE_BLAS_ZC_H

#include "plasma_async.h"
#include "plasma_types.h"
#include "plasma_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************/
void core_zlag2c(int m, int n,
                 plasma_complex64_t *A,  int lda,
                 plasma_complex32_t *As, int ldas);

void core_clag2z(int m, int n,
                 plasma_complex32_t *As, int ldas,
                 plasma_complex64_t *A,  int lda);

/******************************************************************************/
void core_omp_zlag2c(int m, int n,
                     plasma_complex64_t *A,  int lda,
                     plasma_complex32_t *As, int ldas,
                     plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_clag2z(int m, int n,
                     plasma_complex32_t *As, int ldas,
                     plasma_complex64_t *A,  int lda,
                     plasma_sequence_t *sequence, plasma_request_t *request);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_CORE_BLAS_ZC_H
