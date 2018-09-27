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
#ifndef PLASMA_ZC_H
#define PLASMA_ZC_H

#include "plasma_async.h"
#include "plasma_descriptor.h"
#include "plasma_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

/***************************************************************************//**
 *  Standard interface
 **/
int plasma_zcgesv(int n, int nrhs,
                  plasma_complex64_t *pA, int lda, int *ipiv,
                  plasma_complex64_t *pB, int ldb,
                  plasma_complex64_t *pX, int ldx, int *iter);

int plasma_zcposv(plasma_enum_t uplo, int n, int nrhs,
                  plasma_complex64_t *pA, int lda,
                  plasma_complex64_t *pB, int ldb,
                  plasma_complex64_t *pX, int ldx, int *iter);

int plasma_zcgbsv(int n, int kl, int ku, int nrhs,
                  plasma_complex64_t *pAB, int ldab, int *ipiv,
                  plasma_complex64_t *pB, int ldb,
                  plasma_complex64_t *pX, int ldx, int *iter);

int plasma_zlag2c(int m, int n,
                  plasma_complex64_t *pA,  int lda,
                  plasma_complex32_t *pAs, int ldas);

int plasma_clag2z(int m, int n,
                  plasma_complex32_t *pAs, int ldas,
                  plasma_complex64_t *pA,  int lda);

/***************************************************************************//**
 *  Tile asynchronous interface
 **/
void plasma_omp_zcgesv(plasma_desc_t A,  int *ipiv,
                       plasma_desc_t B,  plasma_desc_t X,
                       plasma_desc_t As, plasma_desc_t Xs, plasma_desc_t R,
                       double *work, double *Rnorm, double *Xnorm, int *iter,
                       plasma_sequence_t *sequence,
                       plasma_request_t  *request);

void plasma_omp_zcposv(plasma_enum_t uplo,
                       plasma_desc_t A,  plasma_desc_t B,  plasma_desc_t X,
                       plasma_desc_t As, plasma_desc_t Xs, plasma_desc_t R,
                       double *W,  double *Rnorm, double *Xnorm, int *iter,
                       plasma_sequence_t *sequence,
                       plasma_request_t  *request);

void plasma_omp_zcgbsv(plasma_desc_t A,  int *ipiv,
                       plasma_desc_t B,  plasma_desc_t X,
                       plasma_desc_t As, plasma_desc_t Xs, plasma_desc_t R,
                       double *work, double *Rnorm, double *Xnorm, int *iter,
                       plasma_sequence_t *sequence,
                       plasma_request_t  *request);

void plasma_omp_zlag2c(plasma_desc_t A, plasma_desc_t As,
                       plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_omp_clag2z(plasma_desc_t As, plasma_desc_t A,
                       plasma_sequence_t *sequence, plasma_request_t *request);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // PLASMA_ZC_H
