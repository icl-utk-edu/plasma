/**
 *
 * @file
 *
 *  PLASMA header.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of Manchester, Univ. of California Berkeley and
 *  Univ. of Colorado Denver.
 *
 * @precisions normal z -> s d c
 *
 **/
#ifndef ICL_PLASMA_Z_H
#define ICL_PLASMA_Z_H

#include "plasma_async.h"
#include "plasma_descriptor.h"
#include "plasma_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

/***************************************************************************//**
 *  Standard interface.
 **/
int PLASMA_zgelqf(int m, int n,
                  plasma_complex64_t *A, int lda,
                  plasma_desc_t *descT);

int PLASMA_zgelqs(int m, int n, int nrhs,
                  plasma_complex64_t *A, int lda,
                  plasma_desc_t *descT,
                  plasma_complex64_t *B, int ldb);

int PLASMA_zgels(plasma_enum_t trans, int m, int n, int nrhs,
                 plasma_complex64_t *A, int lda,
                 plasma_desc_t *descT,
                 plasma_complex64_t *B, int ldb);

int PLASMA_zgemm(plasma_enum_t transA, plasma_enum_t transB,
                 int m, int n, int k,
                 plasma_complex64_t alpha, plasma_complex64_t *A, int lda,
                                           plasma_complex64_t *B, int ldb,
                 plasma_complex64_t beta,  plasma_complex64_t *C, int ldc);

int PLASMA_zgeqrf(int m, int n,
                  plasma_complex64_t *A, int lda,
                  plasma_desc_t *descT);

int PLASMA_zgeqrs(int m, int n, int nrhs,
                  plasma_complex64_t *A, int lda,
                  plasma_desc_t *descT,
                  plasma_complex64_t *B, int ldb);

int PLASMA_zhemm(plasma_enum_t side, plasma_enum_t uplo, int m, int n,
                 plasma_complex64_t alpha, plasma_complex64_t *A, int lda,
                                           plasma_complex64_t *B, int ldb,
                 plasma_complex64_t beta,  plasma_complex64_t *C, int ldc);

int PLASMA_zher2k(plasma_enum_t uplo, plasma_enum_t trans,
                  int n, int k,
                  plasma_complex64_t alpha, plasma_complex64_t *A, int lda,
                                            plasma_complex64_t *B, int ldb,
                               double beta, plasma_complex64_t *C, int ldc);

int PLASMA_zherk(plasma_enum_t uplo, plasma_enum_t trans,
                 int n, int k,
                 double alpha, plasma_complex64_t *A, int lda,
                 double beta,  plasma_complex64_t *C, int ldc);

int PLASMA_zlacpy(plasma_enum_t uplo, int m, int n,
                  plasma_complex64_t *A, int lda,
                  plasma_complex64_t *B, int ldb);

int PLASMA_zpbsv(plasma_enum_t uplo, int n, int kd, int nrhs,
                 plasma_complex64_t *AB, int ldab,
                 plasma_complex64_t *B, int ldb);

int PLASMA_zpbtrs(plasma_enum_t uplo, int n, int kd, int nrhs,
                  plasma_complex64_t *AB, int ldab,
                  plasma_complex64_t *B, int ldb);

int PLASMA_zpbtrf(plasma_enum_t uplo,
                  int n, int kd,
                  plasma_complex64_t *AB, int ldab);

int PLASMA_zposv(plasma_enum_t uplo, int n, int nrhs,
                 plasma_complex64_t *A, int lda,
                 plasma_complex64_t *B, int ldb);

int PLASMA_zpotrf(plasma_enum_t uplo,
                  int n,
                  plasma_complex64_t *A, int lda);

int PLASMA_zpotrs(plasma_enum_t uplo,
                  int n, int nrhs,
                  plasma_complex64_t *A, int lda,
                  plasma_complex64_t *B, int ldb);

int PLASMA_zsymm(plasma_enum_t side, plasma_enum_t uplo, int m, int n,
                 plasma_complex64_t alpha, plasma_complex64_t *A, int lda,
                                           plasma_complex64_t *B, int ldb,
                 plasma_complex64_t beta,  plasma_complex64_t *C, int ldc);

int PLASMA_zsyr2k(plasma_enum_t uplo, plasma_enum_t trans,
                  int n, int k,
                  plasma_complex64_t alpha, plasma_complex64_t *A, int lda,
                                            plasma_complex64_t *B, int ldb,
                  plasma_complex64_t beta,  plasma_complex64_t *C, int ldc);

int PLASMA_zsyrk(plasma_enum_t uplo, plasma_enum_t trans, int n, int k,
                 plasma_complex64_t alpha, plasma_complex64_t *A, int lda,
                 plasma_complex64_t beta,  plasma_complex64_t *C, int ldc);

int PLASMA_ztradd(plasma_enum_t uplo, plasma_enum_t transA, int m, int n,
                  plasma_complex64_t  alpha,
                  plasma_complex64_t *A, int lda,
                  plasma_complex64_t  beta,
                  plasma_complex64_t *B, int ldb);

int PLASMA_ztrmm(plasma_enum_t side, plasma_enum_t uplo,
                 plasma_enum_t transA, plasma_enum_t diag,
                 int m, int n, plasma_complex64_t alpha,
                 plasma_complex64_t *A, int lda,
                 plasma_complex64_t *B, int ldb);

int PLASMA_ztrsm(plasma_enum_t side, plasma_enum_t uplo,
                 plasma_enum_t transA, plasma_enum_t diag,
                 int m, int n,
                 plasma_complex64_t alpha, plasma_complex64_t *A, int lda,
                                           plasma_complex64_t *B, int ldb);

int PLASMA_zunglq(int m, int n, int k,
                  plasma_complex64_t *A, int lda,
                  plasma_desc_t *descT,
                  plasma_complex64_t *Q, int ldq);

int PLASMA_zungqr(int m, int n, int k,
                  plasma_complex64_t *A, int lda,
                  plasma_desc_t *descT,
                  plasma_complex64_t *Q, int ldq);

int PLASMA_zunmlq(plasma_enum_t side, plasma_enum_t trans, int m, int n, int k,
                  plasma_complex64_t *A, int lda,
                  plasma_desc_t *descT,
                  plasma_complex64_t *C, int ldc);

int PLASMA_zunmqr(plasma_enum_t side, plasma_enum_t trans, int m, int n, int k,
                  plasma_complex64_t *A, int lda,
                  plasma_desc_t *descT,
                  plasma_complex64_t *C, int ldc);

/***************************************************************************//**
 *  Tile asynchronous interface.
 **/

void PLASMA_zccrb2cm_Async(plasma_desc_t *A, plasma_complex64_t *Af77, int lda,
                           plasma_sequence_t *sequence,
                           plasma_request_t *request);

void PLASMA_zccrb2cm_band_Async(plasma_enum_t uplo,
                                plasma_desc_t *A,
                                plasma_complex64_t *Af77, int lda,
                                plasma_sequence_t *sequence,
                                plasma_request_t *request);

void PLASMA_zcm2ccrb_Async(plasma_complex64_t *Af77, int lda,
                           plasma_desc_t *A,
                           plasma_sequence_t *sequence,
                           plasma_request_t *request);

void PLASMA_zcm2ccrb_band_Async(plasma_enum_t uplo,
                                plasma_complex64_t *Af77, int lda,
                                plasma_desc_t *A,
                                plasma_sequence_t *sequence,
                                plasma_request_t *request);

void plasma_omp_zgelqf(plasma_desc_t *descA, plasma_desc_t *descT,
                       plasma_workspace_t *work,
                       plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_omp_zgelqs(plasma_desc_t *descA, plasma_desc_t *descT,
                       plasma_desc_t *descB, plasma_workspace_t *work,
                       plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_omp_zgels(plasma_enum_t trans,
                      plasma_desc_t *descA, plasma_desc_t *descT,
                      plasma_desc_t *descB, plasma_workspace_t *work,
                      plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_omp_zgemm(plasma_enum_t transA, plasma_enum_t transB,
                      plasma_complex64_t alpha, plasma_desc_t *A,
                                                plasma_desc_t *B,
                      plasma_complex64_t beta,  plasma_desc_t *C,
                      plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_omp_zgeqrf(plasma_desc_t *descA, plasma_desc_t *descT,
                       plasma_workspace_t *work,
                       plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_omp_zgeqrs(plasma_desc_t *descA, plasma_desc_t *descT,
                       plasma_desc_t *descB, plasma_workspace_t *work,
                       plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_omp_zhemm(plasma_enum_t side, plasma_enum_t uplo,
                      plasma_complex64_t alpha, plasma_desc_t *A,
                                                plasma_desc_t *B,
                      plasma_complex64_t beta,  plasma_desc_t *C,
                      plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_omp_zher2k(plasma_enum_t uplo, plasma_enum_t trans,
                       plasma_complex64_t alpha, plasma_desc_t *A,
                                                 plasma_desc_t *B,
                       double beta,              plasma_desc_t *C,
                       plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_omp_zherk(plasma_enum_t uplo, plasma_enum_t trans,
                      double alpha, plasma_desc_t *A,
                      double beta,  plasma_desc_t *C,
                      plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_omp_zlacpy(plasma_enum_t uplo, plasma_desc_t *A, plasma_desc_t *B,
                       plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_omp_zpbsv(plasma_enum_t uplo,
                      plasma_desc_t *AB,
                      plasma_desc_t *B,
                      plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_omp_zpbtrf(plasma_enum_t uplo, plasma_desc_t *AB,
                      plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_omp_zpbtrs(plasma_enum_t uplo, plasma_desc_t *AB, plasma_desc_t *B,
                       plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_omp_zposv(plasma_enum_t uplo, plasma_desc_t *A, plasma_desc_t *B,
                      plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_omp_zpotrf(plasma_enum_t uplo, plasma_desc_t *A,
                       plasma_sequence_t *sequence,
                       plasma_request_t *request);

void plasma_omp_zpotrs(plasma_enum_t uplo, plasma_desc_t *A, plasma_desc_t *B,
                        plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_omp_zsymm(plasma_enum_t side, plasma_enum_t uplo,
                      plasma_complex64_t alpha, plasma_desc_t *A,
                                                plasma_desc_t *B,
                      plasma_complex64_t beta,  plasma_desc_t *C,
                      plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_omp_zsyr2k(plasma_enum_t uplo, plasma_enum_t trans,
                       plasma_complex64_t alpha, plasma_desc_t *A,
                                                  plasma_desc_t *B,
                       plasma_complex64_t beta,  plasma_desc_t *C,
                       plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_omp_zsyrk(plasma_enum_t uplo, plasma_enum_t trans,
                      plasma_complex64_t alpha, plasma_desc_t *A,
                      plasma_complex64_t beta,  plasma_desc_t *C,
                      plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_omp_ztradd(plasma_enum_t uplo, plasma_enum_t transA,
                       plasma_complex64_t alpha, plasma_desc_t *A,
                       plasma_complex64_t beta,  plasma_desc_t *B,
                       plasma_sequence_t *sequence, plasma_request_t  *request);

void plasma_omp_ztrmm(plasma_enum_t side, plasma_enum_t uplo,
                      plasma_enum_t transA, plasma_enum_t diag,
                      plasma_complex64_t alpha, plasma_desc_t *A,
                                                plasma_desc_t *B,
                      plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_omp_ztrsm(plasma_enum_t side, plasma_enum_t uplo,
                      plasma_enum_t transA, plasma_enum_t diag,
                      plasma_complex64_t alpha, plasma_desc_t *A,
                                                plasma_desc_t *B,
                      plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_omp_zunglq(plasma_desc_t *descA, plasma_desc_t *descT,
                       plasma_desc_t *descQ, plasma_workspace_t *work,
                       plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_omp_zungqr(plasma_desc_t *descA, plasma_desc_t *descT,
                       plasma_desc_t *descQ, plasma_workspace_t *work,
                       plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_omp_zunmlq(plasma_enum_t side, plasma_enum_t trans,
                       plasma_desc_t *descA, plasma_desc_t *descT,
                       plasma_desc_t *descC, plasma_workspace_t *work,
                       plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_omp_zunmqr(plasma_enum_t side, plasma_enum_t trans,
                       plasma_desc_t *descA, plasma_desc_t *descT,
                       plasma_desc_t *descC, plasma_workspace_t *work,
                       plasma_sequence_t *sequence, plasma_request_t *request);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_PLASMA_Z_H
