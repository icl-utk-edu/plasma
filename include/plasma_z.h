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
                  PLASMA_Complex64_t *A, int lda,
                  PLASMA_desc *descT);

int PLASMA_zgelqs(int m, int n, int nrhs,
                  PLASMA_Complex64_t *A, int lda,
                  PLASMA_desc *descT,
                  PLASMA_Complex64_t *B, int ldb);

int PLASMA_zgels(PLASMA_enum trans, int m, int n, int nrhs,
                 PLASMA_Complex64_t *A, int lda,
                 PLASMA_desc *descT,
                 PLASMA_Complex64_t *B, int ldb);

int PLASMA_zgemm(PLASMA_enum transA, PLASMA_enum transB,
                 int m, int n, int k,
                 PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int lda,
                                           PLASMA_Complex64_t *B, int ldb,
                 PLASMA_Complex64_t beta,  PLASMA_Complex64_t *C, int ldc);

int PLASMA_zgeqrf(int m, int n,
                  PLASMA_Complex64_t *A, int lda,
                  PLASMA_desc *descT);

int PLASMA_zgeqrs(int m, int n, int nrhs,
                  PLASMA_Complex64_t *A, int lda,
                  PLASMA_desc *descT,
                  PLASMA_Complex64_t *B, int ldb);

int PLASMA_zhemm(PLASMA_enum side, PLASMA_enum uplo, int m, int n,
                 PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int lda,
                                           PLASMA_Complex64_t *B, int ldb,
                 PLASMA_Complex64_t beta,  PLASMA_Complex64_t *C, int ldc);

int PLASMA_zher2k(PLASMA_enum uplo, PLASMA_enum trans,
                  int n, int k,
                  PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int lda,
                                            PLASMA_Complex64_t *B, int ldb,
                               double beta, PLASMA_Complex64_t *C, int ldc);

int PLASMA_zherk(PLASMA_enum uplo, PLASMA_enum trans,
                 int n, int k,
                 double alpha, PLASMA_Complex64_t *A, int lda,
                 double beta,  PLASMA_Complex64_t *C, int ldc);

int PLASMA_zpbsv(PLASMA_enum uplo, int n, int kd, int nrhs,
                 PLASMA_Complex64_t *AB, int ldab,
                 PLASMA_Complex64_t *B, int ldb);

int PLASMA_zpbtrs(PLASMA_enum uplo, int n, int kd, int nrhs,
                  PLASMA_Complex64_t *AB, int ldab,
                  PLASMA_Complex64_t *B, int ldb);

int PLASMA_zpbtrf(PLASMA_enum uplo,
                  int n, int kd,
                  PLASMA_Complex64_t *AB, int ldab);

int PLASMA_zposv(PLASMA_enum uplo, int n, int nrhs,
                 PLASMA_Complex64_t *A, int lda,
                 PLASMA_Complex64_t *B, int ldb);

int PLASMA_zpotrf(PLASMA_enum uplo,
                  int n,
                  PLASMA_Complex64_t *A, int lda);

int PLASMA_zpotrs(PLASMA_enum uplo,
                  int n, int nrhs,
                  PLASMA_Complex64_t *A, int lda,
                  PLASMA_Complex64_t *B, int ldb);

int PLASMA_zsymm(PLASMA_enum side, PLASMA_enum uplo, int m, int n,
                 PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int lda,
                                           PLASMA_Complex64_t *B, int ldb,
                 PLASMA_Complex64_t beta,  PLASMA_Complex64_t *C, int ldc);

int PLASMA_zsyr2k(PLASMA_enum uplo, PLASMA_enum trans,
                  int n, int k,
                  PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int lda,
                                            PLASMA_Complex64_t *B, int ldb,
                  PLASMA_Complex64_t beta,  PLASMA_Complex64_t *C, int ldc);

int PLASMA_zsyrk(PLASMA_enum uplo, PLASMA_enum trans, int n, int k,
                 PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int lda,
                 PLASMA_Complex64_t beta,  PLASMA_Complex64_t *C, int ldc);

int PLASMA_ztradd(PLASMA_enum uplo, PLASMA_enum transA, int m, int n,
                  PLASMA_Complex64_t  alpha,
                  PLASMA_Complex64_t *A, int lda,
                  PLASMA_Complex64_t  beta,
                  PLASMA_Complex64_t *B, int ldb);

int PLASMA_ztrmm(PLASMA_enum side, PLASMA_enum uplo,
                 PLASMA_enum transA, PLASMA_enum diag,
                 int m, int n, PLASMA_Complex64_t alpha,
                 PLASMA_Complex64_t *A, int lda,
                 PLASMA_Complex64_t *B, int ldb);

int PLASMA_ztrsm(PLASMA_enum side, PLASMA_enum uplo,
                 PLASMA_enum transA, PLASMA_enum diag,
                 int m, int n,
                 PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int lda,
                                           PLASMA_Complex64_t *B, int ldb);

int PLASMA_zunglq(int m, int n, int k,
                  PLASMA_Complex64_t *A, int lda,
                  PLASMA_desc *descT,
                  PLASMA_Complex64_t *Q, int ldq);

int PLASMA_zungqr(int m, int n, int k,
                  PLASMA_Complex64_t *A, int lda,
                  PLASMA_desc *descT,
                  PLASMA_Complex64_t *Q, int ldq);

int PLASMA_zunmlq(PLASMA_enum side, PLASMA_enum trans, int m, int n, int k,
                  PLASMA_Complex64_t *A, int lda,
                  PLASMA_desc *descT,
                  PLASMA_Complex64_t *C, int ldc);

int PLASMA_zunmqr(PLASMA_enum side, PLASMA_enum trans, int m, int n, int k,
                  PLASMA_Complex64_t *A, int lda,
                  PLASMA_desc *descT,
                  PLASMA_Complex64_t *C, int ldc);

/***************************************************************************//**
 *  Tile asynchronous interface.
 **/

void PLASMA_zccrb2cm_Async(PLASMA_desc *A, PLASMA_Complex64_t *Af77, int lda,
                           PLASMA_sequence *sequence,
                           PLASMA_request *request);

void PLASMA_zccrb2cm_band_Async(PLASMA_enum uplo,
                                PLASMA_desc *A,
                                PLASMA_Complex64_t *Af77, int lda,
                                PLASMA_sequence *sequence,
                                PLASMA_request *request);

void PLASMA_zcm2ccrb_Async(PLASMA_Complex64_t *Af77, int lda,
                           PLASMA_desc *A,
                           PLASMA_sequence *sequence,
                           PLASMA_request *request);

void PLASMA_zcm2ccrb_band_Async(PLASMA_enum uplo,
                                PLASMA_Complex64_t *Af77, int lda,
                                PLASMA_desc *A,
                                PLASMA_sequence *sequence,
                                PLASMA_request *request);

void PLASMA_zgelqf_Tile_Async(PLASMA_desc *descA, PLASMA_desc *descT,
                              PLASMA_workspace *work,
                              PLASMA_sequence *sequence,
                              PLASMA_request *request);

void PLASMA_zgelqs_Tile_Async(PLASMA_desc *descA,
                              PLASMA_desc *descT,
                              PLASMA_desc *descB,
                              PLASMA_workspace *work,
                              PLASMA_sequence *sequence,
                              PLASMA_request *request);

void PLASMA_zgels_Tile_Async(PLASMA_enum trans,
                             PLASMA_desc *descA, PLASMA_desc *descT,
                             PLASMA_desc *descB,
                             PLASMA_workspace *work,
                             PLASMA_sequence *sequence,
                             PLASMA_request *request);

void PLASMA_zgemm_Tile_Async(PLASMA_enum transA, PLASMA_enum transB,
                             PLASMA_Complex64_t alpha, PLASMA_desc *A,
                                                       PLASMA_desc *B,
                             PLASMA_Complex64_t beta,  PLASMA_desc *C,
                             PLASMA_sequence *sequence,
                             PLASMA_request *request);

void PLASMA_zgeqrf_Tile_Async(PLASMA_desc *descA, PLASMA_desc *descT,
                              PLASMA_workspace *work,
                              PLASMA_sequence *sequence,
                              PLASMA_request *request);

void PLASMA_zgeqrs_Tile_Async(PLASMA_desc *descA, PLASMA_desc *descT,
                              PLASMA_desc *descB,
                              PLASMA_workspace *work,
                              PLASMA_sequence *sequence,
                              PLASMA_request *request);

void PLASMA_zhemm_Tile_Async(PLASMA_enum side, PLASMA_enum uplo,
                             PLASMA_Complex64_t alpha, PLASMA_desc *A,
                                                       PLASMA_desc *B,
                             PLASMA_Complex64_t beta,  PLASMA_desc *C,
                             PLASMA_sequence *sequence,
                             PLASMA_request *request);

void PLASMA_zher2k_Tile_Async(PLASMA_enum uplo, PLASMA_enum trans,
                              PLASMA_Complex64_t alpha, PLASMA_desc *A,
                                                        PLASMA_desc *B,
                              double beta,              PLASMA_desc *C,
                              PLASMA_sequence *sequence,
                              PLASMA_request *request);

void PLASMA_zherk_Tile_Async(PLASMA_enum uplo, PLASMA_enum trans,
                             double alpha, PLASMA_desc *A,
                             double beta,  PLASMA_desc *C,
                             PLASMA_sequence *sequence,
                             PLASMA_request *request);

void PLASMA_zpbsv_Tile_Async(PLASMA_enum uplo,
                             PLASMA_desc *AB,
                             PLASMA_desc *B,
                             PLASMA_sequence *sequence,
                             PLASMA_request *request);

void PLASMA_zpbsv_Tile_Async(PLASMA_enum uplo,
                             PLASMA_desc *AB,
                             PLASMA_desc *B,
                             PLASMA_sequence *sequence,
                             PLASMA_request *request);

void PLASMA_zpbtrf_Tile_Async(PLASMA_enum uplo,
                              PLASMA_desc *AB,
                              PLASMA_sequence *sequence,
                              PLASMA_request *request);

void PLASMA_zpbtrs_Tile_Async(PLASMA_enum uplo,
                              PLASMA_desc *AB,
                              PLASMA_desc *B,
                              PLASMA_sequence *sequence,
                              PLASMA_request *request);

void PLASMA_zposv_Tile_Async(PLASMA_enum uplo,
                             PLASMA_desc *A,
                             PLASMA_desc *B,
                             PLASMA_sequence *sequence,
                             PLASMA_request *request);

void PLASMA_zpotrf_Tile_Async(PLASMA_enum uplo,
                              PLASMA_desc *A,
                              PLASMA_sequence *sequence,
                              PLASMA_request *request);

void PLASMA_zpotrs_Tile_Async(PLASMA_enum uplo,
                              PLASMA_desc *A,
                              PLASMA_desc *B,
                              PLASMA_sequence *sequence,
                              PLASMA_request *request);

void PLASMA_zsymm_Tile_Async(PLASMA_enum side, PLASMA_enum uplo,
                             PLASMA_Complex64_t alpha, PLASMA_desc *A,
                                                       PLASMA_desc *B,
                             PLASMA_Complex64_t beta,  PLASMA_desc *C,
                             PLASMA_sequence *sequence,
                             PLASMA_request *request);

void PLASMA_zsyr2k_Tile_Async(PLASMA_enum uplo, PLASMA_enum trans,
                              PLASMA_Complex64_t alpha, PLASMA_desc *A,
                                                        PLASMA_desc *B,
                              PLASMA_Complex64_t beta,  PLASMA_desc *C,
                              PLASMA_sequence *sequence,
                              PLASMA_request *request);

void PLASMA_zsyrk_Tile_Async(PLASMA_enum uplo, PLASMA_enum trans,
                             PLASMA_Complex64_t alpha, PLASMA_desc *A,
                             PLASMA_Complex64_t beta,  PLASMA_desc *C,
                             PLASMA_sequence *sequence,
                             PLASMA_request *request);

void PLASMA_ztradd_Tile_Async(PLASMA_enum uplo, PLASMA_enum transA,
                              PLASMA_Complex64_t alpha, PLASMA_desc *A,
                              PLASMA_Complex64_t beta,  PLASMA_desc *B,
                              PLASMA_sequence *sequence,
                              PLASMA_request  *request);

void PLASMA_ztrmm_Tile_Async(PLASMA_enum side, PLASMA_enum uplo,
                             PLASMA_enum transA, PLASMA_enum diag,
                             PLASMA_Complex64_t alpha, PLASMA_desc *A,
                                                       PLASMA_desc *B,
                             PLASMA_sequence *sequence,
                             PLASMA_request  *request);

void PLASMA_ztrsm_Tile_Async(PLASMA_enum side, PLASMA_enum uplo,
                             PLASMA_enum transA, PLASMA_enum diag,
                             PLASMA_Complex64_t alpha, PLASMA_desc *A,
                                                       PLASMA_desc *B,
                             PLASMA_sequence *sequence,
                             PLASMA_request *request);

void PLASMA_zunglq_Tile_Async(PLASMA_desc *descA,
                              PLASMA_desc *descT,
                              PLASMA_desc *descQ,
                              PLASMA_workspace *work,
                              PLASMA_sequence *sequence,
                              PLASMA_request *request);

void PLASMA_zungqr_Tile_Async(PLASMA_desc *descA, PLASMA_desc *descT,
                              PLASMA_desc *descQ,
                              PLASMA_workspace *work,
                              PLASMA_sequence *sequence,
                              PLASMA_request *request);

void PLASMA_zunmlq_Tile_Async(PLASMA_enum side, PLASMA_enum trans,
                              PLASMA_desc *descA, PLASMA_desc *descT,
                              PLASMA_desc *descC,
                              PLASMA_workspace *work,
                              PLASMA_sequence *sequence,
                              PLASMA_request *request);

void PLASMA_zunmqr_Tile_Async(PLASMA_enum side, PLASMA_enum trans,
                              PLASMA_desc *descA, PLASMA_desc *descT,
                              PLASMA_desc *descC,
                              PLASMA_workspace *work,
                              PLASMA_sequence *sequence,
                              PLASMA_request *request);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_PLASMA_Z_H
