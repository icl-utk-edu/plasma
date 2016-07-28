/**
 *
 * @file plasma_z.h
 *
 *  PLASMA header.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of Manchester, Univ. of California Berkeley and
 *  Univ. of Colorado Denver.
 *
 * @version 3.0.0
 * @author Jakub Kurzak
 * @author Samuel D. Relton
 * @author Pedro V. Lara
 * @date 2016-05-16
 * @precisions normal z -> s d c
 *
 **/
#ifndef ICL_PLASMA_Z_H
#define ICL_PLASMA_Z_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************************************************************//**
 *  Standard interface.
 **/
int PLASMA_zgemm(PLASMA_enum transA, PLASMA_enum transB,
                 int m, int n, int k,
                 PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int lda,
                                           PLASMA_Complex64_t *B, int ldb,
                 PLASMA_Complex64_t beta,  PLASMA_Complex64_t *C, int ldc);

int PLASMA_zhemm(PLASMA_enum side, PLASMA_enum uplo,
                 int m, int n,
                 PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int lda,
                                           PLASMA_Complex64_t *B, int ldb,
                 PLASMA_Complex64_t beta,  PLASMA_Complex64_t *C, int ldc);

int PLASMA_zher2k(PLASMA_enum uplo, PLASMA_enum trans,
                  int n, int k,
                  PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int lda,
                                            PLASMA_Complex64_t *B, int ldb,
                  double beta,              PLASMA_Complex64_t *C, int ldc);

int PLASMA_zherk(PLASMA_enum uplo, PLASMA_enum trans,
                 int n, int k,
                 double alpha, PLASMA_Complex64_t *A, int lda,
                 double beta,  PLASMA_Complex64_t *C, int ldc);

int PLASMA_zposv(PLASMA_enum uplo, int n, int nrhs,
                 PLASMA_Complex64_t *A, int lda,
                 PLASMA_Complex64_t *B, int ldb);

int PLASMA_zpotrf(PLASMA_enum uplo, int n,
                  PLASMA_Complex64_t *A, int lda);

int PLASMA_zpotrs(PLASMA_enum uplo, int n, int nrhs,
                  PLASMA_Complex64_t *A, int lda,
                  PLASMA_Complex64_t *B, int ldb);

int PLASMA_zsymm(PLASMA_enum side, PLASMA_enum uplo,
                 int m, int n,
                 PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int lda,
                                           PLASMA_Complex64_t *B, int ldb,
                 PLASMA_Complex64_t beta,  PLASMA_Complex64_t *C, int ldc);

int PLASMA_zsyr2k(PLASMA_enum uplo, PLASMA_enum trans,
                  int n, int k,
                  PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int lda,
                                            PLASMA_Complex64_t *B, int ldb,
                  PLASMA_Complex64_t beta,  PLASMA_Complex64_t *C, int ldc);

int PLASMA_zsyrk(PLASMA_enum uplo, PLASMA_enum trans,
                 int n, int k,
                 PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int lda,
                 PLASMA_Complex64_t beta,  PLASMA_Complex64_t *C, int ldc);

int PLASMA_ztrsm(PLASMA_enum side, PLASMA_enum uplo,
                 PLASMA_enum transA, PLASMA_enum diag,
                 int n, int nrhs,
                 PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int lda,
                                           PLASMA_Complex64_t *B, int ldb);

/***************************************************************************//**
 *  Tile asynchronous interface.
 **/
void PLASMA_zgemm_Tile_Async(PLASMA_enum transA, PLASMA_enum transB,
                             PLASMA_Complex64_t alpha, PLASMA_desc *descA,
                                                       PLASMA_desc *descB,
                             PLASMA_Complex64_t beta,  PLASMA_desc *descC,
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

void PLASMA_ztrsm_Tile_Async(PLASMA_enum side, PLASMA_enum uplo,
                             PLASMA_enum transA, PLASMA_enum diag,
                             PLASMA_Complex64_t alpha, PLASMA_desc *A,
                                                       PLASMA_desc *B,
                             PLASMA_sequence *sequence,
                             PLASMA_request *request);

/***************************************************************************//**
 *  Layout translation async.
 **/
void PLASMA_zcm2ccrb_Async(PLASMA_Complex64_t *Af77, int lda, PLASMA_desc *A,
                           PLASMA_sequence *sequence, PLASMA_request *request);

void PLASMA_zccrb2cm_Async(PLASMA_desc *A, PLASMA_Complex64_t *Af77, int lda,
                           PLASMA_sequence *sequence, PLASMA_request *request);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_PLASMA_Z_H
