/**
 *
 * @file plasma_z.h
 *
 *  PLASMA header.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver.
 *
 * @version 3.0.0
 * @author Jakub Kurzak
 * @date 2016-01-01
 * @precisions normal z -> s d c
 *
 **/
#ifndef PLASMA_Z_H
#define PLASMA_Z_H

/***************************************************************************//**
 *  Standard interface.
 **/
int PLASMA_zgemm(PLASMA_enum transA, PLASMA_enum transB,
                 int m, int n, int k,
                 PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int lda,
                                           PLASMA_Complex64_t *B, int ldb,
                 PLASMA_Complex64_t beta,  PLASMA_Complex64_t *C, int ldc);

/***************************************************************************//**
 *  Tile interface.
 **/
int PLASMA_zgemm_Tile(PLASMA_enum transA, PLASMA_enum transB,
                      PLASMA_Complex64_t alpha, PLASMA_desc *descA,
                                                PLASMA_desc *descB,
                      PLASMA_Complex64_t beta,  PLASMA_desc *descC);

/***************************************************************************//**
 *  Tile asynchronous interface.
 **/
int PLASMA_zgemm_Tile_Async(PLASMA_enum transA, PLASMA_enum transB,
                            PLASMA_Complex64_t alpha, PLASMA_desc *descA,
                                                      PLASMA_desc *descB,
                            PLASMA_Complex64_t beta,  PLASMA_desc *descC,
                            PLASMA_sequence *sequence, PLASMA_request *request);

/***************************************************************************//**
 *  Layout translation sync.
 **/
int PLASMA_zcm2ccrb(PLASMA_Complex64_t *Af77, int lda, PLASMA_desc *A);
int PLASMA_zccrb2cm(PLASMA_desc *A, PLASMA_Complex64_t *Af77, int lda);

/***************************************************************************//**
 *  Layout translation async.
 **/
int PLASMA_zcm2ccrb_Async(PLASMA_Complex64_t *Af77, int lda, PLASMA_desc *A,
                          PLASMA_sequence *sequence, PLASMA_request *request);

int PLASMA_zccrb2cm_Async(PLASMA_desc *A, PLASMA_Complex64_t *Af77, int lda,
                          PLASMA_sequence *sequence, PLASMA_request *request);

#endif // PLASMA_Z_H
