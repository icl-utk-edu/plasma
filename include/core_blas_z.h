/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> s d c
 *
 **/
#ifndef ICL_CORE_BLAS_Z_H
#define ICL_CORE_BLAS_Z_H

#include "plasma_async.h"
#include "plasma_types.h"
#include "plasma_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************/
void CORE_zgelqt(int m, int n, int ib,
                 PLASMA_Complex64_t *A, int lda,
                 PLASMA_Complex64_t *T, int ldt,
                 PLASMA_Complex64_t *TAU,
                 PLASMA_Complex64_t *WORK);

void CORE_zgemm(PLASMA_enum transA, PLASMA_enum transB,
                int m, int n, int k,
                PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                                          const PLASMA_Complex64_t *B, int ldb,
                PLASMA_Complex64_t beta,        PLASMA_Complex64_t *C, int ldc);

void CORE_zgemm(PLASMA_enum transA, PLASMA_enum transB,
                int m, int n, int k,
                PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                                          const PLASMA_Complex64_t *B, int ldb,
                PLASMA_Complex64_t beta,        PLASMA_Complex64_t *C, int ldc);

void CORE_zgeqrt(int m, int n, int ib,
                 PLASMA_Complex64_t *A, int lda,
                 PLASMA_Complex64_t *T, int ldt,
                 PLASMA_Complex64_t *TAU,
                 PLASMA_Complex64_t *WORK, int lwork);

void CORE_zhemm(PLASMA_enum side, PLASMA_enum uplo,
                int m, int n,
                PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                                          const PLASMA_Complex64_t *B, int ldb,
                PLASMA_Complex64_t beta,        PLASMA_Complex64_t *C, int ldc);

void CORE_zher2k(PLASMA_enum uplo, PLASMA_enum trans,
                 int n, int k,
                 PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                                           const PLASMA_Complex64_t *B, int ldb,
                 double beta,                    PLASMA_Complex64_t *C, int ldc);

void CORE_zherk(PLASMA_enum uplo, PLASMA_enum trans,
                int n, int k,
                double alpha, const PLASMA_Complex64_t *A, int lda,
                double beta,        PLASMA_Complex64_t *C, int ldc);

void CORE_zlacpy(PLASMA_enum uplo,
                 int m, int n,
                 const PLASMA_Complex64_t *A, int lda,
                       PLASMA_Complex64_t *B, int ldb);

void CORE_zlaset(PLASMA_enum uplo,
                 int m, int n,
                 PLASMA_Complex64_t alpha, PLASMA_Complex64_t beta,
                 PLASMA_Complex64_t *A, int lda);

void CORE_zpamm(int op, PLASMA_enum side, PLASMA_enum storev,
                int m, int n, int k, int l,
                const PLASMA_Complex64_t *A1, int lda1,
                      PLASMA_Complex64_t *A2, int lda2,
                const PLASMA_Complex64_t *V,  int ldv,
                      PLASMA_Complex64_t *W,  int ldw);

void CORE_zparfb(PLASMA_enum side, PLASMA_enum trans, PLASMA_enum direct,
                 PLASMA_enum storev,
                 int m1, int n1, int m2, int n2, int k, int l,
                       PLASMA_Complex64_t *A1,   int lda1,
                       PLASMA_Complex64_t *A2,   int lda2,
                 const PLASMA_Complex64_t *V,    int ldv,
                 const PLASMA_Complex64_t *T,    int ldt,
                       PLASMA_Complex64_t *WORK, int ldwork);

int CORE_zpotrf(PLASMA_enum uplo,
                int n,
                PLASMA_Complex64_t *A, int lda);

void CORE_zsymm(PLASMA_enum side, PLASMA_enum uplo,
                int m, int n,
                PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                                          const PLASMA_Complex64_t *B, int ldb,
                PLASMA_Complex64_t beta,        PLASMA_Complex64_t *C, int ldc);

void CORE_zsyr2k(
    PLASMA_enum uplo, PLASMA_enum trans,
    int n, int k,
    PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                              const PLASMA_Complex64_t *B, int ldb,
    PLASMA_Complex64_t beta,        PLASMA_Complex64_t *C, int ldc);

void CORE_zsyrk(PLASMA_enum uplo, PLASMA_enum trans,
                int n, int k,
                PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                PLASMA_Complex64_t beta,        PLASMA_Complex64_t *C, int ldc);

void CORE_ztrmm(PLASMA_enum side, PLASMA_enum uplo,
                PLASMA_enum transA, PLASMA_enum diag,
                int m, int n,
                PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                                                PLASMA_Complex64_t *B, int ldb);

void CORE_ztrsm(PLASMA_enum side, PLASMA_enum uplo,
                PLASMA_enum transA, PLASMA_enum diag,
                int m, int n,
                PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                                                PLASMA_Complex64_t *B, int ldb);

void CORE_ztslqt(int m, int n, int ib,
                 PLASMA_Complex64_t *A1, int lda1,
                 PLASMA_Complex64_t *A2, int lda2,
                 PLASMA_Complex64_t *T,  int ldt,
                 PLASMA_Complex64_t *TAU, PLASMA_Complex64_t *WORK);

void CORE_ztsmlq(PLASMA_enum side, PLASMA_enum trans,
                 int m1, int n1, int m2, int n2, int k, int ib,
                       PLASMA_Complex64_t *A1,   int lda1,
                       PLASMA_Complex64_t *A2,   int lda2,
                 const PLASMA_Complex64_t *V,    int ldv,
                 const PLASMA_Complex64_t *T,    int ldt,
                       PLASMA_Complex64_t *WORK, int ldwork);

void CORE_ztsmqr(PLASMA_enum side, PLASMA_enum trans,
                 int m1, int n1, int m2, int n2, int k, int ib,
                       PLASMA_Complex64_t *A1,   int lda1,
                       PLASMA_Complex64_t *A2,   int lda2,
                 const PLASMA_Complex64_t *V,    int ldv,
                 const PLASMA_Complex64_t *T,    int ldt,
                       PLASMA_Complex64_t *WORK, int ldwork);

void CORE_ztsqrt(int m, int n, int ib,
                 PLASMA_Complex64_t *A1, int lda1,
                 PLASMA_Complex64_t *A2, int lda2,
                 PLASMA_Complex64_t *T,  int ldt,
                 PLASMA_Complex64_t *TAU,
                 PLASMA_Complex64_t *WORK);

void CORE_zunmlq(PLASMA_enum side, PLASMA_enum trans,
                 int m, int n, int k, int ib,
                 const PLASMA_Complex64_t *A,    int lda,
                 const PLASMA_Complex64_t *T,    int ldt,
                       PLASMA_Complex64_t *C,    int ldc,
                       PLASMA_Complex64_t *WORK, int ldwork);

void CORE_zunmqr(PLASMA_enum side, PLASMA_enum trans,
                 int m, int n, int k, int ib,
                 const PLASMA_Complex64_t *A,    int lda,
                 const PLASMA_Complex64_t *T,    int ldt,
                       PLASMA_Complex64_t *C,    int ldc,
                       PLASMA_Complex64_t *WORK, int ldwork);

/******************************************************************************/
void CORE_OMP_zgelqt(int m, int n, int ib, int nb,
                     PLASMA_Complex64_t *A, int lda,
                     PLASMA_Complex64_t *T, int ldt);

void CORE_OMP_zgemm(
    PLASMA_enum transA, PLASMA_enum transB,
    int m, int n, int k,
    PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                              const PLASMA_Complex64_t *B, int ldb,
    PLASMA_Complex64_t beta,        PLASMA_Complex64_t *C, int ldc);

void CORE_OMP_zgeqrt(int m, int n, int ib, int nb,
                     PLASMA_Complex64_t *A, int lda,
                     PLASMA_Complex64_t *T, int ldt,
                     PLASMA_workspace *work);

void CORE_OMP_zhemm(
    PLASMA_enum side, PLASMA_enum uplo,
    int m, int n,
    PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                              const PLASMA_Complex64_t *B, int ldb,
    PLASMA_Complex64_t beta,        PLASMA_Complex64_t *C, int ldc);

void CORE_OMP_zher2k(
    PLASMA_enum uplo, PLASMA_enum trans,
    int n, int k,
    PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                              const PLASMA_Complex64_t *B, int ldb,
    double beta,                    PLASMA_Complex64_t *C, int ldc);

void CORE_OMP_zherk(PLASMA_enum uplo, PLASMA_enum trans,
                    int n, int k,
                    double alpha, const PLASMA_Complex64_t *A, int lda,
                    double beta,        PLASMA_Complex64_t *C, int ldc);

void CORE_OMP_zlacpy(PLASMA_enum uplo,
                     int m, int n, int nb,
                     const PLASMA_Complex64_t *A, int lda,
                           PLASMA_Complex64_t *B, int ldb);

void CORE_OMP_zlaset(PLASMA_enum uplo,
                     int m, int n,
                     PLASMA_Complex64_t alpha, PLASMA_Complex64_t beta,
                     PLASMA_Complex64_t *A, int lda);

void CORE_OMP_zpotrf(PLASMA_enum uplo,
                     int n,
                     PLASMA_Complex64_t *A, int lda,
                     PLASMA_sequence *sequence, PLASMA_request *request,
                     int iinfo);

void CORE_OMP_zsymm(
    PLASMA_enum side, PLASMA_enum uplo,
    int m, int n,
    PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                              const PLASMA_Complex64_t *B, int ldb,
    PLASMA_Complex64_t beta,        PLASMA_Complex64_t *C, int ldc);

void CORE_OMP_zsyr2k(
    PLASMA_enum uplo, PLASMA_enum trans,
    int n, int k,
    PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                              const PLASMA_Complex64_t *B, int ldb,
    PLASMA_Complex64_t beta,        PLASMA_Complex64_t *C, int ldc);

void CORE_OMP_zsyrk(
    PLASMA_enum uplo, PLASMA_enum trans,
    int n, int k,
    PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
    PLASMA_Complex64_t beta,        PLASMA_Complex64_t *C, int ldc);

void CORE_OMP_ztrmm(
    PLASMA_enum side, PLASMA_enum uplo,
    PLASMA_enum transA, PLASMA_enum diag,
    int m, int n,
    PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                                    PLASMA_Complex64_t *B, int ldb);

void CORE_OMP_ztrsm(
    PLASMA_enum side, PLASMA_enum uplo,
    PLASMA_enum transA, PLASMA_enum diag,
    int m, int n,
    PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                                    PLASMA_Complex64_t *B, int ldb);

void CORE_OMP_ztslqt(int m, int n, int ib, int nb,
                     PLASMA_Complex64_t *A1, int lda1,
                     PLASMA_Complex64_t *A2, int lda2,
                     PLASMA_Complex64_t *T,  int ldt);

void CORE_OMP_ztsmlq(PLASMA_enum side, PLASMA_enum trans,
                     int m1, int n1, int m2, int n2, int k, int ib, int nb,
                           PLASMA_Complex64_t *A1, int lda1,
                           PLASMA_Complex64_t *A2, int lda2,
                     const PLASMA_Complex64_t *V,  int ldv,
                     const PLASMA_Complex64_t *T,  int ldt);

void CORE_OMP_ztsmqr(PLASMA_enum side, PLASMA_enum trans,
                     int m1, int n1, int m2, int n2, int k, int ib, int nb,
                           PLASMA_Complex64_t *A1, int lda1,
                           PLASMA_Complex64_t *A2, int lda2,
                     const PLASMA_Complex64_t *V, int ldv,
                     const PLASMA_Complex64_t *T, int ldt);

void CORE_OMP_ztsqrt(int m, int n, int ib, int nb,
                     PLASMA_Complex64_t *A1, int lda1,
                     PLASMA_Complex64_t *A2, int lda2,
                     PLASMA_Complex64_t *T,  int ldt);

void CORE_OMP_zunmlq(PLASMA_enum side, PLASMA_enum trans,
                     int m, int n, int k, int ib, int nb,
                     const PLASMA_Complex64_t *A, int lda,
                     const PLASMA_Complex64_t *T, int ldt,
                           PLASMA_Complex64_t *C, int ldc);

void CORE_OMP_zunmqr(PLASMA_enum side, PLASMA_enum trans,
                     int m, int n, int k, int ib, int nb,
                     const PLASMA_Complex64_t *A, int lda,
                     const PLASMA_Complex64_t *T, int ldt,
                           PLASMA_Complex64_t *C, int ldc);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_CORE_BLAS_Z_H
