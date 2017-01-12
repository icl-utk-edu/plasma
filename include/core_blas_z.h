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
#include "plasma_barrier.h"
#include "plasma_descriptor.h"
#include "plasma_types.h"
#include "plasma_workspace.h"
#include "plasma_descriptor.h"

#ifdef __cplusplus
extern "C" {
#endif

#define COMPLEX

/******************************************************************************/
#ifdef COMPLEX
double core_dcabs1(plasma_complex64_t alpha);
#endif

int core_zgeadd(plasma_enum_t transa,
                int m, int n,
                plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
                plasma_complex64_t beta,        plasma_complex64_t *B, int ldb);

int core_zgelqt(int m, int n, int ib,
                plasma_complex64_t *A, int lda,
                plasma_complex64_t *T, int ldt,
                plasma_complex64_t *tau,
                plasma_complex64_t *work);

void core_zgemm(plasma_enum_t transa, plasma_enum_t transb,
                int m, int n, int k,
                plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
                                          const plasma_complex64_t *B, int ldb,
                plasma_complex64_t beta,        plasma_complex64_t *C, int ldc);

int core_zgeqrt(int m, int n, int ib,
                plasma_complex64_t *A, int lda,
                plasma_complex64_t *T, int ldt,
                plasma_complex64_t *tau,
                plasma_complex64_t *work);

void core_zgessq(int m, int n,
                 const plasma_complex64_t *A, int lda,
                 double *scale, double *sumsq);

int core_zgetrf(plasma_desc_t A, int *ipiv, int ib, int rank, int size,
                plasma_barrier_t *barrier);

void core_zhemm(plasma_enum_t side, plasma_enum_t uplo,
                int m, int n,
                plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
                                          const plasma_complex64_t *B, int ldb,
                plasma_complex64_t beta,        plasma_complex64_t *C, int ldc);

void core_zher2k(plasma_enum_t uplo, plasma_enum_t trans,
                 int n, int k,
                 plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
                                           const plasma_complex64_t *B, int ldb,
                 double beta,                    plasma_complex64_t *C, int ldc);

void core_zherk(plasma_enum_t uplo, plasma_enum_t trans,
                int n, int k,
                double alpha, const plasma_complex64_t *A, int lda,
                double beta,        plasma_complex64_t *C, int ldc);

void core_zhessq(plasma_enum_t uplo,
                 int n,
                 const plasma_complex64_t *A, int lda,
                 double *scale, double *sumsq);

void core_zsyssq(plasma_enum_t uplo,
                 int n,
                 const plasma_complex64_t *A, int lda,
                 double *scale, double *sumsq);

void core_zlacpy(plasma_enum_t uplo,
                 int m, int n,
                 const plasma_complex64_t *A, int lda,
                       plasma_complex64_t *B, int ldb);

void core_zlacpy_lapack2tile_band(plasma_enum_t uplo,
                                  int it, int jt,
                                  int m, int n, int nb, int kl, int ku,
                                  const plasma_complex64_t *A, int lda,
                                        plasma_complex64_t *B, int ldb);

void core_zlacpy_tile2lapack_band(plasma_enum_t uplo,
                                  int it, int jt,
                                  int m, int n, int nb, int kl, int ku,
                                  const plasma_complex64_t *B, int ldb,
                                        plasma_complex64_t *A, int lda);

void core_zlange(plasma_enum_t norm,
                 int m, int n,
                 const plasma_complex64_t *A, int lda,
                 double *work, double *result);

void core_zlanhe(plasma_enum_t norm, plasma_enum_t uplo,
                 int n,
                 const plasma_complex64_t *A, int lda,
                 double *work, double *value);

void core_zlansy(plasma_enum_t norm, plasma_enum_t uplo,
                 int n,
                 const plasma_complex64_t *A, int lda,
                 double *work, double *value);

void core_zlantr(plasma_enum_t norm, plasma_enum_t uplo, plasma_enum_t diag,
                 int m, int n,
                 const plasma_complex64_t *A, int lda,
                 double *work, double *value);

void core_zlascl(plasma_enum_t uplo,
                 double cfrom, double cto,
                 int m, int n,
                 plasma_complex64_t *A, int lda);

void core_zlaset(plasma_enum_t uplo,
                 int m, int n,
                 plasma_complex64_t alpha, plasma_complex64_t beta,
                 plasma_complex64_t *A, int lda);

void core_zlaswp(plasma_enum_t colrow,
                 plasma_desc_t A, int k1, int k2, const int *ipiv, int incx);

int core_zlauum(plasma_enum_t uplo,
                int n,
                plasma_complex64_t *A, int lda);

int core_zpamm(int op, plasma_enum_t side, plasma_enum_t storev,
               int m, int n, int k, int l,
               const plasma_complex64_t *A1, int lda1,
                     plasma_complex64_t *A2, int lda2,
               const plasma_complex64_t *V,  int ldv,
                     plasma_complex64_t *W,  int ldw);

int core_zparfb(plasma_enum_t side, plasma_enum_t trans, plasma_enum_t direct,
                plasma_enum_t storev,
                int m1, int n1, int m2, int n2, int k, int l,
                      plasma_complex64_t *A1,   int lda1,
                      plasma_complex64_t *A2,   int lda2,
                const plasma_complex64_t *V,    int ldv,
                const plasma_complex64_t *T,    int ldt,
                      plasma_complex64_t *work, int ldwork);

int core_zpemv(plasma_enum_t trans, int storev,
               int m, int n, int l,
               plasma_complex64_t alpha,
               const plasma_complex64_t *A, int lda,
               const plasma_complex64_t *X, int incx,
               plasma_complex64_t beta,
               plasma_complex64_t *Y, int incy,
               plasma_complex64_t *work);

int core_zpotrf(plasma_enum_t uplo,
                int n,
                plasma_complex64_t *A, int lda);

void core_zsymm(plasma_enum_t side, plasma_enum_t uplo,
                int m, int n,
                plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
                                          const plasma_complex64_t *B, int ldb,
                plasma_complex64_t beta,        plasma_complex64_t *C, int ldc);

void core_zsyr2k(
    plasma_enum_t uplo, plasma_enum_t trans,
    int n, int k,
    plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
                              const plasma_complex64_t *B, int ldb,
    plasma_complex64_t beta,        plasma_complex64_t *C, int ldc);

void core_zsyrk(plasma_enum_t uplo, plasma_enum_t trans,
                int n, int k,
                plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
                plasma_complex64_t beta,        plasma_complex64_t *C, int ldc);

int core_ztradd(plasma_enum_t uplo, plasma_enum_t transa,
                int m, int n,
                plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
                plasma_complex64_t beta,        plasma_complex64_t *B, int ldb);

void core_ztrmm(plasma_enum_t side, plasma_enum_t uplo,
                plasma_enum_t transa, plasma_enum_t diag,
                int m, int n,
                plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
                                                plasma_complex64_t *B, int ldb);

void core_ztrsm(plasma_enum_t side, plasma_enum_t uplo,
                plasma_enum_t transa, plasma_enum_t diag,
                int m, int n,
                plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
                                                plasma_complex64_t *B, int ldb);

void core_ztrssq(plasma_enum_t uplo, plasma_enum_t diag,
                 int m, int n,
                 const plasma_complex64_t *A, int lda,
                 double *scale, double *sumsq);

int core_ztrtri(plasma_enum_t uplo, plasma_enum_t diag,
                int n,
                plasma_complex64_t *A, int lda);

int core_ztslqt(int m, int n, int ib,
                plasma_complex64_t *A1, int lda1,
                plasma_complex64_t *A2, int lda2,
                plasma_complex64_t *T,  int ldt,
                plasma_complex64_t *tau,
                plasma_complex64_t *work);

int core_ztsmlq(plasma_enum_t side, plasma_enum_t trans,
                int m1, int n1, int m2, int n2, int k, int ib,
                      plasma_complex64_t *A1,   int lda1,
                      plasma_complex64_t *A2,   int lda2,
                const plasma_complex64_t *V,    int ldv,
                const plasma_complex64_t *T,    int ldt,
                      plasma_complex64_t *work, int ldwork);

int core_ztsmqr(plasma_enum_t side, plasma_enum_t trans,
                int m1, int n1, int m2, int n2, int k, int ib,
                      plasma_complex64_t *A1,   int lda1,
                      plasma_complex64_t *A2,   int lda2,
                const plasma_complex64_t *V,    int ldv,
                const plasma_complex64_t *T,    int ldt,
                      plasma_complex64_t *work, int ldwork);

int core_ztsqrt(int m, int n, int ib,
                plasma_complex64_t *A1, int lda1,
                plasma_complex64_t *A2, int lda2,
                plasma_complex64_t *T,  int ldt,
                plasma_complex64_t *tau,
                plasma_complex64_t *work);

int core_zttlqt(int m, int n, int ib,
                plasma_complex64_t *A1, int lda1,
                plasma_complex64_t *A2, int lda2,
                plasma_complex64_t *T,  int ldt,
                plasma_complex64_t *tau,
                plasma_complex64_t *work);

int core_zttmlq(plasma_enum_t side, plasma_enum_t trans,
                int m1, int n1, int m2, int n2, int k, int ib,
                      plasma_complex64_t *A1,   int lda1,
                      plasma_complex64_t *A2,   int lda2,
                const plasma_complex64_t *V,    int ldv,
                const plasma_complex64_t *T,    int ldt,
                      plasma_complex64_t *work, int ldwork);

int core_zttmqr(plasma_enum_t side, plasma_enum_t trans,
                int m1, int n1, int m2, int n2, int k, int ib,
                      plasma_complex64_t *A1,   int lda1,
                      plasma_complex64_t *A2,   int lda2,
                const plasma_complex64_t *V,    int ldv,
                const plasma_complex64_t *T,    int ldt,
                      plasma_complex64_t *work, int ldwork);

int core_zttqrt(int m, int n, int ib,
                plasma_complex64_t *A1, int lda1,
                plasma_complex64_t *A2, int lda2,
                plasma_complex64_t *T,  int ldt,
                plasma_complex64_t *tau,
                plasma_complex64_t *work);

int core_zunmlq(plasma_enum_t side, plasma_enum_t trans,
                int m, int n, int k, int ib,
                const plasma_complex64_t *A,    int lda,
                const plasma_complex64_t *T,    int ldt,
                      plasma_complex64_t *C,    int ldc,
                      plasma_complex64_t *work, int ldwork);

int core_zunmqr(plasma_enum_t side, plasma_enum_t trans,
                int m, int n, int k, int ib,
                const plasma_complex64_t *A,    int lda,
                const plasma_complex64_t *T,    int ldt,
                      plasma_complex64_t *C,    int ldc,
                      plasma_complex64_t *work, int ldwork);

/******************************************************************************/
void core_omp_dzamax(int colrow, int m, int n,
                     const plasma_complex64_t *A, int lda,
                     double *values,
                     plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zgeadd(
    plasma_enum_t transa, int m, int n,
    plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
    plasma_complex64_t beta,        plasma_complex64_t *B, int ldb,
    plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zgelqt(int m, int n, int ib,
                     plasma_complex64_t *A, int lda,
                     plasma_complex64_t *T, int ldt,
                     plasma_workspace_t work,
                     plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zgemm(
    plasma_enum_t transa, plasma_enum_t transb,
    int m, int n, int k,
    plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
                              const plasma_complex64_t *B, int ldb,
    plasma_complex64_t beta,        plasma_complex64_t *C, int ldc,
    plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zgeqrt(int m, int n, int ib,
                     plasma_complex64_t *A, int lda,
                     plasma_complex64_t *T, int ldt,
                     plasma_workspace_t work,
                     plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zgessq(int m, int n,
                     const plasma_complex64_t *A, int lda,
                     double *scale, double *sumsq,
                     plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zgessq_aux(int n,
                         const double *scale, const double *sumsq,
                         double *value,
                         plasma_sequence_t *sequence,
                         plasma_request_t *request);

void core_omp_zhemm(
    plasma_enum_t side, plasma_enum_t uplo,
    int m, int n,
    plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
                              const plasma_complex64_t *B, int ldb,
    plasma_complex64_t beta,        plasma_complex64_t *C, int ldc,
    plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zher2k(
    plasma_enum_t uplo, plasma_enum_t trans,
    int n, int k,
    plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
                              const plasma_complex64_t *B, int ldb,
    double beta,                    plasma_complex64_t *C, int ldc,
    plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zherk(plasma_enum_t uplo, plasma_enum_t trans,
                    int n, int k,
                    double alpha, const plasma_complex64_t *A, int lda,
                    double beta,        plasma_complex64_t *C, int ldc,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zhessq(plasma_enum_t uplo,
                     int n,
                     const plasma_complex64_t *A, int lda,
                     double *scale, double *sumsq,
                     plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zsyssq(plasma_enum_t uplo,
                     int n,
                     const plasma_complex64_t *A, int lda,
                     double *scale, double *sumsq,
                     plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zsyssq_aux(int m, int n,
                         const double *scale, const double *sumsq,
                         double *value,
                         plasma_sequence_t *sequence,
                         plasma_request_t *request);

void core_omp_zlacpy(plasma_enum_t uplo,
                     int m, int n,
                     const plasma_complex64_t *A, int lda,
                           plasma_complex64_t *B, int ldb,
                     plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zlacpy_lapack2tile_band(plasma_enum_t uplo,
                                      int it, int jt,
                                      int m, int n, int nb, int kl, int ku,
                                      const plasma_complex64_t *A, int lda,
                                            plasma_complex64_t *B, int ldb);

void core_omp_zlacpy_tile2lapack_band(plasma_enum_t uplo,
                                      int it, int jt,
                                      int m, int n, int nb, int kl, int ku,
                                      const plasma_complex64_t *B, int ldb,
                                            plasma_complex64_t *A, int lda);

void core_omp_zlange(plasma_enum_t norm,
                     int m, int n,
                     const plasma_complex64_t *A, int lda,
                     double *work, double *result,
                     plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zlange_aux(plasma_enum_t norm,
                         int m, int n,
                         const plasma_complex64_t *A, int lda,
                         double *value,
                         plasma_sequence_t *sequence,
                         plasma_request_t *request);

void core_omp_zlanhe(plasma_enum_t norm, plasma_enum_t uplo,
                     int n,
                     const plasma_complex64_t *A, int lda,
                     double *work, double *value,
                     plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zlanhe_aux(plasma_enum_t norm, plasma_enum_t uplo,
                         int n,
                         const plasma_complex64_t *A, int lda,
                         double *value,
                         plasma_sequence_t *sequence,
                         plasma_request_t *request);

void core_omp_zlansy(plasma_enum_t norm, plasma_enum_t uplo,
                     int n,
                     const plasma_complex64_t *A, int lda,
                     double *work, double *value,
                     plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zlansy_aux(plasma_enum_t norm, plasma_enum_t uplo,
                         int n,
                         const plasma_complex64_t *A, int lda,
                         double *value,
                         plasma_sequence_t *sequence,
                         plasma_request_t *request);

void core_omp_zlantr(plasma_enum_t norm, plasma_enum_t uplo, plasma_enum_t diag,
                     int m, int n,
                     const plasma_complex64_t *A, int lda,
                     double *work, double *value,
                     plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zlantr_aux(plasma_enum_t norm, plasma_enum_t uplo,
                         plasma_enum_t diag,
                         int m, int n,
                         const plasma_complex64_t *A, int lda,
                         double *value,
                         plasma_sequence_t *sequence,
                         plasma_request_t *request);

void core_omp_zlascl(plasma_enum_t uplo,
                     double cfrom, double cto,
                     int m, int n,
                     plasma_complex64_t *A, int lda,
                     plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zlaset(plasma_enum_t uplo,
                     int mb, int nb,
                     int i, int j,
                     int m, int n,
                     plasma_complex64_t alpha, plasma_complex64_t beta,
                     plasma_complex64_t *A);

void core_omp_zlauum(plasma_enum_t uplo,
                     int n,
                     plasma_complex64_t *A, int lda,
                     plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zpotrf(plasma_enum_t uplo,
                     int n,
                     plasma_complex64_t *A, int lda,
                     int iinfo,
                     plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zsymm(
    plasma_enum_t side, plasma_enum_t uplo,
    int m, int n,
    plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
                              const plasma_complex64_t *B, int ldb,
    plasma_complex64_t beta,        plasma_complex64_t *C, int ldc,
    plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zsyr2k(
    plasma_enum_t uplo, plasma_enum_t trans,
    int n, int k,
    plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
                              const plasma_complex64_t *B, int ldb,
    plasma_complex64_t beta,        plasma_complex64_t *C, int ldc,
    plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zsyrk(
    plasma_enum_t uplo, plasma_enum_t trans,
    int n, int k,
    plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
    plasma_complex64_t beta,        plasma_complex64_t *C, int ldc,
    plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_ztradd(
    plasma_enum_t uplo, plasma_enum_t transa,
    int m, int n,
    plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
    plasma_complex64_t beta,        plasma_complex64_t *B, int ldb,
    plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_ztrmm(
    plasma_enum_t side, plasma_enum_t uplo,
    plasma_enum_t transa, plasma_enum_t diag,
    int m, int n,
    plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
                                    plasma_complex64_t *B, int ldb,
    plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_ztrsm(
    plasma_enum_t side, plasma_enum_t uplo,
    plasma_enum_t transa, plasma_enum_t diag,
    int m, int n,
    plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
                                    plasma_complex64_t *B, int ldb,
    plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_ztrssq(plasma_enum_t uplo, plasma_enum_t diag,
                     int m, int n,
                     const plasma_complex64_t *A, int lda,
                     double *scale, double *sumsq,
                     plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_ztrtri(plasma_enum_t uplo, plasma_enum_t diag,
                     int n,
                     plasma_complex64_t *A, int lda,
                     int iinfo,
                     plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_ztslqt(int m, int n, int ib,
                     plasma_complex64_t *A1, int lda1,
                     plasma_complex64_t *A2, int lda2,
                     plasma_complex64_t *T,  int ldt,
                     plasma_workspace_t work,
                     plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_ztsmlq(plasma_enum_t side, plasma_enum_t trans,
                     int m1, int n1, int m2, int n2, int k, int ib,
                           plasma_complex64_t *A1, int lda1,
                           plasma_complex64_t *A2, int lda2,
                     const plasma_complex64_t *V,  int ldv,
                     const plasma_complex64_t *T,  int ldt,
                     plasma_workspace_t work,
                     plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_ztsmqr(plasma_enum_t side, plasma_enum_t trans,
                     int m1, int n1, int m2, int n2, int k, int ib,
                           plasma_complex64_t *A1, int lda1,
                           plasma_complex64_t *A2, int lda2,
                     const plasma_complex64_t *V, int ldv,
                     const plasma_complex64_t *T, int ldt,
                     plasma_workspace_t work,
                     plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_ztsqrt(int m, int n, int ib,
                     plasma_complex64_t *A1, int lda1,
                     plasma_complex64_t *A2, int lda2,
                     plasma_complex64_t *T,  int ldt,
                     plasma_workspace_t work,
                     plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zttlqt(int m, int n, int ib,
                     plasma_complex64_t *A1, int lda1,
                     plasma_complex64_t *A2, int lda2,
                     plasma_complex64_t *T,  int ldt,
                     plasma_workspace_t work,
                     plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zttmlq(plasma_enum_t side, plasma_enum_t trans,
                     int m1, int n1, int m2, int n2, int k, int ib,
                           plasma_complex64_t *A1, int lda1,
                           plasma_complex64_t *A2, int lda2,
                     const plasma_complex64_t *V,  int ldv,
                     const plasma_complex64_t *T,  int ldt,
                     plasma_workspace_t work,
                     plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zttmqr(plasma_enum_t side, plasma_enum_t trans,
                     int m1, int n1, int m2, int n2, int k, int ib,
                           plasma_complex64_t *A1, int lda1,
                           plasma_complex64_t *A2, int lda2,
                     const plasma_complex64_t *V, int ldv,
                     const plasma_complex64_t *T, int ldt,
                     plasma_workspace_t work,
                     plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zttqrt(int m, int n, int ib,
                     plasma_complex64_t *A1, int lda1,
                     plasma_complex64_t *A2, int lda2,
                     plasma_complex64_t *T,  int ldt,
                     plasma_workspace_t work,
                     plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zunmlq(plasma_enum_t side, plasma_enum_t trans,
                     int m, int n, int k, int ib,
                     const plasma_complex64_t *A, int lda,
                     const plasma_complex64_t *T, int ldt,
                           plasma_complex64_t *C, int ldc,
                     plasma_workspace_t work,
                     plasma_sequence_t *sequence, plasma_request_t *request);

void core_omp_zunmqr(plasma_enum_t side, plasma_enum_t trans,
                     int m, int n, int k, int ib,
                     const plasma_complex64_t *A, int lda,
                     const plasma_complex64_t *T, int ldt,
                           plasma_complex64_t *C, int ldc,
                     plasma_workspace_t work,
                     plasma_sequence_t *sequence, plasma_request_t *request);

#undef COMPLEX

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_CORE_BLAS_Z_H
