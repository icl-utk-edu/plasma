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
#ifndef PLASMA_INTERNAL_Z_H
#define PLASMA_INTERNAL_Z_H

#include "plasma_async.h"
#include "plasma_descriptor.h"
#include "plasma_types.h"
#include "plasma_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************/
void plasma_pdzamax(plasma_enum_t colrow,
                    plasma_desc_t A, double *work, double *values,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzgbtrf(plasma_desc_t A, int *ipiv,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzdesc2ge(plasma_desc_t A,
                      plasma_complex64_t *pA, int lda,
                      plasma_sequence_t *sequence,
                      plasma_request_t *request);

void plasma_pzdesc2pb(plasma_desc_t A,
                      plasma_complex64_t *pA, int lda,
                      plasma_sequence_t *sequence,
                      plasma_request_t *request);

void plasma_pzdesc2tr(plasma_desc_t A,
                      plasma_complex64_t *pA, int lda,
                      plasma_sequence_t *sequence,
                      plasma_request_t *request);

void plasma_pzge2desc(plasma_complex64_t *pA, int lda,
                      plasma_desc_t A,
                      plasma_sequence_t *sequence,
                      plasma_request_t *request);

void plasma_pzgeadd(plasma_enum_t transa,
                    plasma_complex64_t alpha,  plasma_desc_t A,
                    plasma_complex64_t beta,   plasma_desc_t B,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzgelqf(plasma_desc_t A, plasma_desc_t T,
                    plasma_workspace_t work,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzgelqf_tree(plasma_desc_t A, plasma_desc_t T,
                         plasma_workspace_t work,
                         plasma_sequence_t *sequence,
                         plasma_request_t *request);

void plasma_pzgemm(plasma_enum_t transa, plasma_enum_t transb,
                   plasma_complex64_t alpha, plasma_desc_t A,
                                             plasma_desc_t B,
                   plasma_complex64_t beta,  plasma_desc_t C,
                   plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzgeqrf(plasma_desc_t A, plasma_desc_t T,
                    plasma_workspace_t work,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzgeqrf_tree(plasma_desc_t A, plasma_desc_t T,
                         plasma_workspace_t work,
                         plasma_sequence_t *sequence,
                         plasma_request_t *request);

void plasma_pzgetri_aux(plasma_desc_t A, plasma_desc_t W,
                        plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzgetrf(plasma_desc_t A, int *ipiv,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzhemm(plasma_enum_t side, plasma_enum_t uplo,
                   plasma_complex64_t alpha, plasma_desc_t A,
                                             plasma_desc_t B,
                   plasma_complex64_t beta,  plasma_desc_t C,
                   plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzher2k(plasma_enum_t uplo, plasma_enum_t trans,
                    plasma_complex64_t alpha, plasma_desc_t A,
                                              plasma_desc_t B,
                    double beta,              plasma_desc_t C,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzherk(plasma_enum_t uplo, plasma_enum_t trans,
                   double alpha, plasma_desc_t A,
                   double beta,  plasma_desc_t C,
                   plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzhetrf_aasen(plasma_enum_t uplo,
                          plasma_desc_t A, int *ipiv,
                          plasma_desc_t T,
                          plasma_desc_t W,
                          plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzlacpy(plasma_enum_t uplo, plasma_enum_t transa,
                    plasma_desc_t A, plasma_desc_t B,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzlangb(plasma_enum_t norm,
                    plasma_desc_t A, double *work, double *value,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzlange(plasma_enum_t norm,
                    plasma_desc_t A, double *work, double *value,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzlanhe(plasma_enum_t norm, plasma_enum_t uplo,
                    plasma_desc_t A, double *work, double *value,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzlansy(plasma_enum_t norm, plasma_enum_t uplo,
                    plasma_desc_t A, double *work, double *value,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzlantr(plasma_enum_t norm, plasma_enum_t uplo, plasma_enum_t diag,
                    plasma_desc_t A, double *work, double *value,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzlangb(plasma_enum_t norm,
                    plasma_desc_t A, double *work, double *value,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzlascl(plasma_enum_t uplo,
                    double cfrom, double cto,
                    plasma_desc_t A,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzlaset(plasma_enum_t uplo,
                    plasma_complex64_t alpha, plasma_complex64_t beta,
                    plasma_desc_t A,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzgeswp(plasma_enum_t colrow,
                    plasma_desc_t A, int *ipiv, int incx,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzlauum(plasma_enum_t uplo, plasma_desc_t A,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzpb2desc(plasma_complex64_t *pA, int lda,
                      plasma_desc_t A,
                      plasma_sequence_t *sequence,
                      plasma_request_t *request);

void plasma_pzpbtrf(plasma_enum_t uplo, plasma_desc_t A,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzpotrf(plasma_enum_t uplo, plasma_desc_t A,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzsymm(plasma_enum_t side, plasma_enum_t uplo,
                   plasma_complex64_t alpha, plasma_desc_t A,
                                             plasma_desc_t B,
                   plasma_complex64_t beta,  plasma_desc_t C,
                   plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzsyr2k(plasma_enum_t uplo, plasma_enum_t trans,
                    plasma_complex64_t alpha, plasma_desc_t A,
                                              plasma_desc_t B,
                    plasma_complex64_t beta,  plasma_desc_t C,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzsyrk(plasma_enum_t uplo, plasma_enum_t trans,
                   plasma_complex64_t alpha, plasma_desc_t A,
                   plasma_complex64_t beta,  plasma_desc_t C,
                   plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pztbsm(plasma_enum_t side, plasma_enum_t uplo,
                   plasma_enum_t trans, plasma_enum_t diag,
                   plasma_complex64_t alpha, plasma_desc_t A,
                                             plasma_desc_t B,
                   const int *ipiv,
                   plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pztr2desc(plasma_complex64_t *pA, int lda,
                      plasma_desc_t A,
                      plasma_sequence_t *sequence,
                      plasma_request_t *request);

void plasma_pztradd(plasma_enum_t uplo, plasma_enum_t transa,
                    plasma_complex64_t alpha,  plasma_desc_t A,
                    plasma_complex64_t beta,   plasma_desc_t B,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pztrmm(plasma_enum_t side, plasma_enum_t uplo,
                   plasma_enum_t trans, plasma_enum_t diag,
                   plasma_complex64_t alpha, plasma_desc_t A,
                                             plasma_desc_t B,
                   plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pztrsm(plasma_enum_t side, plasma_enum_t uplo,
                   plasma_enum_t trans, plasma_enum_t diag,
                   plasma_complex64_t alpha, plasma_desc_t A,
                                             plasma_desc_t B,
                   plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pztrtri(plasma_enum_t uplo, plasma_enum_t diag,
                    plasma_desc_t A,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzunglq(plasma_desc_t A, plasma_desc_t T, plasma_desc_t Q,
                    plasma_workspace_t work,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzunglq_tree(plasma_desc_t A, plasma_desc_t T, plasma_desc_t Q,
                         plasma_workspace_t work,
                         plasma_sequence_t *sequence,
                         plasma_request_t *request);

void plasma_pzungqr(plasma_desc_t A, plasma_desc_t T, plasma_desc_t Q,
                    plasma_workspace_t work,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzungqr_tree(plasma_desc_t A, plasma_desc_t T, plasma_desc_t Q,
                         plasma_workspace_t work,
                         plasma_sequence_t *sequence,
                         plasma_request_t *request);

void plasma_pzunmlq(plasma_enum_t side, plasma_enum_t trans,
                    plasma_desc_t A, plasma_desc_t T, plasma_desc_t B,
                    plasma_workspace_t work,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzunmlq_tree(plasma_enum_t side, plasma_enum_t trans,
                         plasma_desc_t A, plasma_desc_t T, plasma_desc_t B,
                         plasma_workspace_t work,
                         plasma_sequence_t *sequence,
                         plasma_request_t *request);

void plasma_pzunmqr(plasma_enum_t side, plasma_enum_t trans,
                    plasma_desc_t A, plasma_desc_t T, plasma_desc_t B,
                    plasma_workspace_t work,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzunmqr_tree(plasma_enum_t side, plasma_enum_t trans,
                         plasma_desc_t A, plasma_desc_t T, plasma_desc_t B,
                         plasma_workspace_t work,
                         plasma_sequence_t *sequence,
                         plasma_request_t *request);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // PLASMA_INTERNAL_Z_H
