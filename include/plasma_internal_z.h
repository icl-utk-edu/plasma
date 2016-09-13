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
#ifndef ICL_PLASMA_INTERNAL_Z_H
#define ICL_PLASMA_INTERNAL_Z_H

#include "plasma_async.h"
#include "plasma_descriptor.h"
#include "plasma_types.h"
#include "plasma_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************/
void plasma_pzgelqf(plasma_desc_t A, plasma_desc_t T,
                    plasma_workspace_t *work,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzgemm(PLASMA_enum transA, PLASMA_enum transB,
                   PLASMA_Complex64_t alpha, plasma_desc_t A,
                                             plasma_desc_t B,
                   PLASMA_Complex64_t beta,  plasma_desc_t C,
                   plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzgeqrf(plasma_desc_t A, plasma_desc_t T,
                    plasma_workspace_t *work,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzhemm(PLASMA_enum side, PLASMA_enum uplo,
                   PLASMA_Complex64_t alpha, plasma_desc_t A,
                                             plasma_desc_t B,
                   PLASMA_Complex64_t beta,  plasma_desc_t C,
                   plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzher2k(PLASMA_enum uplo, PLASMA_enum trans,
                    PLASMA_Complex64_t alpha, plasma_desc_t A,
                    plasma_desc_t B, double beta,  plasma_desc_t C,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzherk(PLASMA_enum uplo, PLASMA_enum trans,
                   double alpha, plasma_desc_t A,
                   double beta,  plasma_desc_t C,
                   plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzlaset(PLASMA_enum uplo,
                    PLASMA_Complex64_t alpha, PLASMA_Complex64_t beta,
                    plasma_desc_t A,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzooccrb2cm(plasma_desc_t A, PLASMA_Complex64_t *Af77, int lda,
                        plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzooccrb2cm_band(PLASMA_enum uplo,
                             plasma_desc_t A, PLASMA_Complex64_t *Af77, int lda,
                             plasma_sequence_t *sequence,
                             plasma_request_t  *request);

void plasma_pzoocm2ccrb(PLASMA_Complex64_t *Af77, int lda, plasma_desc_t A,
                        plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzoocm2ccrb_band(PLASMA_enum uplo,
                             PLASMA_Complex64_t *Af77, int lda, plasma_desc_t A,
                             plasma_sequence_t *sequence,
                             plasma_request_t  *request);

void plasma_pzpbtrf(PLASMA_enum uplo, plasma_desc_t A,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzpotrf(PLASMA_enum uplo, plasma_desc_t A,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzsymm(PLASMA_enum side, PLASMA_enum uplo,
                   PLASMA_Complex64_t alpha, plasma_desc_t A,
                                             plasma_desc_t B,
                   PLASMA_Complex64_t beta,  plasma_desc_t C,
                   plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzsyr2k(PLASMA_enum uplo, PLASMA_enum trans,
                    PLASMA_Complex64_t alpha, plasma_desc_t A,
                    plasma_desc_t B, PLASMA_Complex64_t beta,  plasma_desc_t C,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzsyrk(PLASMA_enum uplo, PLASMA_enum trans,
                   PLASMA_Complex64_t alpha, plasma_desc_t A,
                   PLASMA_Complex64_t beta,  plasma_desc_t C,
                   plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pztbsm(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans,
                   PLASMA_enum diag, PLASMA_Complex64_t alpha,
                   plasma_desc_t A, plasma_desc_t B, const int *IPIV,
                   plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pztradd(PLASMA_enum uplo, PLASMA_enum transA,
                    PLASMA_Complex64_t alpha,  plasma_desc_t A,
                    PLASMA_Complex64_t beta,   plasma_desc_t B,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pztrmm(PLASMA_enum side, PLASMA_enum uplo,
                   PLASMA_enum trans, PLASMA_enum diag,
                   PLASMA_Complex64_t alpha, plasma_desc_t A, plasma_desc_t B,
                   plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pztrsm(PLASMA_enum side, PLASMA_enum uplo,
                   PLASMA_enum trans, PLASMA_enum diag,
                   PLASMA_Complex64_t alpha, plasma_desc_t A,
                                             plasma_desc_t B,
                   plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzunglq(plasma_desc_t A, plasma_desc_t Q, plasma_desc_t T,
                    plasma_workspace_t *work,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzungqr(plasma_desc_t A, plasma_desc_t Q, plasma_desc_t T,
                    plasma_workspace_t *work,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzunmlq(PLASMA_enum side, PLASMA_enum trans,
                    plasma_desc_t A, plasma_desc_t B, plasma_desc_t T,
                    plasma_workspace_t *work,
                    plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_pzunmqr(PLASMA_enum side, PLASMA_enum trans,
                    plasma_desc_t A, plasma_desc_t B, plasma_desc_t T,
                    plasma_workspace_t *work,
                    plasma_sequence_t *sequence, plasma_request_t *request);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_PLASMA_INTERNAL_Z_H
