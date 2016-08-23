/**
 *
 * @file plasma_internal_z.h
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

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************/
void plasma_pzgemm(PLASMA_enum transA, PLASMA_enum transB,
                   PLASMA_Complex64_t alpha, PLASMA_desc A,
                                             PLASMA_desc B,
                   PLASMA_Complex64_t beta,  PLASMA_desc C,
                   PLASMA_sequence *sequence, PLASMA_request *request);

void plasma_pzgeqrf(PLASMA_desc A, PLASMA_desc T,
                    PLASMA_sequence *sequence, PLASMA_request *request);

void plasma_pzhemm(PLASMA_enum side, PLASMA_enum uplo,
                   PLASMA_Complex64_t alpha, PLASMA_desc A,
                                             PLASMA_desc B,
                   PLASMA_Complex64_t beta,  PLASMA_desc C,
                   PLASMA_sequence *sequence, PLASMA_request *request);

void plasma_pzher2k(PLASMA_enum uplo, PLASMA_enum trans,
                    PLASMA_Complex64_t alpha, PLASMA_desc A,
                    PLASMA_desc B, double beta,  PLASMA_desc C,
                    PLASMA_sequence *sequence, PLASMA_request *request);

void plasma_pzherk(PLASMA_enum uplo, PLASMA_enum trans,
                   double alpha, PLASMA_desc A,
                   double beta,  PLASMA_desc C,
                   PLASMA_sequence *sequence, PLASMA_request *request);

void plasma_pzlaset(PLASMA_enum uplo,
                    PLASMA_Complex64_t alpha, PLASMA_Complex64_t beta,
                    PLASMA_desc A,
                    PLASMA_sequence *sequence, PLASMA_request *request);

void plasma_pzooccrb2cm(PLASMA_desc A, PLASMA_Complex64_t *Af77, int lda,
                        PLASMA_sequence *sequence, PLASMA_request *request);

void plasma_pzoocm2ccrb(PLASMA_Complex64_t *Af77, int lda, PLASMA_desc A,
                        PLASMA_sequence *sequence, PLASMA_request *request);

void plasma_pzpotrf(PLASMA_enum uplo, PLASMA_desc A,
                    PLASMA_sequence *sequence, PLASMA_request *request);

void plasma_pzsymm(PLASMA_enum side, PLASMA_enum uplo,
                   PLASMA_Complex64_t alpha, PLASMA_desc A,
                                             PLASMA_desc B,
                   PLASMA_Complex64_t beta,  PLASMA_desc C,
                   PLASMA_sequence *sequence, PLASMA_request *request);

void plasma_pzsyr2k(PLASMA_enum uplo, PLASMA_enum trans,
                    PLASMA_Complex64_t alpha, PLASMA_desc A,
                    PLASMA_desc B, PLASMA_Complex64_t beta,  PLASMA_desc C,
                    PLASMA_sequence *sequence, PLASMA_request *request);

void plasma_pzsyrk(PLASMA_enum uplo, PLASMA_enum trans,
                   PLASMA_Complex64_t alpha, PLASMA_desc A,
                   PLASMA_Complex64_t beta,  PLASMA_desc C,
                   PLASMA_sequence *sequence, PLASMA_request *request);

void plasma_pztrsm(PLASMA_enum side, PLASMA_enum uplo,
                   PLASMA_enum trans, PLASMA_enum diag,
                   PLASMA_Complex64_t alpha, PLASMA_desc A,
                                             PLASMA_desc B,
                   PLASMA_sequence *sequence, PLASMA_request *request);

void plasma_pzungqr(PLASMA_desc A, PLASMA_desc Q, PLASMA_desc T,
                    PLASMA_sequence *sequence, PLASMA_request *request);

void plasma_pzunmqr(PLASMA_enum side, PLASMA_enum trans,
                    PLASMA_desc A, PLASMA_desc B, PLASMA_desc T,
                    PLASMA_sequence *sequence, PLASMA_request *request);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_PLASMA_INTERNAL_Z_H
