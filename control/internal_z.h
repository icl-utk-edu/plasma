/**
 *
 * @file internal.h
 *
 *  PLASMA control routines.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver.
 *
 * @version 3.0.0
 * @author Jakub Kurzak
 * @date 2016-01-01
 * @precisions normal z -> s d c
 *
 **/
#ifndef INTERNAL_Z_H
#define INTERNAL_Z_H

#include "../control/async.h"
#include "../control/descriptor.h"
#include "../include/plasmatypes.h"

/******************************************************************************/
void plasma_pzgemm(PLASMA_enum transA, PLASMA_enum transB,
                   PLASMA_Complex64_t alpha, PLASMA_desc A,
                                             PLASMA_desc B,
                   PLASMA_Complex64_t beta,  PLASMA_desc C,
                   PLASMA_sequence *sequence, PLASMA_request *request);

void plasma_pzooccrb2cm(PLASMA_desc A, PLASMA_Complex64_t *Af77, int lda,
                        PLASMA_sequence *sequence, PLASMA_request *request);

void plasma_pzoocm2ccrb(PLASMA_Complex64_t *Af77, int lda, PLASMA_desc A,
                        PLASMA_sequence *sequence, PLASMA_request *request);

#endif // INTERNAL_Z_H
