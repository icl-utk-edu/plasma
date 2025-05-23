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
#ifndef TEST_Z_H
#define TEST_Z_H

#include "test.h"

//==============================================================================
// test routines
//==============================================================================
void test_dzamax(param_value_t param[], bool run);
void test_zgbmm(param_value_t param[], bool run);
void test_zgbsv(param_value_t param[], bool run);
void test_zgbtrf(param_value_t param[], bool run);
void test_zgeadd(param_value_t param[], bool run);
void test_zgeinv(param_value_t param[], bool run);
void test_zgelqf(param_value_t param[], bool run);
void test_zgelqs(param_value_t param[], bool run);
void test_zgels(param_value_t param[], bool run);
void test_zgemm(param_value_t param[], bool run);
void test_zgeqrf(param_value_t param[], bool run);
void test_zgeqrs(param_value_t param[], bool run);
void test_zgesdd(param_value_t param[], bool run);
void test_zgesv(param_value_t param[], bool run);
void test_zgetrf(param_value_t param[], bool run);
void test_zgetri(param_value_t param[], bool run);
void test_zgetri_aux(param_value_t param[], bool run);
void test_zgetrs(param_value_t param[], bool run);
void test_zhbtrd(param_value_t param[], bool run);
void test_zheevd(param_value_t param[], bool run);
void test_zhemm(param_value_t param[], bool run);
void test_zher2k(param_value_t param[], bool run);
void test_zherk(param_value_t param[], bool run);
void test_zhetrf(param_value_t param[], bool run);
void test_zhesv(param_value_t param[], bool run);
void test_zlacpy(param_value_t param[], bool run);
void test_zlag2c(param_value_t param[], bool run);
void test_zlange(param_value_t param[], bool run);
void test_zlangb(param_value_t param[], bool run);
void test_zlanhe(param_value_t param[], bool run);
void test_zlansy(param_value_t param[], bool run);
void test_zlantr(param_value_t param[], bool run);
void test_zlascl(param_value_t param[], bool run);
void test_zlaset(param_value_t param[], bool run);
void test_zgeswp(param_value_t param[], bool run);
void test_zlauum(param_value_t param[], bool run);
void test_zpbsv(param_value_t param[], bool run);
void test_zpbtrf(param_value_t param[], bool run);
void test_zpoinv(param_value_t param[], bool run);
void test_zposv(param_value_t param[], bool run);
void test_zpotrf(param_value_t param[], bool run);
void test_zpotri(param_value_t param[], bool run);
void test_zpotrs(param_value_t param[], bool run);
void test_zsymm(param_value_t param[], bool run);
void test_zstevx2(param_value_t param[], bool run);
void test_zsyr2k(param_value_t param[], bool run);
void test_zsyrk(param_value_t param[], bool run);
void test_ztradd(param_value_t param[], bool run);
void test_ztrmm(param_value_t param[], bool run);
void test_ztrsm(param_value_t param[], bool run);
void test_ztrtri(param_value_t param[], bool run);
void test_zunmlq(param_value_t param[], bool run);
void test_zunmqr(param_value_t param[], bool run);

//==============================================================================
// utilities
//==============================================================================
void plasma_zprint_matrix(
    const char* label, int m, int n, plasma_complex64_t* A, int lda );

void plasma_zprint_vector(
    const char* label, int n, plasma_complex64_t* x, int incx );

#endif // TEST_Z_H
