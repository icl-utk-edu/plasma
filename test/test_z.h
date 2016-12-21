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
void test_dzamax(param_value_t param[], char *info);
void test_zgbsv(param_value_t param[], char *info);
void test_zgbtrf(param_value_t param[], char *info);
void test_zgeadd(param_value_t param[], char *info);
void test_zgelqf(param_value_t param[], char *info);
void test_zgelqs(param_value_t param[], char *info);
void test_zgels(param_value_t param[], char *info);
void test_zgemm(param_value_t param[], char *info);
void test_zgeqrf(param_value_t param[], char *info);
void test_zgeqrs(param_value_t param[], char *info);
void test_zgesv(param_value_t param[], char *info);
void test_zgetrf(param_value_t param[], char *info);
void test_zgetri(param_value_t param[], char *info);
void test_zgetri_aux(param_value_t param[], char *info);
void test_zgetrs(param_value_t param[], char *info);
void test_zhemm(param_value_t param[], char *info);
void test_zher2k(param_value_t param[], char *info);
void test_zherk(param_value_t param[], char *info);
void test_zlacpy(param_value_t param[], char *info);
void test_zlag2c(param_value_t param[], char *info);
void test_zlange(param_value_t param[], char *info);
void test_zlanhe(param_value_t param[], char *info);
void test_zlansy(param_value_t param[], char *info);
void test_zlantr(param_value_t param[], char *info);
void test_zlascl(param_value_t param[], char *info);
void test_zlaset(param_value_t param[], char *info);
void test_zlaswp(param_value_t param[], char *info);
void test_zlauum(param_value_t param[], char *info);
void test_zpbsv(param_value_t param[], char *info);
void test_zpbtrf(param_value_t param[], char *info);
void test_zposv(param_value_t param[], char *info);
void test_zpotrf(param_value_t param[], char *info);
void test_zpotri(param_value_t param[], char *info);
void test_zpotrs(param_value_t param[], char *info);
void test_zsymm(param_value_t param[], char *info);
void test_zsyr2k(param_value_t param[], char *info);
void test_zsyrk(param_value_t param[], char *info);
void test_ztradd(param_value_t param[], char *info);
void test_ztrmm(param_value_t param[], char *info);
void test_ztrsm(param_value_t param[], char *info);
void test_ztrtri(param_value_t param[], char *info);
void test_zunmlq(param_value_t param[], char *info);
void test_zunmqr(param_value_t param[], char *info);

#endif // TEST_Z_H
