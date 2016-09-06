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
void test_zgelqf(param_value_t param[], char *info);
void test_zgelqs(param_value_t param[], char *info);
void test_zgels(param_value_t param[], char *info);
void test_zgemm(param_value_t param[], char *info);
void test_zgeqrf(param_value_t param[], char *info);
void test_zgeqrs(param_value_t param[], char *info);
void test_zhemm(param_value_t param[], char *info);
void test_zher2k(param_value_t param[], char *info);
void test_zherk(param_value_t param[], char *info);
void test_zpbsv(param_value_t param[], char *info);
void test_zpbtrf(param_value_t param[], char *info);
void test_zposv(param_value_t param[], char *info);
void test_zpotrf(param_value_t param[], char *info);
void test_zpotrs(param_value_t param[], char *info);
void test_zsymm(param_value_t param[], char *info);
void test_zsyr2k(param_value_t param[], char *info);
void test_zsyrk(param_value_t param[], char *info);
void test_ztrmm(param_value_t param[], char *info);
void test_ztrsm(param_value_t param[], char *info);

#endif // TEST_Z_H
