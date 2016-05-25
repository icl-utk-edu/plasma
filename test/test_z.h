/**
 *
 * @file test_z.h
 *
 *  PLASMA test routines.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of Manchester, Univ. of California Berkeley and
 *  Univ. of Colorado Denver.
 *
 * @version 3.0.0
 * @author Jakub Kurzak
 * @author Samuel D. Relton
 * @date 2016-05-17
 * @precisions normal z -> s d c
 *
 **/
#ifndef TEST_Z_H
#define TEST_Z_H

//==============================================================================
// test routines
//==============================================================================
void test_zgemm(param_value_t param[], char *info);
void test_zsymm(param_value_t param[], char *info);
void test_zsyrk(param_value_t param[], char *info);

#define COMPLEX
void test_zhemm(param_value_t param[], char *info);
void test_zherk(param_value_t param[], char *info);
#undef COMPLEX

#endif // TEST_Z_H
