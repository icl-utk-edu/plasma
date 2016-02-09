/**
 *
 * @file test.h
 *
 *  PLASMA test routines.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver.
 *
 * @version 3.0.0
 * @author Jakub Kurzak
 * @date 2016-01-01
 * @precisions normal z -> s d c
 *
 **/
#ifndef TEST_Z_H
#define TEST_Z_H

/* testing routines */
void test_zgemm(param_value_t param[]);
void test_zsymm(param_value_t param[]);

#endif // TEST_Z_H
