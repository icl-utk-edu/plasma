/**
 *
 * @file test_zgemm.c
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
#include "test.h"
#include <stddef.h>
#include <stdio.h>

void test_zgemm(param_value_t param[])
{
    if (param == NULL) {
        print_usage(PARAM_TRANSA);
        print_usage(PARAM_TRANSB);
        print_usage(PARAM_M);
        print_usage(PARAM_N);
        print_usage(PARAM_K);
        print_usage(PARAM_LDA);
        print_usage(PARAM_LDB);
        print_usage(PARAM_LDC);
        return;
    }
    printf("\ttesting zgemm...\n");



}
