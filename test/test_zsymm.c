/**
 *
 * @file test_zsymm.c
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

void test_zsymm(param_value_t param[])
{
    if (param == NULL) {
        print_usage(PARAM_SIDE);
        print_usage(PARAM_UPLO);
        print_usage(PARAM_M);
        print_usage(PARAM_N);
        print_usage(PARAM_LDA);
        print_usage(PARAM_LDB);
        print_usage(PARAM_LDC);
        return;
    }



}
