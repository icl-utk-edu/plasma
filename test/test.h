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
 *
 **/
#ifndef TEST_H
#define TEST_H

/* initial size of values array */
static const int InitValArraySize = 1024;

/* indentation of option description */
static const int DescriptionIndent = -20;

/* parameter labels */
typedef enum {
    PARAM_OUTER,  /* outer product iteration? */
    PARAM_TEST,   /* test the solution? */
    PARAM_TOL,    /* tolerance */
    PARAM_TRANSA, /* transposition of A */
    PARAM_TRANSB, /* transposition of B */
    PARAM_SIDE,   /* left/right side */
    PARAM_UPLO,   /* lower/upper (triangular) matrix */
    PARAM_M,      /* M dimension */
    PARAM_N,      /* N dimension */
    PARAM_K,      /* K dimension */
    PARAM_LDA,    /* leading dimension of A */
    PARAM_LDB,    /* leading dimension of B */
    PARAM_LDC,    /* leading dimension of C */
    PARAM_PADA,   /* padding of A */
    PARAM_PADB,   /* padding of B */
    PARAM_PADC,   /* padding of C */
    PARAM_SIZEOF  /* size of parameter array */
} param_label_t;

/* parameter descriptions */
static const char *ParamUsage[][2] = {
    {"--outer=[y|n]", "outer product iteration [default: n]"},
    {"--test=[y|n]", "test the solution [default: y]"},
    {"--tol=", "tolerance"},
    {"--transa=[n|t|c]", "transposition of A"},
    {"--transb=[n|t|c]", "transposition of B"},
    {"--side=[l|r]", "left/right side (operation)"},
    {"--uplo=[l|u]", "lower/upper (triangular) matrix"},
    {"--m=", "M dimension (number of rows)"},
    {"--n=", "N dimension (number of columns)"},
    {"--k=", "K dimension (number of rows or columns)"},
    {"--lda=", "leading dimension of A"},
    {"--ldb=", "leading dimension of B"},
    {"--ldc=", "leading dimension of C"},
    {"--pada=", "padding of A"},
    {"--padb=", "padding of B"},
    {"--padc=", "padding of C"}
};

/* parameter value */
typedef union {
    int i;          /* integer */
    char c;         /* character */
    double d;       /* double precision */
} param_value_t;

/* parameter */
typedef struct {
    int num;            /* number of values for a parameter */
    int pos;            /* current position in the array */
    int size;           /* size of parameter values array */
    param_value_t *val; /* array of values for a parameter */
} param_t;

/* infrastructure routines */
void print_main_usage();
void print_routine_usage(char *name);
void print_usage(int label);
void test_routine(char *name, param_value_t param[]);
void param_init(param_t param[]);
void param_read(int argc, char **argv, param_t param[]);
int param_starts_with(const char *str, const char *prefix);
void param_scan_int(char *str, param_t *param);
void param_scan_char(char *str, param_t *param);
void param_scan_double(char *str, param_t *param);
void param_add_int(int val, param_t *param);
void param_add_char(char cval, param_t *param);
void param_add_double(double dval, param_t *param);
int param_step_inner(param_t param[]);
int param_step_outer(param_t param[], int idx);
int param_snap(param_t param[], param_value_t value[]);

#include "test_s.h"
#include "test_d.h"
#include "test_c.h"
#include "test_z.h"

#endif // TEST_H
