/**
 *
 * @file test.c
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
#include "test.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>

/******************************************************************************/
int main(int argc, char **argv)
{
    if (argc == 1 ||
        strcmp(argv[1], "-h") == 0 ||
        strcmp(argv[1], "--help") == 0) {

        print_main_usage();
        return EXIT_FAILURE;
    }

    if (argc == 2 ||
        strcmp(argv[2], "-h") == 0 ||
        strcmp(argv[2], "--help") == 0) {

        print_routine_usage(argv[1]);
        return EXIT_FAILURE;
    }

    param_t param[PARAM_SIZEOF];       /* set of parameters */
    param_value_t value[PARAM_SIZEOF]; /* snapshot of values */
    param_init(param);
    param_read(argc, argv, param);

    if (param[PARAM_OUTER].val[0].c == 'y') {
        // outer product iteration
        do {
            param_snap(param, value);
            test_routine(argv[1], value);
        }
        while (param_step_outer(param, 0));
    }
    else {
        // inner product iteration
        do {
            param_snap(param, value);
            test_routine(argv[1], value);
        }
        while (param_step_inner(param));
    }
    return EXIT_SUCCESS;
}

/******************************************************************************/
void print_main_usage()
{
    printf("Usage:\n"
           "\ttest [-h|--help]\n"
           "\ttest routine [-h|--help]\n"
           "\ttest routine [parameter1, parameter2, ...]\n");
}

/******************************************************************************/
void print_routine_usage(char *name)
{
    printf("Usage:\n"
           "\ttest %s [-h|--help]\n"
           "\ttest %s (parameter1, parameter2, ...)\n\n"
           "Options:\n"
           "\t%*sshow this screen\n",
           name, name,
           DescriptionIndent, "-h --help");
    print_usage(PARAM_OUTER);
    print_usage(PARAM_TEST);
    print_usage(PARAM_TOL);
    printf("\n");

    if (strcmp(name, "zgemm") == 0)
        test_zgemm(NULL);
    else if (strcmp(name, "dgemm") == 0)
        test_dgemm(NULL);
    else if (strcmp(name, "cgemm") == 0)
        test_cgemm(NULL);
    else if (strcmp(name, "sgemm") == 0)
        test_sgemm(NULL);
    else if (strcmp(name, "zsymm") == 0)
        test_zsymm(NULL);
    else if (strcmp(name, "dsymm") == 0)
        test_dsymm(NULL);
    else if (strcmp(name, "csymm") == 0)
        test_csymm(NULL);
    else if (strcmp(name, "ssymm") == 0)
        test_ssymm(NULL);
}

/******************************************************************************/
void print_usage(int label)
{
    printf("\t%*s%s\n",
        DescriptionIndent,
        ParamUsage[label][0],
        ParamUsage[label][1]);
}

/******************************************************************************/
void test_routine(char *name, param_value_t value[])
{
    if (strcmp(name, "zgemm") == 0)
        test_zgemm(value);
    else if (strcmp(name, "dgemm") == 0)
        test_dgemm(value);
    else if (strcmp(name, "cgemm") == 0)
        test_cgemm(value);
    else if (strcmp(name, "sgemm") == 0)
        test_sgemm(value);
    else if (strcmp(name, "zsymm") == 0)
        test_zsymm(value);
    else if (strcmp(name, "dsymm") == 0)
        test_dsymm(value);
    else if (strcmp(name, "csymm") == 0)
        test_csymm(value);
    else if (strcmp(name, "ssymm") == 0)
        test_ssymm(value);
}

/******************************************************************************/
void param_init(param_t param[])
{
    for (int i = 0; i < PARAM_SIZEOF; i++) {
        param[i].num = 0;
        param[i].pos = 0;
        param[i].val =
            (param_value_t*)malloc(InitValArraySize*sizeof(param_value_t));
        assert(param[i].val != NULL);
        param[i].size = InitValArraySize;
    }
}

/******************************************************************************/
void param_read(int argc, char **argv, param_t param[])
{
    for (int i = 1; i < argc && argv[i]; i++) {

        /* Scan character parameters. */
        if (param_starts_with(argv[i], "--outer="))
            param_scan_char(strchr(argv[i], '=')+1, &param[PARAM_OUTER]);
        else if (param_starts_with(argv[i], "--test="))
            param_scan_char(strchr(argv[i], '=')+1, &param[PARAM_TEST]);

        else if (param_starts_with(argv[i], "--transa="))
            param_scan_char(strchr(argv[i], '=')+1, &param[PARAM_TRANSA]);
        else if (param_starts_with(argv[i], "--transb="))
            param_scan_char(strchr(argv[i], '=')+1, &param[PARAM_TRANSB]);

        /* Scan integer parameters. */
        else if (param_starts_with(argv[i], "--m="))
            param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_M]);
        else if (param_starts_with(argv[i], "--n="))
            param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_N]);
        else if (param_starts_with(argv[i], "--k="))
            param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_K]);

        else if (param_starts_with(argv[i], "--lda="))
            param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_LDA]);
        else if (param_starts_with(argv[i], "--ldb="))
            param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_LDB]);
        else if (param_starts_with(argv[i], "--ldc="))
            param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_LDC]);

        else if (param_starts_with(argv[i], "--pada="))
            param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_PADA]);
        else if (param_starts_with(argv[i], "--padb="))
            param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_PADB]);
        else if (param_starts_with(argv[i], "--padc="))
            param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_PADC]);

        /* Scan double precision parameters. */
        else if (param_starts_with(argv[i], "--tol="))
            param_scan_double(strchr(argv[i], '=')+1, &param[PARAM_TOL]);

    }
}

/******************************************************************************/
int param_starts_with(const char *str, const char *prefix)
{
    size_t n = strlen(prefix);
    if (strncmp(str, prefix, n))
        return 0;
    return 1;
}

/******************************************************************************/
void param_scan_int(char *str, param_t *param)
{
    char *endptr;
    do {
        long start = strtol(str, &endptr, 10);
        if (*endptr == ':') {
            long stop = strtol(endptr+1, &endptr, 10);
            long step = strtol(endptr+1, &endptr, 10);
            for (int i = start; i <= stop; i += step)
                param_add_int(i, param);
        }
        else {
            param_add_int(start, param);
        }
        str = endptr+1;
    }
    while (*endptr != '\0');
}

/******************************************************************************/
void param_scan_char(char *str, param_t *param)
{
    char *endptr;
    do {
        param_add_char(*str, param);
        endptr = str+1;
        str = endptr+1;
    }
    while (*endptr != '\0');
}

/******************************************************************************/
void param_scan_double(char *str, param_t *param)
{
    char *endptr;
    do {
        double start = strtod(str, &endptr);
        if (*endptr == ':') {
            double stop = strtod(endptr+1, &endptr);
            double step = strtod(endptr+1, &endptr);
            for (double d = start; d <= stop; d += step)
                param_add_double(d, param);
        }
        else {
            param_add_double(start, param);
        }
        str = endptr+1;
    }
    while (*endptr != '\0');
}

/******************************************************************************/
void param_add_int(int ival, param_t *param)
{
    param->val[param->num].i = ival;
    param->num++;
    if (param->num == param->size) {
        param->size *= 2;
        param->val = realloc(param->val, param->size*sizeof(param_value_t));
        assert(param->val != NULL);
    }
}

/******************************************************************************/
void param_add_char(char cval, param_t *param)
{
    param->val[param->num].c = cval;
    param->num++;
    if (param->num == param->size) {
        param->size *= 2;
        param->val = realloc(param->val, param->size*sizeof(param_value_t));
        assert(param->val != NULL);
    }
}

/******************************************************************************/
void param_add_double(double dval, param_t *param)
{
    param->val[param->num].d = dval;
    param->num++;
    if (param->num == param->size) {
        param->size *= 2;
        param->val = realloc(param->val, param->size*sizeof(param_value_t));
        assert(param->val != NULL);
    }
}

/******************************************************************************/
int param_step_inner(param_t param[])
{
    int finished = 1;
    for (int i = 0; i < PARAM_SIZEOF; i++) {
        if (param[i].pos < param[i].num-1) {
            param[i].pos++;
            finished = 0;
        }
    }
    return ! finished;
}

/******************************************************************************/
int param_step_outer(param_t param[], int idx)
{
    while (param[idx].num == 0)
        if (++idx == PARAM_SIZEOF)
            return 0;

    if (++param[idx].pos == param[idx].num) {
        param[idx].pos = 0;
        return param_step_outer(param, idx+1);
    }
    return 1;
}

/******************************************************************************/
int param_snap(param_t param[], param_value_t value[])
{
    for (int i = 0; i < PARAM_SIZEOF; i++)
        value[i] = param[i].val[param[i].pos];
    return 0;
}
