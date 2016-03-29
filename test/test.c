/**
 *
 * @file test.c
 *
 *  PLASMA testing harness.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver.
 *
 * @version 3.0.0
 * @author Jakub Kurzak
 * @date 2016-01-01
 *
 **/
#include "test.h"
#include "../control/context.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/***************************************************************************//**
 *
 * @brief Tests and times a PLASMA routine.
 *        Prints usage information when ran without options.
 *
 * @param[in] argc
 * @param[in] argv
 *
 * @retval EXIT_SUCCESS - correct invocation
 * @retval EXIT_FAILURE - incorrect invocation
 *
 ******************************************************************************/
int main(int argc, char **argv)
{
    if (argc == 1 ||
        strcmp(argv[1], "-h") == 0 ||
        strcmp(argv[1], "--help") == 0) {

        print_main_usage();
        return EXIT_FAILURE;
    }

    if (argc == 3 &&
        (strcmp(argv[2], "-h") == 0 ||
         strcmp(argv[2], "--help") == 0)) {

        print_routine_usage(argv[1]);
        return EXIT_FAILURE;
    }

    param_t param[PARAM_SIZEOF];      // set of parameters
    param_value_t pval[PARAM_SIZEOF]; // snapshot of values

    param_init(param);
    int iter = param_read(argc, argv, param);
    int outer = param[PARAM_OUTER].val[0].c == 'y';
    int test = param[PARAM_TEST].val[0].c == 'y';

    // Print labels.
    if (test)
        test_routine(argv[1], NULL);
    else
        time_routine(argv[1], NULL);

    PLASMA_Init();
    if (outer) {
        // outer product iteration
        do {
            param_snap(param, pval);
            for (int i = 0; i < iter; i++)
                if (test)
                    test_routine(argv[1], pval);
                else
                    time_routine(argv[1], pval);
        }
        while (param_step_outer(param, 0));
    }
    else {
        // inner product iteration
        do {
            param_snap(param, pval);
            for (int i = 0; i < iter; i++)
                if (test)
                    test_routine(argv[1], pval);
                else
                    time_routine(argv[1], pval);
        }
        while (param_step_inner(param));
    }
    PLASMA_Finalize();
    printf("\n");
    return EXIT_SUCCESS;
}

/***************************************************************************//**
 *
 * @brief Prints generic usage information.
 *
 ******************************************************************************/
void print_main_usage()
{
    printf("Usage:\n"
           "\ttest [-h|--help]\n"
           "\ttest routine [-h|--help]\n"
           "\ttest routine [parameter1, parameter2, ...]\n");
}

/***************************************************************************//**
 *
 * @brief Prints routine-specific usage information.
 *
 * @param[in] name - routine name
 *
 ******************************************************************************/
void print_routine_usage(char *name)
{
    printf("Usage:\n"
           "\ttest %s [-h|--help]\n"
           "\ttest %s (parameter1, parameter2, ...)\n\n"
           "Options:\n"
           "\t%*sshow this screen\n",
           name, name,
           DescriptionIndent, "-h --help");
    print_usage(PARAM_ITER);
    print_usage(PARAM_OUTER);
    print_usage(PARAM_TEST);
    print_usage(PARAM_TOL);

    printf("\n");
    run_routine(name, NULL, NULL);
}

/***************************************************************************//**
 *
 * @brief Prints usage information for a specific command line option.
 *
 * @param[in] label - command line option label
 *
 ******************************************************************************/
void print_usage(int label)
{
    printf("\t%*s%s\n",
        DescriptionIndent,
        ParamUsage[label][0],
        ParamUsage[label][1]);
}

/***************************************************************************//**
 *
 * @brief Tests a routine for a set of parameter values.
 *        Performs testing and timing.
 *        If pval is NULL, prints column labels.
 *        Otherwise, prints column values.
 *
 * @param[in] name - routine name
 * @param[inout] pval - array of parameter values
 *
 ******************************************************************************/
void test_routine(char *name, param_value_t pval[])
{
    char info[InfoLen];
    run_routine(name, pval, info);

    if (pval == NULL) {
        printf("\n");
        printf("%*s%*s%*s%*s%*s\n",
            InfoSpacing, "Status",
            InfoSpacing, "Error",
            InfoSpacing, "Seconds",
            InfoSpacing, "GFLOPS",
            InfoSpacing, info);
        printf("\n");
    }
    else {
        printf("%*s%*.2le%*lf%*lf%s\n",
            InfoSpacing, pval[PARAM_SUCCESS].i ? "pass" : "FAILED",
            InfoSpacing, pval[PARAM_ERROR].d,
            InfoSpacing, pval[PARAM_TIME].d,
            InfoSpacing, pval[PARAM_GFLOPS].d,
                         info);
    }
}

/***************************************************************************//**
 *
 * @brief Times a routine for a set of parameter values.
 *        Times the routine only, does not test it.
 *        If pval is NULL, prints column labels.
 *        Otherwise, prints column values.
 *
 * @param[in] name - routine name
 * @param[inout] pval - array of parameter values
 *
 ******************************************************************************/
void time_routine(char *name, param_value_t pval[])
{
    char info[InfoLen];
    run_routine(name, pval, info);

    if (pval == NULL) {
        printf("\n");
        printf("%*s%*s%*s\n",
            InfoSpacing, "Seconds",
            InfoSpacing, "GFLOPS",
            InfoSpacing, info);
        printf("\n");
    }
    else {
        printf("%*lf%*lf%s\n",
            InfoSpacing, pval[PARAM_TIME].d,
            InfoSpacing, pval[PARAM_GFLOPS].d,
                         info);
    }
}

/***************************************************************************//**
 *
 * @brief Invokes a specific routine.
 *
 * @param[in] name - routine name
 * @param[inout] pval - array of parameter values
 * @param[out] info - string of column labels or column values
 *
 ******************************************************************************/
void run_routine(char *name, param_value_t pval[], char *info)
{
    if (strcmp(name, "zgemm") == 0)
        test_zgemm(pval, info);
    else if (strcmp(name, "dgemm") == 0)
        test_dgemm(pval, info);
    else if (strcmp(name, "cgemm") == 0)
        test_cgemm(pval, info);
    else if (strcmp(name, "sgemm") == 0)
        test_sgemm(pval, info);
    else if (strcmp(name, "zsymm") == 0)
        test_zsymm(pval, info);
    else if (strcmp(name, "dsymm") == 0)
        test_dsymm(pval, info);
    else if (strcmp(name, "csymm") == 0)
        test_csymm(pval, info);
    else if (strcmp(name, "ssymm") == 0)
        test_ssymm(pval, info);
    else
        assert(0);
}

/***************************************************************************//**
 *
 * @brief Creates an empty array of parameter iterators.
 *
 * @param[out] param - array of parameter iterators.
 *
 ******************************************************************************/
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

/***************************************************************************//**
 *
 * @brief Initializes an array of parameter iterators
 *        according to command lineoptions.
 *
 * @param[in] argc
 * @param[in] argv
 * @param[inout] param - array of parameter iterators
 *
 ******************************************************************************/
int param_read(int argc, char **argv, param_t param[])
{
    int iter = 1;

    //================================================================
    // Initialize parameters from the command line.
    //================================================================
    for (int i = 1; i < argc && argv[i]; i++) {

        //--------------------------------------------------
        // Scan character parameters.
        //--------------------------------------------------
        if (param_starts_with(argv[i], "--outer="))
            param_scan_char(strchr(argv[i], '=')+1, &param[PARAM_OUTER]);
        else if (param_starts_with(argv[i], "--test="))
            param_scan_char(strchr(argv[i], '=')+1, &param[PARAM_TEST]);

        else if (param_starts_with(argv[i], "--transa="))
            param_scan_char(strchr(argv[i], '=')+1, &param[PARAM_TRANSA]);
        else if (param_starts_with(argv[i], "--transb="))
            param_scan_char(strchr(argv[i], '=')+1, &param[PARAM_TRANSB]);

        //--------------------------------------------------
        // Scan integer parameters.
        //--------------------------------------------------
        else if (param_starts_with(argv[i], "--iter="))
            iter = strtol(strchr(argv[i], '=')+1, NULL, 10);

        else if (param_starts_with(argv[i], "--m="))
            param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_M]);
        else if (param_starts_with(argv[i], "--n="))
            param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_N]);
        else if (param_starts_with(argv[i], "--k="))
            param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_K]);

        else if (param_starts_with(argv[i], "--pada="))
            param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_PADA]);
        else if (param_starts_with(argv[i], "--padb="))
            param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_PADB]);
        else if (param_starts_with(argv[i], "--padc="))
            param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_PADC]);

        //--------------------------------------------------
        // Scan double precision parameters.
        //--------------------------------------------------
        else if (param_starts_with(argv[i], "--tol="))
            param_scan_double(strchr(argv[i], '=')+1, &param[PARAM_TOL]);
    }
    //================================================================
    // Set default values for uninitialized parameters.
    //================================================================

    //--------------------------------------------------
    // Set character parameters.
    //--------------------------------------------------
    if (param[PARAM_OUTER].num == 0)
        param_add_char('n', &param[PARAM_OUTER]);
    if (param[PARAM_TEST].num == 0)
        param_add_char('y', &param[PARAM_TEST]);

    if (param[PARAM_TRANSA].num == 0)
        param_add_char('n', &param[PARAM_TRANSA]);
    if (param[PARAM_TRANSB].num == 0)
        param_add_char('n', &param[PARAM_TRANSB]);

    //--------------------------------------------------
    // Set integer parameters.
    //--------------------------------------------------
    if (param[PARAM_M].num == 0)
        param_add_int(1000, &param[PARAM_M]);
    if (param[PARAM_N].num == 0)
        param_add_int(1000, &param[PARAM_N]);
    if (param[PARAM_K].num == 0)
        param_add_int(1000, &param[PARAM_K]);

    if (param[PARAM_PADA].num == 0)
        param_add_int(0, &param[PARAM_PADA]);
    if (param[PARAM_PADB].num == 0)
        param_add_int(0, &param[PARAM_PADB]);
    if (param[PARAM_PADC].num == 0)
        param_add_int(0, &param[PARAM_PADC]);

    //--------------------------------------------------
    // Set double precision parameters.
    //--------------------------------------------------
    if (param[PARAM_TOL].num == 0)
        param_add_double(50.0, &param[PARAM_TOL]);

    return iter;
    }

/***************************************************************************//**
 *
 * @brief Checks if a string starts with a specific prefix.
 *
 * @param[in] str
 * @param[in] prefix
 *
 * @retval 1 - match
 * @retval 0 - no match
 *
 ******************************************************************************/
int param_starts_with(char *str, char *prefix)
{
    size_t n = strlen(prefix);
    if (strncmp(str, prefix, n))
        return 0;
    return 1;
}

/***************************************************************************//**
 *
 * @brief Scans an integer or a range.
 *        Adds the value(s) to a parameter iterator.
 *
 * @param[in] str - string containin an integer
 * @param[inout] param - parameter iterator
 *
 ******************************************************************************/
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

/***************************************************************************//**
 *
 * @brief Scans a character.
 *        Adds the value to a parameter iterator.
 *
 * @param[in] str - string containing a single character
 * @param[inout] param - parameter iterator
 *
 ******************************************************************************/
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

/***************************************************************************//**
 *
 * @brief Scans a double precision number.
 *        Adds the value to a parameter iterator.
 *
 * @param[in] str - string containing a double precision number
 * @param[inout] param - parameter iterator
 *
 ******************************************************************************/
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

/***************************************************************************//**
 *
 * @brief Adds an integer to a parameter iterator.
 *
 * @param[in] ival - integer
 * @param[inout] param - parameter iterator
 *
 ******************************************************************************/
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

/***************************************************************************//**
 *
 * @brief Adds a character to a parameter iterator.
 *
 * @param[in] cval - character
 * @param[inout] param - parameter iterator
 *
 ******************************************************************************/
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

/***************************************************************************//**
 *
 * @brief Adds a double precision number to a parameter iterator.
 *
 * @param[in] dval - double precision value
 * @param[inout] param - parameter iterator
 *
 ******************************************************************************/
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

/***************************************************************************//**
 *
 * @brief Steps through an array of parameter iterators
 *        (inner product evaluation).
 *        Advances all iteratos at the same time.
 *        Iterators that exhausted their range return the last value.
 *
 * @param[inout] param - array of parameter iterators
 *
 * @retval 1 - more iterations
 * @retval 0 - no more iterations
 *
 ******************************************************************************/
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

/***************************************************************************//**
 *
 * @brief Steps through an array of parameter iterators
 *        (outer product evaluation).
 *        Advances one iterator at a time.
 *
 * @param[inout] param - array of parameter iterators
 *
 * @retval 1 - more iterations
 * @retval 0 - no more iterations
 *
 ******************************************************************************/
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

/***************************************************************************//**
 *
 ******************************************************************************/
int param_snap(param_t param[], param_value_t pval[])
{
    for (int i = 0; i < PARAM_SIZEOF; i++)
        pval[i] = param[i].val[param[i].pos];
    return 0;
}
