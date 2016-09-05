/**
 *
 * @file test.c
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#include "test.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "plasma.h"

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
 * @retval > 0 - number of tests that failed
 *
 ******************************************************************************/
int main(int argc, char **argv)
{
    if (argc == 1 ||
        strcmp(argv[1], "-h") == 0 ||
        strcmp(argv[1], "--help") == 0) {

        print_main_usage();
        return EXIT_SUCCESS;
    }

    const char *routine = argv[1];

    // Ensure that ParamUsage has an entry for every param_label_t value.
    assert(PARAM_SIZEOF == sizeof(ParamUsage)/(2*sizeof(char*)));

    param_t param[PARAM_SIZEOF];      // set of parameters
    param_value_t pval[PARAM_SIZEOF]; // snapshot of values

    param_init(param);
    int iter = param_read(argc, argv, param);
    int outer = param[PARAM_OUTER].val[0].c == 'y';
    int test = param[PARAM_TEST].val[0].c == 'y';
    int err = 0;

    // Print labels.
    test_routine(test, routine, NULL);

    PLASMA_Init();
    if (outer) {
        // outer product iteration
        do {
            param_snap(param, pval);
            for (int i = 0; i < iter; i++) {
                err += test_routine(test, routine, pval);
            }
            if (iter > 1) {
                printf("\n");
            }
        }
        while (param_step_outer(param, 0));
    }
    else {
        // inner product iteration
        do {
            param_snap(param, pval);
            for (int i = 0; i < iter; i++) {
                err += test_routine(test, routine, pval);
            }
            if (iter > 1) {
                printf("\n");
            }
        }
        while (param_step_inner(param));
    }
    PLASMA_Finalize();
    printf("\n");
    return err;
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
void print_routine_usage(const char *name)
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
 *        Otherwise, runs routine and prints column values.
 *
 * @param[in]    test - if true, tests routine, else only times routine
 * @param[in]    name - routine name
 * @param[inout] pval - array of parameter values
 *
 * @retval 1 - failure
 * @retval 0 - success
 *
 ******************************************************************************/
int test_routine(int test, const char *name, param_value_t pval[])
{
    char info[InfoLen];
    run_routine(name, pval, info);

    if (pval == NULL) {
        printf("\n");
        printf("%*s %*s %s %*s %*s\n",
            InfoSpacing, "Seconds",
            InfoSpacing, "GFLOPS",
                         info,
            InfoSpacing, "Error",
            InfoSpacing, "Status");
        printf("\n");
        return 0;
    }
    else if (test) {
        printf("%*.4lf %*.4lf %s %*.2le %*s\n",
            InfoSpacing, pval[PARAM_TIME].d,
            InfoSpacing, pval[PARAM_GFLOPS].d,
                         info,
            InfoSpacing, pval[PARAM_ERROR].d,
            InfoSpacing, pval[PARAM_SUCCESS].i ? "pass" : "FAILED");
        return (pval[PARAM_SUCCESS].i == 0);
    }
    else {
        printf("%*.4lf %*.4lf %s %*s %*s\n",
            InfoSpacing, pval[PARAM_TIME].d,
            InfoSpacing, pval[PARAM_GFLOPS].d,
                         info,
            InfoSpacing, "---",
            InfoSpacing, "---");
        return 0;
    }
}

/***************************************************************************//**
 *
 * @brief Invokes a specific routine.
 *
 * @param[in]    name - routine name
 * @param[inout] pval - array of parameter values
 * @param[out]   info - string of column labels or column values; length InfoLen
 *
 ******************************************************************************/
void run_routine(const char *name, param_value_t pval[], char *info)
{
    if      (strcmp(name, "zgelqf") == 0)
        test_zgelqf(pval, info);
    else if (strcmp(name, "dgelqf") == 0)
        test_dgelqf(pval, info);
    else if (strcmp(name, "cgelqf") == 0)
        test_cgelqf(pval, info);
    else if (strcmp(name, "sgelqf") == 0)
        test_sgelqf(pval, info);

    else if (strcmp(name, "zgelqs") == 0)
        test_zgelqs(pval, info);
    else if (strcmp(name, "dgelqs") == 0)
        test_dgelqs(pval, info);
    else if (strcmp(name, "cgelqs") == 0)
        test_cgelqs(pval, info);
    else if (strcmp(name, "sgelqs") == 0)
        test_sgelqs(pval, info);

    else if (strcmp(name, "zgels") == 0)
        test_zgels(pval, info);
    else if (strcmp(name, "dgels") == 0)
        test_dgels(pval, info);
    else if (strcmp(name, "cgels") == 0)
        test_cgels(pval, info);
    else if (strcmp(name, "sgels") == 0)
        test_sgels(pval, info);

    else if (strcmp(name, "zgemm") == 0)
        test_zgemm(pval, info);
    else if (strcmp(name, "dgemm") == 0)
        test_dgemm(pval, info);
    else if (strcmp(name, "cgemm") == 0)
        test_cgemm(pval, info);
    else if (strcmp(name, "sgemm") == 0)
        test_sgemm(pval, info);

    else if (strcmp(name, "zgeqrf") == 0)
        test_zgeqrf(pval, info);
    else if (strcmp(name, "dgeqrf") == 0)
        test_dgeqrf(pval, info);
    else if (strcmp(name, "cgeqrf") == 0)
        test_cgeqrf(pval, info);
    else if (strcmp(name, "sgeqrf") == 0)
        test_sgeqrf(pval, info);

    else if (strcmp(name, "zgeqrs") == 0)
        test_zgeqrs(pval, info);
    else if (strcmp(name, "dgeqrs") == 0)
        test_dgeqrs(pval, info);
    else if (strcmp(name, "cgeqrs") == 0)
        test_cgeqrs(pval, info);
    else if (strcmp(name, "sgeqrs") == 0)
        test_sgeqrs(pval, info);

    else if (strcmp(name, "zhemm") == 0)
        test_zherk(pval, info);
    else if (strcmp(name, "chemm") == 0)
        test_cherk(pval, info);

    else if (strcmp(name, "zher2k") == 0)
        test_zher2k(pval, info);
    else if (strcmp(name, "cher2k") == 0)
        test_cher2k(pval, info);

    else if (strcmp(name, "zherk") == 0)
        test_zherk(pval, info);
    else if (strcmp(name, "cherk") == 0)
        test_cherk(pval, info);

    else if (strcmp(name, "zposv") == 0)
        test_zpotrf(pval, info);
    else if (strcmp(name, "dposv") == 0)
        test_dpotrf(pval, info);
    else if (strcmp(name, "cposv") == 0)
        test_cpotrf(pval, info);
    else if (strcmp(name, "sposv") == 0)
        test_spotrf(pval, info);

    else if (strcmp(name, "zpotrf") == 0)
        test_zpotrf(pval, info);
    else if (strcmp(name, "dpotrf") == 0)
        test_dpotrf(pval, info);
    else if (strcmp(name, "cpotrf") == 0)
        test_cpotrf(pval, info);
    else if (strcmp(name, "spotrf") == 0)
        test_spotrf(pval, info);

    else if (strcmp(name, "zpotrs") == 0)
        test_zpotrf(pval, info);
    else if (strcmp(name, "dpotrs") == 0)
        test_dpotrf(pval, info);
    else if (strcmp(name, "cpotrs") == 0)
        test_cpotrf(pval, info);
    else if (strcmp(name, "spotrs") == 0)
        test_spotrf(pval, info);

    else if (strcmp(name, "zsymm") == 0)
        test_zsyrk(pval, info);
    else if (strcmp(name, "dsymm") == 0)
        test_dsyrk(pval, info);
    else if (strcmp(name, "csymm") == 0)
        test_csyrk(pval, info);
    else if (strcmp(name, "ssymm") == 0)
        test_ssyrk(pval, info);

    else if (strcmp(name, "zsyrk") == 0)
        test_zsyrk(pval, info);
    else if (strcmp(name, "dsyrk") == 0)
        test_dsyrk(pval, info);
    else if (strcmp(name, "csyrk") == 0)
        test_csyrk(pval, info);
    else if (strcmp(name, "ssyrk") == 0)
        test_ssyrk(pval, info);

    else if (strcmp(name, "zsyr2k") == 0)
        test_zsyr2k(pval, info);
    else if (strcmp(name, "dsyr2k") == 0)
        test_dsyr2k(pval, info);
    else if (strcmp(name, "csyr2k") == 0)
        test_csyr2k(pval, info);
    else if (strcmp(name, "ssyr2k") == 0)
        test_ssyr2k(pval, info);

    else if (strcmp(name, "ztrsm") == 0)
        test_ztrsm(pval, info);
    else if (strcmp(name, "dtrsm") == 0)
        test_dtrsm(pval, info);
    else if (strcmp(name, "ctrsm") == 0)
        test_ctrsm(pval, info);
    else if (strcmp(name, "strsm") == 0)
        test_strsm(pval, info);

    else if (strcmp(name, "ztrmm") == 0)
        test_ztrmm(pval, info);
    else if (strcmp(name, "dtrmm") == 0)
        test_dtrmm(pval, info);
    else if (strcmp(name, "ctrmm") == 0)
        test_ctrmm(pval, info);
    else if (strcmp(name, "strmm") == 0)
        test_strmm(pval, info);

    else if (strcmp(name, "zcposv") == 0)
        test_zcposv(pval, info);
//    else if (strcmp(name, "dsposv") == 0)
//        test_dsposv(pval, info);

    else {
        printf("unknown routine: %s\n", name);
        exit(EXIT_FAILURE);
    }
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
 *        Assumes argv[1] is function name; parses argv[2:argc-1].
 *
 * @param[in]    argc
 * @param[in]    argv
 * @param[inout] param - array of parameter iterators
 *
 * @retval iter
 *
 ******************************************************************************/
int param_read(int argc, char **argv, param_t param[])
{
    int err = 0;
    int iter = 1;
    const char *routine = argv[1];

    //================================================================
    // Initialize parameters from the command line.
    //================================================================
    for (int i = 2; i < argc && argv[i]; i++) {
        //--------------------------------------------------
        // Scan character parameters.
        //--------------------------------------------------
        if (param_starts_with(argv[i], "--outer="))
            err = param_scan_char(strchr(argv[i], '=')+1, &param[PARAM_OUTER]);
        else if (param_starts_with(argv[i], "--test="))
            err = param_scan_char(strchr(argv[i], '=')+1, &param[PARAM_TEST]);

        else if (param_starts_with(argv[i], "--side="))
            err = param_scan_char(strchr(argv[i], '=')+1, &param[PARAM_SIDE]);

        else if (param_starts_with(argv[i], "--trans="))
            err = param_scan_char(strchr(argv[i], '=')+1, &param[PARAM_TRANS]);
        else if (param_starts_with(argv[i], "--transa="))
            err = param_scan_char(strchr(argv[i], '=')+1, &param[PARAM_TRANSA]);
        else if (param_starts_with(argv[i], "--transb="))
            err = param_scan_char(strchr(argv[i], '=')+1, &param[PARAM_TRANSB]);

        else if (param_starts_with(argv[i], "--uplo="))
            err = param_scan_char(strchr(argv[i], '=')+1, &param[PARAM_UPLO]);

        else if (param_starts_with(argv[i], "--diag="))
            err = param_scan_char(strchr(argv[i], '=')+1, &param[PARAM_DIAG]);

        //--------------------------------------------------
        // Scan integer parameters.
        //--------------------------------------------------
        else if (param_starts_with(argv[i], "--iter="))
            iter = strtol(strchr(argv[i], '=')+1, NULL, 10);

        else if (param_starts_with(argv[i], "--m="))
            err = param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_M]);
        else if (param_starts_with(argv[i], "--n="))
            err = param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_N]);
        else if (param_starts_with(argv[i], "--k="))
            err = param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_K]);
        else if (param_starts_with(argv[i], "--nrhs="))
            err = param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_NRHS]);

        else if (param_starts_with(argv[i], "--nb="))
            err = param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_NB]);
        else if (param_starts_with(argv[i], "--ib="))
            err = param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_IB]);

        else if (param_starts_with(argv[i], "--pada="))
            err = param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_PADA]);
        else if (param_starts_with(argv[i], "--padb="))
            err = param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_PADB]);
        else if (param_starts_with(argv[i], "--padc="))
            err = param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_PADC]);

        //--------------------------------------------------
        // Scan double precision parameters.
        //--------------------------------------------------
        else if (param_starts_with(argv[i], "--tol="))
            err = param_scan_double(strchr(argv[i], '=')+1, &param[PARAM_TOL]);

        //--------------------------------------------------
        // Scan complex parameters.
        //--------------------------------------------------
        else if (param_starts_with(argv[i], "--alpha="))
            err = param_scan_complex(strchr(argv[i], '=')+1,
                                     &param[PARAM_ALPHA]);
        else if (param_starts_with(argv[i], "--beta="))
            err = param_scan_complex(strchr(argv[i], '=')+1,
                                     &param[PARAM_BETA]);

        //--------------------------------------------------
        // Handle help and errors.
        //--------------------------------------------------
        else if (strcmp(argv[i], "-h") == 0 ||
                 strcmp(argv[i], "--help") == 0) {
            print_routine_usage(routine);
            exit(EXIT_SUCCESS);
        }
        else {
            printf("unknown argument: %s\n", argv[i]);
            exit(EXIT_FAILURE);
        }

        if (err) {
            printf("error scanning argument: %s\n", argv[i]);
            exit(EXIT_FAILURE);
        }
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

    if (param[PARAM_SIDE].num == 0)
        param_add_char('l', &param[PARAM_SIDE]);
    if (param[PARAM_TRANS].num == 0)
        param_add_char('n', &param[PARAM_TRANS]);
    if (param[PARAM_TRANSA].num == 0)
        param_add_char('n', &param[PARAM_TRANSA]);
    if (param[PARAM_TRANSB].num == 0)
        param_add_char('n', &param[PARAM_TRANSB]);
    if (param[PARAM_UPLO].num == 0)
        param_add_char('l', &param[PARAM_UPLO]);
    if (param[PARAM_DIAG].num == 0)
        param_add_char('n', &param[PARAM_DIAG]);

    //--------------------------------------------------
    // Set integer parameters.
    //--------------------------------------------------
    if (param[PARAM_M].num == 0)
        param_add_int(1000, &param[PARAM_M]);
    if (param[PARAM_N].num == 0)
        param_add_int(1000, &param[PARAM_N]);
    if (param[PARAM_K].num == 0)
        param_add_int(1000, &param[PARAM_K]);
    if (param[PARAM_NRHS].num == 0)
        param_add_int(1000, &param[PARAM_NRHS]);

    if (param[PARAM_NB].num == 0)
        param_add_int(256, &param[PARAM_NB]);
    if (param[PARAM_IB].num == 0)
        param_add_int(64, &param[PARAM_IB]);

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

    //--------------------------------------------------
    // Set complex parameters.
    //--------------------------------------------------
    if (param[PARAM_ALPHA].num == 0) {
        PLASMA_Complex64_t z = 1.2345 + 2.3456*_Complex_I;
        param_add_complex(z, &param[PARAM_ALPHA]);
    }
    if (param[PARAM_BETA].num == 0) {
        PLASMA_Complex64_t z = 6.7890 + 7.8901*_Complex_I;
        param_add_double(z, &param[PARAM_BETA]);
    }

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
int param_starts_with(const char *str, const char *prefix)
{
    size_t n = strlen(prefix);
    if (strncmp(str, prefix, n))
        return 0;
    return 1;
}

/***************************************************************************//**
 *
 * @brief Scans a list of integers or ranges (start:end:step).
 *        Adds the value(s) to a parameter iterator.
 *
 * @param[in]    str   - string containin an integer
 * @param[inout] param - parameter iterator
 *
 * @retval 1 - failure
 * @retval 0 - success
 *
 ******************************************************************************/
int param_scan_int(const char *str, param_t *param)
{
    char *endptr;
    do {
        long start = strtol(str, &endptr, 10);
        if (endptr == str) {
            return 1;
        }
        if (*endptr == ':') {
            str = endptr+1;
            long stop = strtol(str, &endptr, 10);
            if (endptr == str || *endptr != ':') {
                return 1;
            }

            str = endptr+1;
            long step = strtol(str, &endptr, 10);
            if (endptr == str || step <= 0) {
                return 1;
            }

            for (int i = start; i <= stop; i += step) {
                param_add_int(i, param);
            }
        }
        else {
            param_add_int(start, param);
        }
        str = endptr+1;
    }
    while (*endptr != '\0');
    return 0;
}

/***************************************************************************//**
 *
 * @brief Scans a list of characters.
 *        Adds the value(s) to a parameter iterator.
 *
 * @param[in]    str   - string containing a single character
 * @param[inout] param - parameter iterator
 *
 * @retval 1 - failure
 * @retval 0 - success
 *
 ******************************************************************************/
int param_scan_char(const char *str, param_t *param)
{
    const char *endptr;
    do {
        if (*str == '\0') {
            return 1;
        }
        param_add_char(*str, param);
        endptr = str+1;
        str = endptr+1;
    }
    while (*endptr != '\0');
    return 0;
}

/***************************************************************************//**
 *
 * @brief Scans a list of double precision numbers or ranges (start:end:step).
 *        Adds the value(s) to a parameter iterator.
 *
 * @param[in]    str   - string containing a double precision number
 * @param[inout] param - parameter iterator
 *
 * @retval 1 - failure
 * @retval 0 - success
 *
 ******************************************************************************/
int param_scan_double(const char *str, param_t *param)
{
    char *endptr;
    do {
        double start = strtod(str, &endptr);
        if (endptr == str) {
            return 1;
        }
        if (*endptr == ':') {
            str = endptr+1;
            double stop = strtod(str, &endptr);
            if (endptr == str || *endptr != ':') {
                return 1;
            }

            str = endptr+1;
            double step = strtod(str, &endptr);
            if (endptr == str || step <= 0) {
                return 1;
            }

            // add fraction of step to allow for rounding error
            for (double d = start; d <= stop + step/10.; d += step) {
                param_add_double(d, param);
            }
        }
        else {
            param_add_double(start, param);
        }
        str = endptr+1;
    }
    while (*endptr != '\0');
    return 0;
}

/***************************************************************************//**
 *
 * @brief Scans a list of complex numbers in format: 1.23 or 1.23+2.45i. (No ranges.)
 *        Adds the value(s) to a parameter iterator.
 *
 * @param[in]    str   - string containing a double precision number
 * @param[inout] param - parameter iterator
 *
 * @retval 1 - failure
 * @retval 0 - success
 *
 ******************************************************************************/
int param_scan_complex(const char *str, param_t *param)
{
    char *endptr;
    do {
        double re = strtod(str, &endptr);
        double im = 0.0;
        if (endptr == str) {
            return 1;
        }
        if (*endptr == '+') {
            str = endptr+1;
            im = strtod(str, &endptr);
            if (endptr == str || *endptr != 'i') {
                return 1;
            }
            endptr += 1;  // skip 'i'
        }
        PLASMA_Complex64_t z = re + im*_Complex_I;
        param_add_complex(z, param);
        str = endptr+1;
    }
    while (*endptr != '\0');
    return 0;
}

/***************************************************************************//**
 *
 * @brief Adds an integer to a parameter iterator.
 *
 * @param[in]    ival  - integer
 * @param[inout] param - parameter iterator
 *
 ******************************************************************************/
void param_add_int(int ival, param_t *param)
{
    param->val[param->num].i = ival;
    param->num++;
    if (param->num == param->size) {
        param->size *= 2;
        param->val = (param_value_t*) realloc(
            param->val, param->size*sizeof(param_value_t));
        assert(param->val != NULL);
    }
}

/***************************************************************************//**
 *
 * @brief Adds a character to a parameter iterator.
 *
 * @param[in]    cval  - character
 * @param[inout] param - parameter iterator
 *
 ******************************************************************************/
void param_add_char(char cval, param_t *param)
{
    param->val[param->num].c = cval;
    param->num++;
    if (param->num == param->size) {
        param->size *= 2;
        param->val = (param_value_t*) realloc(
            param->val, param->size*sizeof(param_value_t));
        assert(param->val != NULL);
    }
}

/***************************************************************************//**
 *
 * @brief Adds a double precision number to a parameter iterator.
 *
 * @param[in]    dval  - double precision value
 * @param[inout] param - parameter iterator
 *
 ******************************************************************************/
void param_add_double(double dval, param_t *param)
{
    param->val[param->num].d = dval;
    param->num++;
    if (param->num == param->size) {
        param->size *= 2;
        param->val = (param_value_t*) realloc(
            param->val, param->size*sizeof(param_value_t));
        assert(param->val != NULL);
    }
}

/***************************************************************************//**
 *
 * @brief Adds a complex number to a parameter iterator.
 *
 * @param[in]    zval  - complex value
 * @param[inout] param - parameter iterator
 *
 ******************************************************************************/
void param_add_complex(PLASMA_Complex64_t zval, param_t *param)
{
    param->val[param->num].z = zval;
    param->num++;
    if (param->num == param->size) {
        param->size *= 2;
        param->val = (param_value_t*) realloc(
            param->val, param->size*sizeof(param_value_t));
        assert(param->val != NULL);
    }
}

/***************************************************************************//**
 *
 * @brief Steps through an array of parameter iterators
 *        (inner product evaluation).
 *        Advances all iterators at the same time.
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
    while (param[idx].num == 0) {
        if (++idx == PARAM_SIZEOF) {
            return 0;
        }
    }

    if (++param[idx].pos == param[idx].num) {
        param[idx].pos = 0;
        return param_step_outer(param, idx+1);
    }
    return 1;
}

/***************************************************************************//**
 *
 * @brief Copies a snapshot of the current iteration of param iterators to pval.
 *
 * @param[in]  param - array of parameter iterators
 * @param[out] pval  - array of parameter values
 *
 ******************************************************************************/
int param_snap(param_t param[], param_value_t pval[])
{
    for (int i = 0; i < PARAM_SIZEOF; i++) {
        pval[i] = param[i].val[param[i].pos];
    }
    return 0;
}
