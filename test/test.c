/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#include "test.h"
#include "plasma.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/******************************************************************************/
typedef void (*test_func_ptr)(param_value_t param[], char *info);

struct routines_t {
    const char *name;
    test_func_ptr func;
};

struct routines_t routines[] =
{
    { "clag2z", test_clag2z },
    { "", NULL },
    { "slag2d", test_slag2d },
    { "", NULL },

    { "dzamax", test_dzamax },
    { "damax",  test_damax  },
    { "scamax", test_scamax },
    { "samax",  test_samax  },

    { "zcposv", test_zcposv },
    { "dsposv", test_dsposv },

    { "zgbsv",  test_zgbsv },
    { "dgbsv",  test_dgbsv },
    { "cgbsv",  test_cgbsv },
    { "sgbsv",  test_sgbsv },

    { "zgbtrf", test_zgbtrf },
    { "dgbtrf", test_dgbtrf },
    { "cgbtrf", test_cgbtrf },
    { "sgbtrf", test_sgbtrf },

    { "zgeadd", test_zgeadd },
    { "dgeadd", test_dgeadd },
    { "cgeadd", test_cgeadd },
    { "sgeadd", test_sgeadd },

    { "zgelqf", test_zgelqf },
    { "dgelqf", test_dgelqf },
    { "cgelqf", test_cgelqf },
    { "sgelqf", test_sgelqf },

    { "zgelqs", test_zgelqs },
    { "dgelqs", test_dgelqs },
    { "cgelqs", test_cgelqs },
    { "sgelqs", test_sgelqs },

    { "zgels", test_zgels },
    { "dgels", test_dgels },
    { "cgels", test_cgels },
    { "sgels", test_sgels },

    { "zgemm", test_zgemm },
    { "dgemm", test_dgemm },
    { "cgemm", test_cgemm },
    { "sgemm", test_sgemm },

    { "zgeqrf", test_zgeqrf },
    { "dgeqrf", test_dgeqrf },
    { "cgeqrf", test_cgeqrf },
    { "sgeqrf", test_sgeqrf },

    { "zgeqrs", test_zgeqrs },
    { "dgeqrs", test_dgeqrs },
    { "cgeqrs", test_cgeqrs },
    { "sgeqrs", test_sgeqrs },

    { "zcgesv", test_zcgesv },
    { "dsgesv", test_dsgesv },

    { "zgesv", test_zgesv },
    { "dgesv", test_dgesv },
    { "cgesv", test_cgesv },
    { "sgesv", test_sgesv },

    { "zgetrf", test_zgetrf },
    { "dgetrf", test_dgetrf },
    { "cgetrf", test_cgetrf },
    { "sgetrf", test_sgetrf },

    { "zgetri", test_zgetri },
    { "dgetri", test_dgetri },
    { "cgetri", test_cgetri },
    { "sgetri", test_sgetri },

    { "zgetri_aux", test_zgetri_aux },
    { "dgetri_aux", test_dgetri_aux },
    { "cgetri_aux", test_cgetri_aux },
    { "sgetri_aux", test_sgetri_aux },

    { "zgetrs", test_zgetrs },
    { "dgetrs", test_dgetrs },
    { "cgetrs", test_cgetrs },
    { "sgetrs", test_sgetrs },

    { "zhemm", test_zhemm },
    { "", NULL },
    { "chemm", test_chemm },
    { "", NULL },

    { "zher2k", test_zher2k },
    { "", NULL },
    { "cher2k", test_cher2k },
    { "", NULL },

    { "zherk", test_zherk },
    { "", NULL },
    { "cherk", test_cherk },
    { "", NULL },

    { "zlacpy", test_zlacpy },
    { "dlacpy", test_dlacpy },
    { "clacpy", test_clacpy },
    { "slacpy", test_slacpy },

    { "zlag2c", test_zlag2c },
    { "", NULL },
    { "dlag2s", test_dlag2s },
    { "", NULL },

    { "zlange", test_zlange },
    { "dlange", test_dlange },
    { "clange", test_clange },
    { "slange", test_slange },

    { "zlanhe", test_zlanhe },
    { "", NULL },
    { "clanhe", test_clanhe },
    { "", NULL },

    { "zlansy", test_zlansy },
    { "dlansy", test_dlansy },
    { "clansy", test_clansy },
    { "slansy", test_slansy },

    { "zlantr", test_zlantr },
    { "dlantr", test_dlantr },
    { "clantr", test_clantr },
    { "slantr", test_slantr },

    { "zlascl", test_zlascl },
    { "dlascl", test_dlascl },
    { "clascl", test_clascl },
    { "slascl", test_slascl },

    { "zlaset", test_zlaset },
    { "dlaset", test_dlaset },
    { "claset", test_claset },
    { "slaset", test_slaset },

    { "zlaswp", test_zlaswp },
    { "dlaswp", test_dlaswp },
    { "claswp", test_claswp },
    { "slaswp", test_slaswp },

    { "zlauum", test_zlauum },
    { "dlauum", test_dlauum },
    { "clauum", test_clauum },
    { "slauum", test_slauum },

    { "zpbsv", test_zpbsv },
    { "dpbsv", test_dpbsv },
    { "cpbsv", test_cpbsv },
    { "spbsv", test_spbsv },

    { "zpbtrf", test_zpbtrf },
    { "dpbtrf", test_dpbtrf },
    { "cpbtrf", test_cpbtrf },
    { "spbtrf", test_spbtrf },

    { "zposv", test_zposv },
    { "dposv", test_dposv },
    { "cposv", test_cposv },
    { "sposv", test_sposv },

    { "zpotrf", test_zpotrf },
    { "dpotrf", test_dpotrf },
    { "cpotrf", test_cpotrf },
    { "spotrf", test_spotrf },

    { "zpotri", test_zpotri },
    { "dpotri", test_dpotri },
    { "cpotri", test_cpotri },
    { "spotri", test_spotri },

    { "zpotrs", test_zpotrs },
    { "dpotrs", test_dpotrs },
    { "cpotrs", test_cpotrs },
    { "spotrs", test_spotrs },

    { "zsymm", test_zsymm },
    { "dsymm", test_dsymm },
    { "csymm", test_csymm },
    { "ssymm", test_ssymm },

    { "zsyr2k", test_zsyr2k },
    { "dsyr2k", test_dsyr2k },
    { "csyr2k", test_csyr2k },
    { "ssyr2k", test_ssyr2k },

    { "zsyrk", test_zsyrk },
    { "dsyrk", test_dsyrk },
    { "csyrk", test_csyrk },
    { "ssyrk", test_ssyrk },

    { "ztradd", test_ztradd },
    { "dtradd", test_dtradd },
    { "ctradd", test_ctradd },
    { "stradd", test_stradd },

    { "ztrmm", test_ztrmm },
    { "dtrmm", test_dtrmm },
    { "ctrmm", test_ctrmm },
    { "strmm", test_strmm },

    { "ztrsm", test_ztrsm },
    { "dtrsm", test_dtrsm },
    { "ctrsm", test_ctrsm },
    { "strsm", test_strsm },

    { "ztrtri", test_ztrtri },
    { "dtrtri", test_dtrtri },
    { "ctrtri", test_ctrtri },
    { "strtri", test_strtri },

    { "zunmlq", test_zunmlq },
    { "dormlq", test_dormlq },
    { "cunmlq", test_cunmlq },
    { "sormlq", test_sormlq },

    { "zunmqr", test_zunmqr },
    { "dormqr", test_dormqr },
    { "cunmqr", test_cunmqr },
    { "sormqr", test_sormqr },

    { NULL, NULL }  // last entry
};

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

    plasma_init();
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
    plasma_finalize();
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
           "\ttest routine [parameter1, parameter2, ...]\n"
           "\n"
           "Available routines:");
    for (int i = 0; routines[i].name != NULL; ++i) {
        if (i % 4 == 0) {
            printf("\n\t");
        }
        printf("%-*s ", InfoSpacing, routines[i].name);
    }
    printf("\n");
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
        printf("%*s %*s %*s %*s %s\n",
            InfoSpacing, "Status",
            InfoSpacing, "Error",
            InfoSpacing, "Seconds",
            InfoSpacing, "GFLOPS",
                         info);
        printf("\n");
        return 0;
    }
    else if (test) {
        printf("%*s %*.2le %*.4lf %*.4lf %s\n",
            InfoSpacing, pval[PARAM_SUCCESS].i ? "pass" : "FAILED",
            InfoSpacing, pval[PARAM_ERROR].d,
            InfoSpacing, pval[PARAM_TIME].d,
            InfoSpacing, pval[PARAM_GFLOPS].d,
                         info);
        return (pval[PARAM_SUCCESS].i == 0);
    }
    else {
        printf("%*s %*s %*.4lf %*.4lf %s\n",
            InfoSpacing, "---",
            InfoSpacing, "---",
            InfoSpacing, pval[PARAM_TIME].d,
            InfoSpacing, pval[PARAM_GFLOPS].d,
                         info);
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
    bool found = false;
    for (int i = 0; routines[i].name != NULL; ++i) {
        if (strcmp(name, routines[i].name) == 0) {
            routines[i].func(pval, info);
            found = true;
            break;
        }
    }
    if (! found) {
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

        else if (param_starts_with(argv[i], "--colrow="))
            err = param_scan_char(strchr(argv[i], '=')+1, &param[PARAM_COLROW]);

        else if (param_starts_with(argv[i], "--norm="))
            err = param_scan_char(strchr(argv[i], '=')+1, &param[PARAM_NORM]);

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
        else if (param_starts_with(argv[i], "--kl="))
            err = param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_KL]);
        else if (param_starts_with(argv[i], "--ku="))
            err = param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_KU]);
        else if (param_starts_with(argv[i], "--nrhs="))
            err = param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_NRHS]);

        else if (param_starts_with(argv[i], "--nb="))
            err = param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_NB]);
        else if (param_starts_with(argv[i], "--ib="))
            err = param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_IB]);

        else if (param_starts_with(argv[i], "--hmode="))
            err = param_scan_char(strchr(argv[i], '=')+1,
                    &param[PARAM_HMODE]);

        else if (param_starts_with(argv[i], "--pada="))
            err = param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_PADA]);
        else if (param_starts_with(argv[i], "--padb="))
            err = param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_PADB]);
        else if (param_starts_with(argv[i], "--padc="))
            err = param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_PADC]);

        else if (param_starts_with(argv[i], "--ntpf="))
            err = param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_NTPF]);
        else if (param_starts_with(argv[i], "--zerocol="))
            err = param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_ZEROCOL]);
        else if (param_starts_with(argv[i], "--incx="))
            err = param_scan_int(strchr(argv[i], '=')+1, &param[PARAM_INCX]);

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
    if (param[PARAM_COLROW].num == 0)
        param_add_char('c', &param[PARAM_COLROW]);
    if (param[PARAM_NORM].num == 0)
        param_add_char('o', &param[PARAM_NORM]);

    //--------------------------------------------------
    // Set integer parameters.
    //--------------------------------------------------
    if (param[PARAM_M].num == 0)
        param_add_int(1000, &param[PARAM_M]);
    if (param[PARAM_N].num == 0)
        param_add_int(1000, &param[PARAM_N]);
    if (param[PARAM_K].num == 0)
        param_add_int(1000, &param[PARAM_K]);
    if (param[PARAM_KL].num == 0)
        param_add_int(200, &param[PARAM_KL]);
    if (param[PARAM_KU].num == 0)
        param_add_int(200, &param[PARAM_KU]);
    if (param[PARAM_NRHS].num == 0)
        param_add_int(1000, &param[PARAM_NRHS]);

    if (param[PARAM_NB].num == 0)
        param_add_int(256, &param[PARAM_NB]);
    if (param[PARAM_IB].num == 0)
        param_add_int(64, &param[PARAM_IB]);

    if (param[PARAM_HMODE].num == 0)
        param_add_char('f', &param[PARAM_HMODE]);

    if (param[PARAM_PADA].num == 0)
        param_add_int(0, &param[PARAM_PADA]);
    if (param[PARAM_PADB].num == 0)
        param_add_int(0, &param[PARAM_PADB]);
    if (param[PARAM_PADC].num == 0)
        param_add_int(0, &param[PARAM_PADC]);

    if (param[PARAM_NTPF].num == 0)
        param_add_int(1, &param[PARAM_NTPF]);
    if (param[PARAM_ZEROCOL].num == 0)
        param_add_int(-1, &param[PARAM_ZEROCOL]);
    if (param[PARAM_INCX].num == 0)
        param_add_int(1, &param[PARAM_INCX]);

    //--------------------------------------------------
    // Set double precision parameters.
    //--------------------------------------------------
    if (param[PARAM_TOL].num == 0)
        param_add_double(50.0, &param[PARAM_TOL]);

    //--------------------------------------------------
    // Set complex parameters.
    //--------------------------------------------------
    if (param[PARAM_ALPHA].num == 0) {
        plasma_complex64_t z = 1.2345 + 2.3456*_Complex_I;
        param_add_complex(z, &param[PARAM_ALPHA]);
    }
    if (param[PARAM_BETA].num == 0) {
        plasma_complex64_t z = 6.7890 + 7.8901*_Complex_I;
        param_add_complex(z, &param[PARAM_BETA]);
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
 * @brief Scans a list of complex numbers in format: 1.23 or 1.23+2.45i.
 *        Adds the value to a parameter iterator. No ranges.
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
        plasma_complex64_t z = re + im*_Complex_I;
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
void param_add_complex(plasma_complex64_t zval, param_t *param)
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
