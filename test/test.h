/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#ifndef TEST_H
#define TEST_H

#include "plasma_types.h"

//==============================================================================
// parameter labels
//==============================================================================
typedef enum {
    //------------------------------------------------------
    // input parameters
    //------------------------------------------------------
    PARAM_ITER,    // outer product iteration?
    PARAM_OUTER,   // outer product iteration?
    PARAM_TEST,    // test the solution?
    PARAM_TOL,     // tolerance
    PARAM_TRANS,   // transposition
    PARAM_TRANSA,  // transposition of A
    PARAM_TRANSB,  // transposition of B
    PARAM_SIDE,    // left of right side application
    PARAM_UPLO,    // general rectangular or upper or lower triangular
    PARAM_DIAG,    // non-unit or unit diagonal
    PARAM_COLROW,  // columnwise or rowwise operation
    PARAM_DIM,     // M, N, K dimensions
    PARAM_KL,      // lower bandwidth
    PARAM_KU,      // upper bandwidth
    PARAM_NRHS,    // number of RHS
    PARAM_NB,      // tile size NBxNB
    PARAM_IB,      // inner blocking size
    PARAM_HMODE,   // Householder mode - tree or flat
    PARAM_ALPHA,   // scalar alpha
    PARAM_BETA,    // scalar beta
    PARAM_PADA,    // padding of A
    PARAM_PADB,    // padding of B
    PARAM_PADC,    // padding of C
    PARAM_NTPF,    // number of threads for panel factorization
    PARAM_NORM,    // type of matrix norm
    PARAM_ZEROCOL, // if positive, a column of zeros inserted at that index
    PARAM_INCX,    // 1 to pivot forward, -1 to pivot backward

    //------------------------------------------------------
    // output parameters
    //------------------------------------------------------
    PARAM_SUCCESS, // success indicator
    PARAM_ERROR,   // numerical error
    PARAM_ORTHO,   // orthogonality error
    PARAM_TIME,    // time to solution
    PARAM_GFLOPS,  // GFLOPS rate

    //------------------------------------------------------
    // Keep at the end!
    //------------------------------------------------------
    PARAM_SIZEOF   // size of parameter array
} param_label_t;

//==============================================================================
// parameter descriptions
//==============================================================================
static const char * const ParamUsage[][2] = {
    //------------------------------------------------------
    // input parameters
    //------------------------------------------------------
    {"--iter=", "number of iterations per set of parameters [default: 1]"},
    {"--outer=[y|n]", "outer product iteration [default: y]"},
    {"--test=[y|n]", "test the solution [default: y]"},
    {"--tol=", "tolerance [default: 50]"},
    {"--trans=[n|t|c]", "transposition [default: n]"},
    {"--transa=[n|t|c]", "transposition of A [default: n]"},
    {"--transb=[n|t|c]", "transposition of B [default: n]"},
    {"--side=[l|r]", "left of right side application [default: l]"},
    {"--uplo=[g|u|l]",
        "general rectangular or upper or lower triangular matrix [default: l]"},
    {"--diag=[n|u]", "not unit triangular or unit matrix [default: n]"},
    {"--colrow=[c|r]", "columnwise or rowwise [default: c]"},
    {"--dim=", "M x N x K dimensions. N and K are optional;"
               " if not given, N=M and K=N [default: 1000x1000x1000]"},
    {"--kl=", "Lower bandwidth [default: 200]"},
    {"--ku=", "Upper bandwidth [default: 200]"},
    {"--nrhs=", "NHRS dimension (number of columns) [default: 1000]"},
    {"--nb=", "NB size of tile (NB by NB) [default: 256]"},
    {"--ib=", "IB inner blocking size [default: 64]"},
    {"--hmode=[f|t]", "Householder mode for QR/LQ - flat or tree [default: f]"},
    {"--alpha=", "scalar alpha"},
    {"--beta=", "scalar beta"},
    {"--pada=", "padding added to lda [default: 0]"},
    {"--padb=", "padding added to ldb [default: 0]"},
    {"--padc=", "padding added to ldc [default: 0]"},
    {"--ntpf=", "number of threads for panel factorization [default: 1]"},
    {"--norm=[m|o|i|f]",
        "type of matrix norm (max, one, inf, frobenius) [default: o]"},
    {"--zerocol=",
        "if positive, a column of zeros inserted at that index [default: -1]"},
    {"--incx=",
        "1 to pivot forward, -1 to pivot backward [default: 1]"},

    //------------------------------------------------------
    // output parameters
    //------------------------------------------------------
    // these are not used, except to assert sizeof(ParamUsage) == PARAM_SIZEOF
    {"success", "success indicator"},
    {"error", "numerical error"},
    {"ortho", "orthogonality error"},
    {"time", "time to solution"},
    {"gflops", "GFLOPS rate"}
};

//==============================================================================
// tester infrastructure
//==============================================================================
typedef struct {
    int m, n, k;
} int3_t;

// parameter value type
typedef union {
    int i;                 // integer
    char c;                // character
    double d;              // double precision
    plasma_complex64_t z;  // double complex
    int3_t dim;            // m, n, k problem size
} param_value_t;

// parameter type
typedef struct {
    int num;            // number of values for a parameter
    int pos;            // current position in the array
    int size;           // size of parameter values array
    param_value_t *val; // array of values for a parameter
} param_t;

// hiding double from precision translation when used for taking time
typedef double plasma_time_t;

// initial size of values array
static const int InitValArraySize = 1024;

// indentation of option descriptions
static const int DescriptionIndent = -20;

// maximum length of info string
static const int InfoLen = 1024;

// spacing in info output string
// each column is InfoSpacing wide + 1 space between columns
static const int InfoSpacing = 11;

// function declarations
void print_main_usage();
void print_routine_usage(const char *name);
void print_usage(int label);
int  test_routine(int test, const char *name, param_value_t param[]);
void run_routine(const char *name, param_value_t pval[], char *info);
void param_init(param_t param[]);
int  param_read(int argc, char **argv, param_t param[]);
int  param_starts_with(const char *str, const char *prefix);
int  scan_irange(const char **strp, int *start, int *end, int *step);
int  scan_drange(const char **strp, double *start, double *end, double *step);
int  param_scan_int(const char *str, param_t *param);
int  param_scan_int3(const char *str, param_t *param);
int  param_scan_char(const char *str, param_t *param);
int  param_scan_double(const char *str, param_t *param);
int  param_scan_complex(const char *str, param_t *param);
void param_add(param_t *param);
void param_add_int(int val, param_t *param);
void param_add_int3(int3_t val, param_t *param);
void param_add_char(char cval, param_t *param);
void param_add_double(double dval, param_t *param);
void param_add_complex(plasma_complex64_t zval, param_t *param);
int  param_step_inner(param_t param[]);
int  param_step_outer(param_t param[], int idx);
int  param_snap(param_t param[], param_value_t value[]);

//==============================================================================
static inline int imin(int a, int b)
{
    if (a < b)
        return a;
    else
        return b;
}

//==============================================================================
static inline int imax(int a, int b)
{
    if (a > b)
        return a;
    else
        return b;
}

#include "test_s.h"
#include "test_d.h"
#include "test_ds.h"
#include "test_c.h"
#include "test_z.h"
#include "test_zc.h"

#endif // TEST_H
