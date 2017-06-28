/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> s d c
 *
 **/

#include "test.h"
#include "flops.h"
#include "core_blas.h"
#include "core_lapack.h"
#include "plasma.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <omp.h>

#define COMPLEX
double zlangb_(char *, int *, int *, int *, plasma_complex64_t *, int *, double *);
 //   double value = plasma_zlangb(norm, m, n, kl, ku,  AB, ldab);
double plasma_zlangb(plasma_enum_t, int, int, int, int, plasma_complex64_t *, int);
/***************************************************************************//**
 *
 * @brief Tests ZLANGB.
 *
 * @param[in,out] param - array of parameters
 * @param[in]     run - whether to run test
 *
 * Sets flags in param indicating which parameters are used.
 * If run is true, also runs test and stores output parameters.
 ******************************************************************************/
void test_zlangb(param_value_t param[], bool run)
{
    // pwu: experiements:
    //test_lapack_zlangb();
    //================================================================
    // Mark which parameters are used.
    //================================================================
    param[PARAM_NORM   ].used = true;
    param[PARAM_DIM    ].used = PARAM_USE_M | PARAM_USE_N;
    param[PARAM_PADA   ].used = true;
    param[PARAM_NB     ].used = true;
    param[PARAM_KL     ].used = true;
    param[PARAM_KU     ].used = true;
    if (! run)
        return;

    //================================================================
    // Set parameters.
    //================================================================
    plasma_enum_t norm = plasma_norm_const(param[PARAM_NORM].c);

    int m = param[PARAM_DIM].dim.m;
    int n = param[PARAM_DIM].dim.n;
    int kl = param[PARAM_KL].i;
    int ku = param[PARAM_KU].i;

    int lda = imax(1, m + param[PARAM_PADA].i);

    int test = param[PARAM_TEST].c == 'y';
    double eps = LAPACKE_dlamch('E');

    //================================================================
    // Set tuning parameters.
    //================================================================
    plasma_set(PlasmaNb, param[PARAM_NB].i);

    //================================================================
    // Allocate and initialize arrays.
    //================================================================
    plasma_complex64_t *A =
        (plasma_complex64_t*)malloc((size_t)lda*n*sizeof(plasma_complex64_t));
    assert(A != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*n, A);
    assert(retval == 0);
    // zero out elements outside the band
    for (int i = 0; i < m; i++) {
        for (int j = i+ku+1; j < n; j++) A[i + j*lda] = 0.0;
    }
    for (int j = 0; j < n; j++) {
        for (int i = j+kl+1; i < m; i++) A[i + j*lda] = 0.0;
    }
#if 0
    printf("[test_zlangb]: inspecting matrix A:\n");
    printf("m\tn\tlda\t\n");
    printf("%d\t%d\t%d\t\n", m, n, lda);
    printf("A\t");
    for (int j=0; j<n; j++) printf("%d\t", j);
    printf("\n");
    for (int i=0; i<m; i++) {
	printf("%d\t",i);
	for (int j=0; j<n; j++) {
	    if (A[i+j*lda]!=0) printf("%.2f\t", A[i+j*lda]);
	    else printf("*\t");
	}
	printf("\n");
    }
#endif 
    plasma_complex64_t *Aref = NULL;
    if (test) {
        Aref = (plasma_complex64_t*)malloc(
            (size_t)lda*n*sizeof(plasma_complex64_t));
        assert(Aref != NULL);

        memcpy(Aref, A, (size_t)lda*n*sizeof(plasma_complex64_t));
    }

    int nb = param[PARAM_NB].i;
    // band matrix A in skewed LAPACK storage
    int kut  = (ku+kl+nb-1)/nb; // # of tiles in upper band (not including diagonal)
    int klt  = (kl+nb-1)/nb;    // # of tiles in lower band (not including diagonal)
    int ldab = (kut+klt+1)*nb;  // since we use zgetrf on panel, we pivot back within panel.
                                // this could fill the last tile of the panel,
                                // and we need extra NB space on the bottom
    plasma_complex64_t *AB = NULL;
    AB = (plasma_complex64_t*)malloc((size_t)ldab*n*sizeof(plasma_complex64_t));
    assert(AB != NULL);
    // convert into LAPACK's skewed storage
    for (int j = 0; j < n; j++) {
        int i_kl = imax(0,   j-ku);
        int i_ku = imin(m-1, j+kl);
        for (int i = 0; i < ldab; i++) AB[i + j*ldab] = 0.0;
        for (int i = i_kl; i <= i_ku; i++) AB[kl + i-(j-ku) + j*ldab] = A[i + j*lda];
    }
    //retval = LAPACKE_zlarnv(1, seed, (size_t)ldab*n, AB);
    //assert(retval == 0);
#if 0
    printf("[test_zlangb]: inspecting matrix AB:\n");
    printf("m\tn\tlda\t\n");
    printf("%d\t%d\t%d\t\n", m, n, ldab);
    printf("AB\t");
    for (int j=0; j<n; j++) printf("%d\t", j);
    printf("\n");
    for (int i=0; i<ldab; i++) {
	printf("%d\t",i);
	for (int j=0; j<n; j++) {
	    if (AB[i+j*ldab]!=0) printf("%.2f\t", AB[i+j*ldab]);
	    else printf("*\t");
	}
	printf("\n");
    }
#endif
    plasma_complex64_t *ABref = NULL;
    if (test) {
        ABref = (plasma_complex64_t*)malloc(
            (size_t)ldab*n*sizeof(plasma_complex64_t));
        assert(ABref != NULL);

        memcpy(ABref, AB, (size_t)ldab*n*sizeof(plasma_complex64_t));
    }
    
    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_time_t start = omp_get_wtime();
    double value = plasma_zlangb(norm, m, n, kl, ku,  AB, ldab);
    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = flops_zlange(m, n, norm) / time / 1e9;

    //================================================================
    // Test results by comparing to a reference implementation.
    //================================================================
    if (test) {
        char *cnorm;
	double *workspace=NULL;
        switch (norm) {
        case PlasmaMaxNorm:
            cnorm = "m";
            break;
        case PlasmaOneNorm:
            cnorm = "1";
            break;
	case PlasmaInfNorm:
	    cnorm = "i";
	    workspace = (double*) malloc(n*sizeof(double));
	    break;
	case PlasmaFrobeniusNorm:
	    cnorm = "f";
	    break;
        default:
            assert(0);
        }

#if 0
        printf("[test_zlangb]: kll=%d,kuu=%d, ldab=%d, ABref[ku,0]=%f\n", 
               kll, kuu, ldab, *(double*)&ABref[kl]);
#endif
        double valueRef =
            //LAPACKE_zlange(LAPACK_COL_MAJOR, lapack_const(norm),
            //               kl+ku+1, n, ABref+kl, ldab);
            //zlange_(cnorm, &kuu, &n, ABref+kl, &ldab, NULL);
            zlangb_(cnorm, &n,&kl, &ku, ABref+kl, &ldab, workspace);

        // Calculate relative error
        double error = fabs(value-valueRef);
#if 0
        printf("[test_zlangb]: value=%f,valueRef=%f\n", value, valueRef);
#endif
        if (valueRef != 0)
            error /= valueRef;
        double tol = eps;
        double normalize = 1;
        switch (norm) {
            case PlasmaInfNorm:
                // Sum order on the line can differ
                normalize = n;
                break;

            case PlasmaOneNorm:
                // Sum order on the column can differ
                normalize = m;
                break;

            case PlasmaFrobeniusNorm:
                // Sum order on every element can differ
                normalize = m*n;
                break;
        }
        error /= normalize;
        param[PARAM_ERROR].d   = error;
        param[PARAM_SUCCESS].i = error < tol;
    }

    //================================================================
    // Free arrays.
    //================================================================
    free(A);free(AB);
    if (test) {
        free(Aref);free(ABref);
    }
}



void test_lapack_zlangb(){
    int kl=2, ku=1, n=6;
    int ldab = kl+ku+1;
    plasma_complex64_t *AB = (plasma_complex64_t*)malloc(ldab*n*sizeof(plasma_complex64_t));
    int seed[] = {0,0,0,1};
    int retval = LAPACKE_zlarnv(1, seed, (size_t)ldab*n, AB);
    assert(retval == 0);
    
    printf("=================BEGIN=======================\n");
    printf("===MATRIX AB====\n");
    for (int i=0; i<ldab; i++) {
        for (int j=0; j<n; j++) 
            printf("%.3f\t", cabs(AB[i+j*ldab]));
        printf("\n");
    }
    printf("====LAPACK_ZLANGB====\n");
    double v, v2;
    char c = 'M';
    v2 = zlangb_(&c, &n, &kl, &ku, AB, &ldab, NULL);
    v = zlange_(&c, &ldab, &n, AB, &ldab, NULL);
    printf("v=%f, v2=%f\n", v, v2);
    
    
    printf("=================END=======================\n");
}
    
