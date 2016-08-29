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

#include "core_blas.h"
#include "test.h"
#include "flops.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef PLASMA_WITH_MKL
    #include <mkl_cblas.h>
    #include <mkl_lapacke.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif
#include <omp.h>

#include "plasma_types.h"
#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_z.h"

#define COMPLEX

/***************************************************************************//**
 *
 * @brief Tests ZGELQF.
 *
 * @param[in]  param - array of parameters
 * @param[out] info  - string of column labels or column values; length InfoLen
 *
 * If param is NULL     and info is NULL,     print usage and return.
 * If param is NULL     and info is non-NULL, set info to column headings
 *                                            and return.
 * If param is non-NULL and info is non-NULL, set info to column values
 *                                            and run test.
 ******************************************************************************/
void test_zgelqf(param_value_t param[], char *info)
{
    //================================================================
    // Print usage info or return column labels or values.
    //================================================================
    if (param == NULL) {
        if (info == NULL) {
            // Print usage info.
            print_usage(PARAM_M);
            print_usage(PARAM_N);
            print_usage(PARAM_PADA);
        }
        else {
            // Return column labels.
            snprintf(info, InfoLen,
                "%*s %*s %*s",
                InfoSpacing, "M",
                InfoSpacing, "N",
                InfoSpacing, "PadA");
        }
        return;
    }
    // Return column values.
    snprintf(info, InfoLen,
        "%*d %*d %*d",
        InfoSpacing, param[PARAM_M].i,
        InfoSpacing, param[PARAM_N].i,
        InfoSpacing, param[PARAM_PADA].i);

    //================================================================
    // Set parameters.
    //================================================================
    int m = param[PARAM_M].i;
    int n = param[PARAM_N].i;

    int lda = imax(1, m + param[PARAM_PADA].i);

    int test = param[PARAM_TEST].c == 'y';
    //double tol = param[PARAM_TOL].d * LAPACKE_dlamch('E');
    double tol = param[PARAM_TOL].d;
    double eps = LAPACKE_dlamch('E');

    //================================================================
    // Allocate and initialize arrays.
    //================================================================
    PLASMA_Complex64_t *A =
        (PLASMA_Complex64_t*)malloc((size_t)lda*n*sizeof(PLASMA_Complex64_t));
    assert(A != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*n, A);
    assert(retval == 0);

    PLASMA_Complex64_t *Aref = NULL;
    if (test) {
        Aref = (PLASMA_Complex64_t*)malloc(
            (size_t)lda*n*sizeof(PLASMA_Complex64_t));
        assert(Aref != NULL);

        memcpy(Aref, A, (size_t)lda*n*sizeof(PLASMA_Complex64_t));
    }

    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    // Initialize tile matrix descriptor for matrix T
    // using multiples of tile size.
    int nb = plasma->nb;
    int ib = plasma->ib;
    int mt = (m%nb == 0) ? (m/nb) : (m/nb+1);
    int nt = (n%nb == 0) ? (n/nb) : (n/nb+1);
    // nt should be doubled if tree-reduction LQ is performed,
    // not implemented now
    PLASMA_desc descT = plasma_desc_init(PlasmaComplexDouble, ib, nb, ib*nb,
                                         mt*ib, nt*nb, 0, 0, mt*ib, nt*nb);
    // allocate memory for the matrix T
    retval = plasma_desc_mat_alloc(&descT);
    assert(retval == 0);

    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_time_t start = omp_get_wtime();
    PLASMA_zgelqf(m, n, A, lda, &descT);
    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = flops_zgelqf(m,n) / time / 1e9;

    //=================================================================
    // Test results by checking orthogonality of Q and precision of L*Q
    //=================================================================
    if (test) {
        // Check the orthogonality of Q

        int minmn = imin(m, n);

        // Allocate space for Q.
        PLASMA_Complex64_t *Q =
            (PLASMA_Complex64_t *)malloc((size_t)m*n*
                                         sizeof(PLASMA_Complex64_t));

        // Build Q.
        int ldq = m;
        PLASMA_zunglq(m, n, minmn, A, lda, &descT, Q, ldq);

        // Build the idendity matrix
        PLASMA_Complex64_t *Id =
            (PLASMA_Complex64_t *) malloc((size_t)minmn*minmn*
                                          sizeof(PLASMA_Complex64_t));
        memset((void*)Id, 0, minmn*minmn*sizeof(PLASMA_Complex64_t));
        for (int i = 0; i < minmn; i++)
            Id[i*minmn+i] = (PLASMA_Complex64_t)1.0;

        PLASMA_Complex64_t zone  =  1.0;
        PLASMA_Complex64_t mzone = -1.0;

        // Perform Id - Q^H * Q    for m >= n,
        //      or Id - Q   * Q^H  for m < n
        if (m >= n)
            cblas_zherk(CblasColMajor, CblasUpper, CblasConjTrans, n, m,
                        mzone, Q, ldq, zone, Id, n);
        else
            cblas_zherk(CblasColMajor, CblasUpper, CblasNoTrans,   m, n,
                        mzone, Q, ldq, zone, Id, m);

        // WORK array of size m is needed for computing L_oo norm
        double *WORK = (double *) malloc((size_t)m*sizeof(double));

        // |Id - Q^H * Q|_oo
        double norm_orth = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'I',
                                               minmn, minmn, Id, minmn, WORK);

        // normalize the result
        // |Id - Q^H * Q|_oo / (n * eps)
        double norm_norm_ortho = norm_orth/(minmn*eps);
        if (norm_norm_ortho > tol)
            printf("WARNING: error in orthogonality %lf is above "
                   "the tolerance %lf \n", norm_norm_ortho, tol);

        free(Q);
        free(Id);

        // Check the accuracy of A - L * Q
        // LAPACK version does not construct Q, it uses only application of it

        // Extract the L.
        PLASMA_Complex64_t *L =
            (PLASMA_Complex64_t *)malloc((size_t)m*n*
                                         sizeof(PLASMA_Complex64_t));
        memset((void*)L, 0., (size_t)m*n*sizeof(PLASMA_Complex64_t));
        LAPACKE_zlacpy_work(LAPACK_COL_MAJOR,'l', m, n, A, lda, L, m);

        // Compute L * Q.
        PLASMA_zunmlq(PlasmaRight, PlasmaNoTrans, m, n, minmn, A, lda, &descT,
                      L, m);

        // Compute the difference.
        // L = A - L*Q
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                L[j*m+i] = Aref[j*lda+i] - L[j*m+i];

        // |A|_oo
        double normA = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'I', m, n,
                                           Aref, lda, WORK);

        // |A - L*Q|_oo
        double norm_AmLQ = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'I', m, n,
                                               L, m, WORK);

        // normalize the result
        // |A-LQ|_oo / (|A|_oo * n * eps)
        double norm_norm_AmLQ = norm_AmLQ / (normA * n * eps);

        // print the worst of the two results
        double result =
            (norm_norm_ortho > norm_norm_AmLQ ? norm_norm_ortho :
                                                norm_norm_AmLQ);

        param[PARAM_ERROR].d = result;
        param[PARAM_SUCCESS].i = result < tol;

        free(WORK);
        free(L);
    }

    //================================================================
    // Free arrays.
    //================================================================
    plasma_desc_mat_free(&descT);
    free(A);
    if (test)
        free(Aref);
}
