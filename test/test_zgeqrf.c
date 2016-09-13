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
#include "core_lapack.h"
#include "plasma.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

#define COMPLEX

/***************************************************************************//**
 *
 * @brief Tests ZGEQRF.
 *
 * @param[in]  param - array of parameters
 * @param[out] info  - string of column labels or column values; length InfoLen
 *
 * If param is NULL and info is NULL,     print usage and return.
 * If param is NULL and info is non-NULL, set info to column labels and return.
 * If param is non-NULL and info is non-NULL, set info to column values
 * and run test.
 ******************************************************************************/
void test_zgeqrf(param_value_t param[], char *info)
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
            print_usage(PARAM_NB);
            print_usage(PARAM_IB);
        }
        else {
            // Return column labels.
            snprintf(info, InfoLen,
                     "%*s %*s %*s %*s %*s %*s",
                     InfoSpacing, "M",
                     InfoSpacing, "N",
                     InfoSpacing, "PadA",
                     InfoSpacing, "NB",
                     InfoSpacing, "IB",
                     InfoSpacing, "Ortho.");
        }
        return;
    }
    // Return column values.
    // ortho. column appended later.
    snprintf(info, InfoLen,
             "%*d %*d %*d %*d %*d",
             InfoSpacing, param[PARAM_M].i,
             InfoSpacing, param[PARAM_N].i,
             InfoSpacing, param[PARAM_PADA].i,
             InfoSpacing, param[PARAM_NB].i,
             InfoSpacing, param[PARAM_IB].i);

    //================================================================
    // Set parameters.
    //================================================================
    int m = param[PARAM_M].i;
    int n = param[PARAM_N].i;

    int lda = imax(1, m + param[PARAM_PADA].i);

    int test = param[PARAM_TEST].c == 'y';
    double tol = param[PARAM_TOL].d * LAPACKE_dlamch('E');

    //================================================================
    // Set tuning parameters.
    //================================================================
    plasma_set(PlasmaNb, param[PARAM_NB].i);
    plasma_set(PlasmaIb, param[PARAM_IB].i);

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

    plasma_complex64_t *Aref = NULL;
    if (test) {
        Aref = (plasma_complex64_t*)malloc(
            (size_t)lda*n*sizeof(plasma_complex64_t));
        assert(Aref != NULL);

        memcpy(Aref, A, (size_t)lda*n*sizeof(plasma_complex64_t));
    }

    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    // Initialize tile matrix descriptor for matrix T
    // using multiples of tile size.
    int nb = plasma->nb;
    int ib = plasma->ib;
    int mt = (m%nb == 0) ? (m/nb) : (m/nb+1);
    int nt = (n%nb == 0) ? (n/nb) : (n/nb+1);
    // nt should be doubled if tree-reduction QR is performed,
    // not implemented now
    plasma_desc_t descT = plasma_desc_init(PlasmaComplexDouble, ib, nb, ib*nb,
                                           mt*ib, nt*nb, 0, 0, mt*ib, nt*nb);
    // allocate memory for the matrix T
    retval = plasma_desc_mat_alloc(&descT);
    assert(retval == 0);

    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_time_t start = omp_get_wtime();
    PLASMA_zgeqrf(m, n, A, lda, &descT);
    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = flops_zgeqrf(m, n) / time / 1e9;

    //=================================================================
    // Test results by checking orthogonality of Q and precision of Q*R
    //=================================================================
    if (test) {
        // Check the orthogonality of Q
        plasma_complex64_t zzero =  0.0;
        plasma_complex64_t zone  =  1.0;
        double one  =  1.0;
        double mone = -1.0;
        int minmn = imin(m, n);

        // Allocate space for Q.
        int ldq = m;
        plasma_complex64_t *Q =
            (plasma_complex64_t *)malloc((size_t)ldq*minmn*
                                         sizeof(plasma_complex64_t));

        // Build Q.
        PLASMA_zungqr(m, minmn, minmn, A, lda, &descT, Q, ldq);

        // Build the identity matrix
        plasma_complex64_t *Id =
            (plasma_complex64_t *) malloc((size_t)minmn*minmn*
                                          sizeof(plasma_complex64_t));
        LAPACKE_zlaset_work(LAPACK_COL_MAJOR, 'g', minmn, minmn,
                            zzero, zone, Id, minmn);

        // Perform Id - Q^H * Q
        cblas_zherk(CblasColMajor, CblasUpper, CblasConjTrans, minmn, m,
                    mone, Q, ldq, one, Id, minmn);

        // WORK array of size m is needed for computing L_oo norm
        double *WORK = (double *) malloc((size_t)m*sizeof(double));

        // |Id - Q^H * Q|_oo
        double ortho = LAPACKE_zlanhe_work(LAPACK_COL_MAJOR, 'I', 'u',
                                           minmn, Id, minmn, WORK);

        // normalize the result
        // |Id - Q^H * Q|_oo / n
        ortho /= minmn;

        free(Q);
        free(Id);

        // Check the accuracy of A - Q * R
        // LAPACK version does not construct Q, it uses only application of it

        // Extract the R.
        plasma_complex64_t *R =
            (plasma_complex64_t *)malloc((size_t)m*n*
                                         sizeof(plasma_complex64_t));
        LAPACKE_zlaset_work(LAPACK_COL_MAJOR, 'l', m, n,
                            zzero, zzero, R, m);
        LAPACKE_zlacpy_work(LAPACK_COL_MAJOR, 'u', m, n, A, lda, R, m);

        // Compute Q * R.
        PLASMA_zunmqr(PlasmaLeft, PlasmaNoTrans, m, n, minmn, A, lda, &descT,
                      R, m);

        // Compute the difference.
        // R = A - Q*R
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                R[j*m+i] = Aref[j*lda+i] - R[j*m+i];

        // |A|_oo
        double normA = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'I', m, n,
                                           Aref, lda, WORK);

        // |A - Q*R|_oo
        double error = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'I', m, n,
                                               R, m, WORK);

        // normalize the result
        // |A-QR|_oo / (|A|_oo * n)
        error /= (normA * n);

        param[PARAM_ERROR].d = error;
        param[PARAM_ORTHO].d = ortho;
        param[PARAM_SUCCESS].i = (error < tol && ortho < tol);

        free(WORK);
        free(R);

        // Return ortho. column value.
        int len = strlen(info);
        snprintf(&info[len], imax(0, InfoLen - len),
                 " %*.2e",
                 InfoSpacing, param[PARAM_ORTHO].d);
    }
    else {
        // No ortho. test.
        int len = strlen(info);
        snprintf(&info[len], imax(0, InfoLen - len),
                 " %*s",
                 InfoSpacing, "---");
    }

    //================================================================
    // Free arrays.
    //================================================================
    plasma_desc_mat_free(&descT);
    free(A);
    if (test)
        free(Aref);
}
