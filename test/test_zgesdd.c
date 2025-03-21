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
#include <math.h>
#include <omp.h>

#define COMPLEX

/***************************************************************************//**
 *
 * @brief Tests zgesdd.
 *
 * @param[in,out] param - array of parameters
 * @param[in]     run - whether to run test
 *
 * Sets flags in param indicating which parameters are used.
 * If run is true, also runs test and stores output parameters.
 ******************************************************************************/
void test_zgesdd(param_value_t param[], bool run)
{
    //================================================================
    // Mark which parameters are used.
    //================================================================
    param[PARAM_JOB   ].used = true;
    param[PARAM_DIM   ].used = PARAM_USE_M | PARAM_USE_N;
    param[PARAM_PADA  ].used = true;
    param[PARAM_NB    ].used = true;
    param[PARAM_IB    ].used = true;
    param[PARAM_ERROR2].used = true;
    param[PARAM_ORTHO_U].used = true;
    param[PARAM_ORTHO_V].used = true;
    if (! run)
        return;

    //================================================================
    // Set parameters.
    //================================================================
    plasma_enum_t job = plasma_job_const(param[PARAM_JOB].c);
    if (job != PlasmaNoVec && job != PlasmaAllVec && job != PlasmaSomeVec) {
        plasma_error("Illegal job value");
        return;
    }
    int m = param[PARAM_DIM].dim.m;
    int n = param[PARAM_DIM].dim.n;

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
    plasma_complex64_t *A = (plasma_complex64_t*)
        malloc((size_t)lda*n*sizeof(plasma_complex64_t));
    assert(A != NULL);

    int seed[] = {0, 0, 0, 1};
    double *Sigma_ref = NULL;
    plasma_complex64_t *Aref = NULL;

    if (test) {
        // Allocate memory for the reference singular values vector.
        // Only the Sigma_ref test requires using latms; orthogonality
        // and backwards error tests work for random matrices (larnv).
        Sigma_ref = (double*) malloc(imin(m,n)*sizeof(double));
        assert(Sigma_ref != NULL);

        int mode = 4;
        double cond = (double) imin(m,n);
        double dmax = 1.0;
        plasma_complex64_t *work = (plasma_complex64_t*)
            malloc(3*imax(m, n)* sizeof(plasma_complex64_t));
        assert(work != NULL);

        // Initialize A with specific singular values
        LAPACKE_zlatms_work(LAPACK_COL_MAJOR, m, n,
                            'U', seed, 'N', Sigma_ref, mode, cond,
                            dmax, m, n,'N', A, lda, work);
        free(work);

        // Make a copy of A for backwards error test.
        Aref = (plasma_complex64_t*)
            malloc((size_t)lda*n*sizeof(plasma_complex64_t));
        assert(Aref != NULL);
        memcpy(Aref, A, (size_t)lda*n*sizeof(plasma_complex64_t));
    }
    else {
        LAPACKE_zlarnv(1, seed, (size_t)lda*n, A);
    }

    //================================================================
    // Prepare the descriptor for matrix T.
    //================================================================
    plasma_desc_t T;

    //================================================================
    // Allocate the array of singular values
    //================================================================
    int minmn = imin(m,n);
    double *Sigma = (double*) malloc((size_t)minmn*sizeof(double));
    assert(Sigma != NULL);

    //================================================================
    // Allocate U and VT
    //================================================================
    plasma_complex64_t *U = NULL;
    plasma_complex64_t *VT = NULL;
    int Un   = 0;
    int VTm  = 0;
    int ldu  = 1;
    int ldvt = 1;

    if (job == PlasmaAllVec || job == PlasmaSomeVec) {
        Un  = (job == PlasmaAllVec ? m : minmn);
        ldu = m;
        U = (plasma_complex64_t*)
            malloc((size_t)ldu * Un * sizeof(plasma_complex64_t));
        assert(U != NULL);

        VTm  = (job == PlasmaAllVec ? n : minmn);
        ldvt = VTm;
        VT = (plasma_complex64_t*)
            malloc((size_t)ldvt * n * sizeof(plasma_complex64_t));
        assert(VT != NULL);
    }

    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_time_t start = omp_get_wtime();
    int info = plasma_zgesdd(job, job, m, n,
                             A, lda, &T, Sigma, U, ldu, VT, ldvt);
    assert(info == 0);
    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop - start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = 0.0 / time / 1e9;

    //=================================================================
    // Test results by checking orthogonality of U, VT
    // and the backward error ||A - U Sigma VT||
    // See LAWN 41, section 7.8.5
    //=================================================================
    if (test) {
        // Use -0.0 as flag to denote error wasn't checked.
        // TODO: use nan flag to denote "no check", as in libtest.
        param[PARAM_ERROR].d   = -0.0;
        param[PARAM_ERROR2].d  = -0.0;
        param[PARAM_ORTHO_U].d = -0.0;
        param[PARAM_ORTHO_V].d = -0.0;

        // Check the correctness of the singular values.
        // Should satisfy abs. error. See eqn (3) in Demmel et al., Computing
        // the singular value decomposition with high relative accuracy, 1999.
        double Smax = Sigma_ref[0];
        double error_sval = 0;
        for (int i = 0; i < minmn; i++) {
            double err = fabs(Sigma[i] - Sigma_ref[i]) / Smax;
            if (err > error_sval || isnan(err))
                error_sval = err;
        }
        param[PARAM_ERROR2].d = error_sval;
        param[PARAM_SUCCESS].i = (error_sval < tol);

        // work array for LAPACK is needed for computing norm
        double *work = (double*) malloc((size_t)imax(m,n)*sizeof(double));
        assert(work != NULL);

        if (job == PlasmaAllVec || job == PlasmaSomeVec) {
            //================================
            // Checking the othorgonality of U
            //================================
            // U is m-by-Un (m-by-m or m-by-n).
            // Build the Un-by-Un identity matrix
            int ldi = Un;
            plasma_complex64_t *Id = (plasma_complex64_t *)
                malloc((size_t)ldi * Un * sizeof(plasma_complex64_t));
            assert(Id != NULL);
            LAPACKE_zlaset_work(LAPACK_COL_MAJOR, 'g', Un, Un,
                                0.0, 1.0, Id, ldi);

            // Perform Id - U^H * U
            cblas_zherk(CblasColMajor, CblasUpper, CblasConjTrans, Un, m,
                        -1.0, U, ldu, 1.0, Id, ldi);

            // |Id - U^H * U|_oo
            double orthoU = LAPACKE_zlanhe_work(LAPACK_COL_MAJOR, 'I', 'U',
                                                Un, Id, ldi, work);

            // normalize the result
            // |Id - U^H * U|_oo / m
            orthoU /= m;
            free(Id);
            param[PARAM_ORTHO_U].d = orthoU;
            param[PARAM_SUCCESS].i = param[PARAM_SUCCESS].i && (orthoU < tol);

            //==============================
            // Check the orthogonality of VT
            //==============================
            // VT is VTm-by-n (n-by-n or m-by-n)
            // Build the VTm-by-VTm identity matrix
            ldi = VTm;
            Id = (plasma_complex64_t *)
                malloc((size_t)ldi * VTm * sizeof(plasma_complex64_t));
            assert(Id != NULL);
            LAPACKE_zlaset_work(LAPACK_COL_MAJOR, 'g', VTm, VTm,
                                0.0, 1.0, Id, ldi);

            // Perform Id - VT * VT^H
            cblas_zherk(CblasColMajor, CblasUpper, CblasNoTrans, VTm, n,
                        -1.0, VT, ldvt, 1.0, Id, ldi);

            // |Id - VT * VT^H|_oo
            double orthoVT = LAPACKE_zlanhe_work(LAPACK_COL_MAJOR, 'I', 'U',
                                                 VTm, Id, ldi, work);

            // normalize the result
            // |Id - VT * VT^H|_oo / n
            orthoVT /= n;
            free(Id);
            param[PARAM_ORTHO_V].d = orthoVT;
            param[PARAM_SUCCESS].i = param[PARAM_SUCCESS].i && (orthoVT < tol);

            //==============================
            // Check the backward error
            // ||A - U Sigma V^H|| / (||A|| min(m, n))
            //==============================
            double Anorm = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'I',
                                               m, n, Aref, lda, work);

            // U = U Sigma
            for (int j = 0; j < minmn; j++) {
                cblas_zdscal(m, Sigma[j], &U[(size_t)ldu*j], 1);
            }

            // A = A - (U Sigma) VT
            // Uses only first min(m, n) columns of U and rows of V^H,
            // since the remaining columns/rows are zeroed by Sigma.
            plasma_complex64_t zone  =  1.0;
            plasma_complex64_t zmone = -1.0;
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, minmn,
                        CBLAS_SADDR(zmone), U, ldu, VT, ldvt,
                        CBLAS_SADDR(zone), Aref, lda);

            // ||A - U Sigma VT||
            double error = LAPACKE_zlange_work(LAPACK_COL_MAJOR, 'I',
                                               m, n, Aref, lda, work);
            // normalize the result
            // ||A - U Sigma VT|| / (||A|| min(m, n))
            error /= (Anorm * imin(m, n));
            param[PARAM_ERROR].d = error;
            param[PARAM_SUCCESS].i = param[PARAM_SUCCESS].i && (error < tol);
        }
        free(work);
    }

    //================================================================
    // Free arrays.
    // If an array wasn't allocated, it's NULL, so free does nothing.
    //================================================================
    free(A);
    free(Aref);
    free(Sigma);
    free(Sigma_ref);
    free(U);
    free(VT);
    plasma_desc_destroy(&T);
}
