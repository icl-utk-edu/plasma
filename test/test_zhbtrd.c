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
#include "plasma_core_blas.h"
#include "core_lapack.h"
#include "plasma.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// pzhbtrd is internal, but including plasma_internal.h conflicts with test.h
// since both define imin, imax.
void plasma_pzhbtrd_static(
    plasma_enum_t uplo, int n, int nb, int Vblksiz,
    plasma_complex64_t *A, int lda,
    plasma_complex64_t *V, plasma_complex64_t *tau,
    double *d, double *e, int wantz,
    plasma_workspace_t work,
    plasma_sequence_t *sequence, plasma_request_t *request);

#undef  REAL
#define COMPLEX

/***************************************************************************//**
 *
 * @brief Tests zhbtrd.
 *
 * @param[in,out] param - array of parameters
 * @param[in]     run - whether to run test
 *
 * Sets flags in param indicating which parameters are used.
 * If run is true, also runs test and stores output parameters.
 ******************************************************************************/
void test_zhbtrd(param_value_t param[], bool run)
{
    //================================================================
    // Mark which parameters are used.
    //================================================================
    param[PARAM_UPLO  ].used = true;
    param[PARAM_DIM   ].used = PARAM_USE_N;
    param[PARAM_NB    ].used = true;
    if (! run)
        return;

    //================================================================
    // Set parameters.
    //================================================================
    plasma_enum_t uplo = plasma_uplo_const(param[PARAM_UPLO].c);
    if (uplo != PlasmaLower) {
        printf( "%s: skipping; only Lower currently supported\n", __func__ );
        return;
    }

    int verbose = param[PARAM_VERBOSE].i;
    int n = param[PARAM_DIM].dim.n;
    int nb = param[PARAM_NB].i;

    int ldab    = 2*nb + 1;
    int Vblksiz = nb/4;
    //int ldt     = Vblksiz;
    int wantz   = 0;

    int    test = param[PARAM_TEST].c == 'y';
    double tol  = param[PARAM_TOL].d * LAPACKE_dlamch('E');

    //================================================================
    // Allocate and initialize arrays.
    //================================================================
    int info;

    plasma_complex64_t *Aband = (plasma_complex64_t *)
        malloc((size_t)ldab*n*sizeof(plasma_complex64_t));

    plasma_complex64_t *Aband_ref = NULL;
    plasma_complex64_t *Z    = NULL;
    double             *Lambda = (double*)malloc((size_t)n*sizeof(double));
    double             *Lambda_ref = NULL;
    plasma_complex64_t *work = NULL;
    int seed[] = {0, 0, 0, 1};
    if (test) {
        Lambda_ref = (double*)malloc((size_t)n*sizeof(double));
        work = (plasma_complex64_t *)
            malloc((size_t)3*n*sizeof(plasma_complex64_t));

        // todo: solve harder (at least random) eigenvalue distributions.
        for (int i = 0; i < n; ++i) {
            Lambda_ref[i] = i + 1;
        }

        // sym='H' for Hermitian, pack='B' for band, with kl = ku = nb.
        int    mode  = 0;
        double dmax  = 1.0;
        double rcond = 1.0e6;
        LAPACKE_zlatms_work(LAPACK_COL_MAJOR, n, n,
                           'S', seed,
                           'H', Lambda_ref, mode, rcond,
                            dmax, nb, nb,
                           'B', Aband, ldab, work);

        // Sort the eigenvalues increasing.
        LAPACKE_dlasrt_work( 'I', n, Lambda_ref );

        // Copy Aband into Aband_ref
        Aband_ref = (plasma_complex64_t *)
            malloc((size_t)ldab*n*sizeof(plasma_complex64_t));
        LAPACKE_zlacpy_work(LAPACK_COL_MAJOR,
                            'A', ldab, n, Aband, ldab, Aband_ref, ldab);
    }
    else {
        LAPACKE_zlarnv(1, seed, (size_t)ldab*n, Aband);
    }

    if (verbose) {
        plasma_zprint_matrix( "Aband", ldab, n, Aband, ldab );
    }

    // Sizes for job == NoVectors.
    plasma_complex64_t* tau
        = (plasma_complex64_t*) calloc((size_t)2*n, sizeof(plasma_complex64_t));
    plasma_complex64_t* V
        = (plasma_complex64_t*) calloc((size_t)2*n, sizeof(plasma_complex64_t));
    double* E
        = (double*) calloc((size_t)n, sizeof(double));

    //================================================================
    // Setup PLASMA internal structures.
    //================================================================
    // Allocate workspace.
    plasma_workspace_t workspace;
    size_t lwork = nb;  // hbtrd needs size n.
    info = plasma_workspace_create(&workspace, lwork, PlasmaComplexDouble);
    assert( info == 0 );

    // Initialize sequence.
    plasma_sequence_t sequence;
    info = plasma_sequence_init(&sequence);
    assert( info == 0 );

    // Initialize request.
    plasma_request_t request;
    info = plasma_request_init(&request);
    assert( info == 0 );

    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_time_t start = omp_get_wtime();

    plasma_pzhbtrd_static(
        uplo, n, nb, Vblksiz,
        Aband, ldab, V, tau, Lambda, E, wantz, workspace,
        &sequence, &request );
    if (sequence.status != 0) {
        printf( "Failed: status %d\n", sequence.status );
    }

    if (verbose) {
        plasma_zprint_matrix( "Aband_out", ldab, n, Aband, ldab );
        plasma_dprint_vector( "D", n, Lambda, 1 );
        plasma_dprint_vector( "E", n, E,      1 );
    }

    //LAPACKE_zhbtrd( LAPACK_COL_MAJOR,
    //               'N', 'L',  n, Aband, lda, Lambda);

    double dummy;
    info = LAPACKE_dstev( LAPACK_COL_MAJOR, 'N', n, Lambda, E, &dummy, 1 );
    assert( info == 0 );

    if (verbose) {
        plasma_dprint_vector( "Lambda",     n, Lambda,     1 );
        plasma_dprint_vector( "Lambda_ref", n, Lambda_ref, 1 );
    }

    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop - start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = 0.0 / time / 1e9;

    if (test) {
        // Check the correctness of the eigenvalues values.
        double error = 0;
        for (int i = 0; i < n; ++i) {
            error += fabs( Lambda[i] - Lambda_ref[i] )
                     / fabs( Lambda_ref[i] );
        }

        error /= n*40;

        // // Othorgonality test
        // double r_one = 1.0;
        //
        // // Build the idendity matrix
        // plasma_complex64_t *Id = (plasma_complex64_t *)
        //     malloc(n*n*sizeof(plasma_complex64_t));
        // LAPACKE_zlaset_work(LAPACK_COL_MAJOR, 'A', n, n, 0., 1., Id, n);
        //
        // double ortho = 0.;
        // if (job == PlasmaVec) {
        //     // Perform Id - Q^H Q
        //     cblas_zherk(
        //         CblasColMajor, CblasUpper, CblasConjTrans,
        //         n, n, r_one, Q, n, -r_one, Id, n);
        //     double normQ = LAPACKE_zlanhe_work(
        //         LAPACK_COL_MAJOR, 'I', 'U', n, Id, n, (double*)work);
        //     ortho = normQ/n;
        // }
        param[PARAM_ERROR].d = error;
        //param[PARAM_ORTHO].d = ortho;
        param[PARAM_SUCCESS].i = (error < tol); // && ortho < tol);
    }

    //================================================================
    // Free arrays.
    //================================================================
    // plasma_desc_destroy(&T);
    free(Aband);
    free(Z);
    free(Lambda);
    free(work);
    if (test) {
        free(Aband_ref);
        free(Lambda_ref);
    }
}
