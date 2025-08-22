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

// pztbbrd is internal, but including plasma_internal.h conflicts with test.h
// since both define imin, imax.
void plasma_pztbbrd_static(
    plasma_enum_t uplo, int n, int nb, int Vblksiz,
    plasma_complex64_t *A, int lda,
    plasma_complex64_t *VQ, plasma_complex64_t *tauQ,
    plasma_complex64_t *VP, plasma_complex64_t *tauP,
    double *D, double *E, int wantz,
    plasma_workspace_t work,
    plasma_sequence_t *sequence, plasma_request_t *request);

#define Aband_lower( i_, j_ ) (Aband + nb + ldab*(j_) + ((i_) - (j_)))
#define Aband_upper( i_, j_ ) (Aband + nb + ldab*(j_) + ((i_) - (j_) + nb))

#define Aband( uplo_, i_, j_ ) \
    ((uplo_) == PlasmaLower \
        ? Aband_lower( (i_), (j_) ) \
        : Aband_upper( (i_), (j_) ))

#define DOUBLE
#define COMPLEX

/***************************************************************************//**
 *
 * @brief Tests ztbbrd.
 *
 * @param[in,out] param - array of parameters
 * @param[in]     run - whether to run test
 *
 * Sets flags in param indicating which parameters are used.
 * If run is true, also runs test and stores output parameters.
 ******************************************************************************/
void test_ztbbrd( param_value_t param[], bool run )
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
    plasma_enum_t uplo = plasma_uplo_const( param[PARAM_UPLO].c );

    int verbose = param[PARAM_VERBOSE].i;
    int n = param[PARAM_DIM].dim.n;
    int nb = param[PARAM_NB].i;

    int ldab    = 3*nb + 1;
    int Vblksiz = nb/4;
    //int ldt     = Vblksiz;
    int wantz   = 0;

    int    test = param[PARAM_TEST].c == 'y';
    double tol  = param[PARAM_TOL].d * LAPACKE_dlamch( 'E' );

    //================================================================
    // Allocate and initialize arrays.
    //================================================================
    int info;

    // Since we give (Aband + nb) to latms, allocate an extra nb here.
    plasma_complex64_t *Aband = (plasma_complex64_t *)
        calloc( (size_t) ldab*n + nb, sizeof( plasma_complex64_t ) );

    plasma_complex64_t *Aband_ref = NULL;
    plasma_complex64_t *Z    = NULL;
    double             *Sigma = (double*) malloc( (size_t) n*sizeof( double ) );
    double             *Sigma_ref = NULL;
    plasma_complex64_t *work = NULL;
    int seed[] = {0, 0, 0, 1};

    #if defined(REAL) && defined(SINGLE) && defined(PLASMA_WITH_MKL)
        // slatms crashes with MKL 2024.2.0, so disabling checking results.
        // [dcz]latms work fine.
        if (uplo == PlasmaUpper) {
            static bool first_warn = true;
            if (first_warn) {
                printf( "%% Skipping checks for uplo=upper, due to apparent bug in MKL slatms, seen in 2024.2.0.\n" );
                first_warn = false;
            }
            test = false;
        }
    #endif

    if (test) {
        Sigma_ref = (double*) malloc( (size_t) n*sizeof( double ) );
        work = (plasma_complex64_t *)
            malloc( (size_t) 3*n*sizeof( plasma_complex64_t ) );

        // Random singular values.
        LAPACKE_dlarnv( 1, seed, n, Sigma_ref );

        // dist='S' for symmetric: uniform on [-1, 1]
        // sym='N' for non-symmetric matrix,
        // pack='Z' for general band.
        int    mode  = 0;       // D = Sigma_ref is input
        double dmax  = 1.0;     // unused for mode = 0
        double rcond = 1.0e6;   // unused for mode = 0
        int kl, ku;
        if (uplo == PlasmaLower) {
            kl = nb;
            ku = 0;
        }
        else {
            kl = 0;
            ku = nb;
        }
        info = LAPACKE_zlatms_work(
            LAPACK_COL_MAJOR, n, n,
            'S', seed,
            'N', Sigma_ref, mode, rcond,
            dmax, kl, ku,
            'Z', Aband + nb, ldab, work);
        assert( info == 0 );

        #ifdef COMPLEX
            // Use reflector to make A( 0, 0 ) real.
            // @todo To test vectors, need to save reflector.
            // he2hb actually makes the whole diag real, but hbbrd needs only
            // the first entry to be real.
            plasma_complex64_t dummy, tau;
            LAPACKE_zlarfg( 1, Aband( uplo, 0, 0 ), &dummy, 1, &tau );
        #endif

        // Sort the singular values decreasing.
        LAPACKE_dlasrt_work( 'D', n, Sigma_ref );

        // Copy Aband into Aband_ref
        Aband_ref = (plasma_complex64_t *)
            malloc( (size_t) ldab*n*sizeof( plasma_complex64_t ) );
        LAPACKE_zlacpy_work( LAPACK_COL_MAJOR,
                             'A', ldab, n, Aband, ldab, Aband_ref, ldab );
    }
    else {
        LAPACKE_zlarnv( 1, seed, (size_t) ldab*n, Aband );
    }

    if (verbose) {
        plasma_dprint_vector( "Sigma", n, Sigma_ref, 1 );
        plasma_zprint_matrix( "Aband", ldab, n, Aband, ldab );
    }

    // Sizes for job == NoVectors.
    plasma_complex64_t* tauQ = (plasma_complex64_t*)
        calloc( (size_t) 2*n, sizeof( plasma_complex64_t ) );
    plasma_complex64_t* VQ   = (plasma_complex64_t*)
        calloc( (size_t) 2*n, sizeof( plasma_complex64_t ) );
    plasma_complex64_t* tauP = (plasma_complex64_t*)
        calloc( (size_t) 2*n, sizeof( plasma_complex64_t ) );
    plasma_complex64_t* VP   = (plasma_complex64_t*)
        calloc( (size_t) 2*n, sizeof( plasma_complex64_t ) );
    double* E = (double*) calloc( (size_t) n, sizeof( double ) );

    //================================================================
    // Setup PLASMA internal structures.
    //================================================================
    // Allocate workspace.
    plasma_workspace_t workspace;
    size_t lwork = nb;  // tbbrd needs size n.
    info = plasma_workspace_create( &workspace, lwork, PlasmaComplexDouble );
    assert( info == 0 );

    // Initialize sequence.
    plasma_sequence_t sequence;
    info = plasma_sequence_init( &sequence );
    assert( info == 0 );

    // Initialize request.
    plasma_request_t request;
    info = plasma_request_init( &request );
    assert( info == 0 );

    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_time_t start = omp_get_wtime();

    plasma_pztbbrd_static(
        uplo, n, nb, Vblksiz,
        Aband, ldab, VQ, tauQ, VP, tauP, Sigma, E, wantz, workspace,
        &sequence, &request );
    if (sequence.status != 0) {
        printf( "Failed: status %d\n", sequence.status );
    }

    if (verbose) {
        plasma_zprint_matrix( "Aband_out", ldab, n, Aband, ldab );
        plasma_dprint_vector( "D", n, Sigma, 1 );
        plasma_dprint_vector( "E", n, E,     1 );
    }

    plasma_complex64_t dummy;
    info = LAPACKE_zbdsqr( LAPACK_COL_MAJOR, lapack_const( uplo ), n, 0, 0, 0,
                           Sigma, E, &dummy, 1, &dummy, 1, &dummy, 1 );
    assert( info == 0 );

    if (verbose) {
        plasma_dprint_vector( "Sigma",     n, Sigma,     1 );
        plasma_dprint_vector( "Sigma_ref", n, Sigma_ref, 1 );
    }

    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop - start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = 0.0 / time / 1e9;

    if (test) {
        // Check the correctness of the singular values.
        double error = 0;
        for (int i = 0; i < n; ++i) {
            error += fabs( Sigma[i] - Sigma_ref[i] )
                     / fabs( Sigma_ref[i] );
        }

        error /= n*40;

        // // Othorgonality test
        // double r_one = 1.0;
        //
        // // Build the idendity matrix
        // plasma_complex64_t *Id = (plasma_complex64_t *)
        //     malloc( n*n*sizeof( plasma_complex64_t ) );
        // LAPACKE_zlaset_work( LAPACK_COL_MAJOR, 'A', n, n, 0., 1., Id, n );
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
    // plasma_desc_destroy( &T );
    free( Aband );
    free( Z );
    free( Sigma );
    free( work );
    if (test) {
        free( Aband_ref );
        free( Sigma_ref );
    }
}
