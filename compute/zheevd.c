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

#include "plasma.h"
#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_tuning.h"
#include "plasma_types.h"
#include "plasma_workspace.h"
#include "core_lapack.h"
#include "bulge.h"

#include <string.h>
#include <omp.h>

/***************************************************************************//**
 *
 * @ingroup plasma_heevd
 *
 *  Computes all eigenvalues and, optionally,
 *  eigenvectors of a complex Hermitian matrix A. The matrix A is
 *  preliminary reduced to tridiagonal form using a two-stage
 *  approach:
 *  First stage: reduction to band tridiagonal form;
 *  Second stage: reduction from band to tridiagonal form.
 *
 *******************************************************************************
 *
 * @param[in] job
 *          Specifies whether to compute eigenvectors.
 *          - PlasmaNoVec: computes eigenvalues only;
 *          - PlasmaVec:   computes eigenvalues and eigenvectors.
 *
 * @param[in] uplo
 *          Specifies whether the matrix A is upper triangular or
 *          lower triangular:
 *          - PlasmaUpper: Upper triangle of A is stored;
 *          - PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in,out] pA
 *          On entry, the Hermitian matrix pA.
 *          If uplo = PlasmaUpper, the leading n-by-n upper triangular
 *          part of pA contains the upper triangular part of the matrix
 *          A, and the strictly lower triangular part of pA is not
 *          referenced.
 *          If uplo = PlasmaLower, the leading n-by-n lower triangular
 *          part of A contains the lower triangular part of the matrix
 *          A, and the strictly upper triangular part of pA is not
 *          referenced.
 *          On exit, the lower triangle (if uplo = PlasmaLower) or the
 *          upper triangle (if uplo = PlasmaUpper) of A, including the
 *          diagonal, is destroyed.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,n).
 *
 * @param[out] Lambda
 *          On exit, if info = 0, the eigenvalues.
 *
 * @param[in, out] T
 *          On exit, auxiliary factorization data, required by plasma_zheevd.
 *          Matrix in T is allocated inside this function and needs to be
 *          destroyed by plasma_desc_destroy.
 *          @todo Shouldn't heevd destroy T? Why is this an argument?
 *
 * @param[out] pQ
 *          On exit, if job = PlasmaVec and info = 0, the eigenvectors.
 *
 * @param[in] ldq
 *          The leading dimension of the array pQ. ldq >= max(1,n).
 *
 *******************************************************************************
 *
 * @retval PlasmaSuccess successful exit
 * @retval < 0 if -i, the i-th argument had an illegal value
 *
 *******************************************************************************
 *
 * @sa plasma_zheevd
 * @sa plasma_cheevd
 * @sa plasma_dheevd
 * @sa plasma_sheevd
 *
 ******************************************************************************/
int plasma_zheevd(
    plasma_enum_t job, plasma_enum_t uplo, int n,
    plasma_complex64_t *pA, int lda,
    plasma_desc_t *T,
    double *Lambda,
    plasma_complex64_t *pQ, int ldq)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    // Check input arguments
    if (job != PlasmaNoVec && job != PlasmaVec) {
        plasma_error("illegal value of job");
        return -1;
    }
    if (uplo != PlasmaLower && uplo != PlasmaUpper) {
        plasma_error("illegal value of uplo");
        return -2;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -3;
    }
    if (lda < imax(1, n)) {
        plasma_error("illegal value of lda");
        return -5;
    }
    if (ldq < imax(1, n)) {
        plasma_error("illegal value of ldq");
        return -9;
    }

    // Quick return
    if (n == 0)
        return PlasmaSuccess;

    // Set tiling parameters.
    int ib = plasma->ib;
    int nb = plasma->nb;

    // Create tile matrix.
    plasma_desc_t A;
    int retval;
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        n, n, 0, 0, n, n, &A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        return retval;
    }

    // Prepare descriptor T.
    retval = plasma_descT_create(A, ib, PlasmaFlatHouseholder, T);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_descT_create() failed");
        return retval;
    }

    // Allocate workspace.
    plasma_workspace_t work;
    size_t lwork = ib*nb + 4*nb*nb;  // geqrt: tau + work
    retval = plasma_workspace_create(&work, lwork, PlasmaComplexDouble);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_workspace_create() failed");
        return retval;
    }

    // Initialize sequence.
    plasma_sequence_t sequence;
    retval = plasma_sequence_init(&sequence);

    // Initialize request.
    plasma_request_t request;
    retval = plasma_request_init(&request);

    // asynchronous block
    #pragma omp parallel
    #pragma omp master
    {
        // Translate to tile layout.
        plasma_omp_zge2desc(pA, lda, A, &sequence, &request);
    }

    // Warning !!! plasma_omp_zheevd is not a fully async function.
    // It contains both async and sync functions.
    plasma_omp_zheevd(job, uplo, A, *T, Lambda, pQ, ldq, work,
                      &sequence, &request);

    #pragma omp parallel
    #pragma omp master
    {
        // Translate back to LAPACK layout.
        plasma_omp_zdesc2ge(A, pA, lda, &sequence, &request);
    }

    plasma_workspace_destroy(&work);

    // Free matrix A in tile layout.
    plasma_desc_destroy(&A);

    // Return status.
    return sequence.status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_heevd
 *
 *  Computes all eigenvalues and,
 *  optionally, eigenvectors of a complex Hermitian matrix A using a
 *  two-stage approach:
 *  First stage: reduction to band tridiagonal form;
 *  Second stage: reduction from band to tridiagonal form.
 *
 *  May return before the computation is finished.
 *  Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] job
 *          Specifies whether to compute eigenvectors.
 *          - PlasmaNoVec: computes eigenvalues only;
 *          - PlasmaVec:   computes eigenvalues and eigenvectors.
 *
 * @param[in] uplo
 *          Specifies whether the matrix A is upper triangular or
 *          lower triangular:
 *          - PlasmaUpper: Upper triangle of A is stored;
 *          - PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in,out] A
 *          Descriptor of matrix A.
 *          A is stored in the tile layout.
 *
 * @param[out] Lambda
 *          On exit, if info = 0, the eigenvalues.
 *
 * @param[out] T
 *          Descriptor of matrix T.
 *          On exit, auxiliary factorization data, required by QR factorization
 *          auxiliary kernels to apply Q.
 *
 * @param[out] Q
 *          On exit, if job = PlasmaVec and info = 0, the eigenvectors.
 *
 * @param[in] ldq
 *          The leading dimention of the eigenvectors matrix Q. ldq >= max(1,n).
 *
 * @param[in] sequence
 *          Identifies the sequence of function calls that this call belongs to
 *          (for completion checks and exception handling purposes).
 *
 * @param[out] request
 *          Identifies this function call (for exception handling purposes).
 *
 *******************************************************************************
 *
 * @sa plasma_zheevd
 * @sa plasma_omp_cheevd
 * @sa plasma_omp_dsyev
 * @sa plasma_omp_ssyev
 *
 ******************************************************************************/
void plasma_omp_zheevd(
    plasma_enum_t job, plasma_enum_t uplo,
    plasma_desc_t A, plasma_desc_t T,
    double *Lambda,
    plasma_complex64_t *pQ, int ldq,
    plasma_workspace_t work,
    plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }

    // Check input arguments.
    if (job != PlasmaNoVec && job != PlasmaVec) {
        plasma_error("illegal value of job");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (uplo != PlasmaLower && uplo != PlasmaUpper) {
        plasma_error("illegal value of uplo");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (plasma_desc_check(A) != PlasmaSuccess) {
        plasma_error("invalid A");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (plasma_desc_check(T) != PlasmaSuccess) {
        plasma_error("invalid T");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (sequence == NULL) {
        plasma_fatal_error("NULL sequence");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (request == NULL) {
        plasma_fatal_error("NULL request");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }

    // quick return
    if (imin(A.m, A.n) == 0)
        return;

    int n  = A.m;
    int nb   = imin(A.mb, A.m);
    int lda_band = 2*nb + 1;

    //Allocate workspace for band storage of the band matrix
    // A and for the off diagonal after tridiagonalisation
    plasma_complex64_t *A_band =
        (plasma_complex64_t *)calloc((size_t)lda_band*n, sizeof(plasma_complex64_t));
    memset( A_band, 0, lda_band*n*sizeof(plasma_complex64_t) );
    if (A_band == NULL) {
        plasma_error("memory allocation(A_band) failed");
        free(A_band);
        return;
    }
    double *E = (double *)calloc((size_t)n, sizeof(double));
    if (E == NULL) {
        plasma_error("malloc(E) failed");
        free(E);
        return;
    }

    //===================
    // Reduction to band
    //===================
    double start = omp_get_wtime();
    #pragma omp parallel
    #pragma omp master
    {
        plasma_pzhe2hb(uplo,
                       A, T,
                       work,
                       sequence, request);

        // Copy tile band to lapack band
        plasma_pzhecpy_tile2lapack_band (uplo,
                                         A,
                                         A_band, lda_band,
                                         sequence, request);
    }
    double stop = omp_get_wtime();
    double time = stop - start;
    //printf("\n n=%d:  1-stage time = %lf\t", n, time);

    //====================
    //  Bulge chasing
    //====================
    plasma_complex64_t *TAU2 = NULL;
    plasma_complex64_t *V2 = NULL;
    plasma_complex64_t *T2 = NULL;
    int Vblksiz;  // Blocking used when applying V2 to the matrix Q
    int blkcnt;   // Number of diamond tile or tile of Vs
    int ldt, ldv;
    int wantz   = 0;
    int blguplo = PlasmaLower;

    if (job == PlasmaNoVec)
        wantz = 0;
    else
        wantz = 2;

    Vblksiz = nb/4;
    ldt     = Vblksiz;
    if (job == PlasmaVec) {
        findVTsiz(n, nb, Vblksiz, &blkcnt, &ldv);
        TAU2 = (plasma_complex64_t *)
            calloc((size_t)blkcnt*Vblksiz, sizeof(plasma_complex64_t));
        V2  = (plasma_complex64_t *)
            calloc((size_t)ldv*blkcnt*Vblksiz, sizeof(plasma_complex64_t));
        T2  = (plasma_complex64_t *)
            calloc((size_t)ldt*blkcnt*Vblksiz, sizeof(plasma_complex64_t));
        if (TAU2 == NULL || V2 == NULL || T2 == NULL) {
            plasma_error("calloc() failed");
            free(TAU2);
            free(V2);
            free(T2);
            return;
        }
        memset(TAU2, 0,     blkcnt*Vblksiz*sizeof(plasma_complex64_t));
        memset(V2,   0, ldv*blkcnt*Vblksiz*sizeof(plasma_complex64_t));
        memset(T2,   0, ldt*blkcnt*Vblksiz*sizeof(plasma_complex64_t));
    }
    else {
        TAU2   = (plasma_complex64_t *)
            calloc((size_t)2*n, sizeof(plasma_complex64_t));
        V2     = (plasma_complex64_t *)
            calloc((size_t)2*n, sizeof(plasma_complex64_t ));
        if (TAU2 == NULL || V2 == NULL) {
            plasma_error("calloc() failed");
            free(TAU2);
            free(V2);
            return;
        }
        memset(TAU2, 0, 2*n*sizeof(plasma_complex64_t));
        memset(V2,   0, 2*n*sizeof(plasma_complex64_t));
    }

    // Main bulge chasing kernel.
    // Contains internal omp parallel section
    start = omp_get_wtime();
    plasma_pzhbtrd_static(blguplo, n, nb, Vblksiz,
                          A_band, lda_band,
                          V2, TAU2,
                          Lambda, E,
                          wantz,
                          work,
                          sequence, request);
    stop = omp_get_wtime();
    time = stop - start;
    //printf("2-stage timing = %lf\t", time);

    //=======================================
    // Tridiagonal eigensolver
    //=======================================
    // call eigensolver using lapack routine for our resulting tridiag,
    // [Lambda E]
    start = omp_get_wtime();
    if (job == PlasmaNoVec) {
        LAPACKE_zstedc( LAPACK_COL_MAJOR, 'N', n, Lambda, E, pQ, ldq );
    }
    else {
        LAPACKE_zstedc( LAPACK_COL_MAJOR, 'I', n, Lambda, E, pQ, ldq );
    }
    stop = omp_get_wtime();
    time = stop - start;
    //printf("Eigenvalue time = %lf\t", time);

    start = omp_get_wtime();
    if (job == PlasmaVec) {
        //=======================================
        // Apply Q2 from the bulge.
        //=======================================
        // compute T2
        #pragma omp parallel
        {
            plasma_pzlarft_blgtrd(n, nb, Vblksiz,
                                  V2, T2, TAU2,
                                  sequence, request);
        }

        // apply Q2 from Left
        #pragma omp parallel
        {
            plasma_pzunmqr_blgtrd(PlasmaLeft,  PlasmaNoTrans,
                                  n, nb, n,
                                  Vblksiz, wantz,
                                  V2, T2, TAU2,
                                  pQ, ldq,
                                  work,
                                  sequence, request);
        }

        //=======================================
        // Apply Q1 from the first stage .
        //=======================================
        // If nb > n, Q1 doesn't need to be applied,
        // only bulge chasing has been done
        if (nb < n) {
            plasma_desc_t Q;
            plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                       n, n, 0, 0, n, n, &Q);

            #pragma omp parallel
            #pragma omp master
            {
                // Translate to tile layout.
                plasma_pzge2desc(pQ, ldq, Q, sequence, request);

                // Accumulate the transformations from the first stage
                if (uplo == PlasmaLower) {
                    plasma_pzunmqr(PlasmaLeft, PlasmaNoTrans,
                                   plasma_desc_view(A, A.mb, 0, A.m - A.mb, A.n - A.nb),
                                   plasma_desc_view(T, T.mb, 0, T.m - T.mb, T.n - T.nb),
                                   plasma_desc_view(Q, Q.mb, 0, Q.m - Q.mb, Q.n),
                                   work,
                                   sequence, request);
                }
                else {
                    plasma_pzunmlq (PlasmaLeft, Plasma_ConjTrans,
                                    plasma_desc_view(A, 0, A.nb, A.m - A.mb, A.n - A.nb),
                                    plasma_desc_view(T, 0, T.nb, T.m - T.mb, T.n - T.nb),
                                    plasma_desc_view(Q, Q.mb, 0, Q.m - Q.mb, Q.n),
                                    work,
                                    sequence, request);
                }

                // Translate back to LAPACK layout.
                plasma_pzdesc2ge(Q, pQ, ldq, sequence, request);
            }

            plasma_desc_destroy(&Q);
        } // end (nb < n)
    }
    stop = omp_get_wtime();
    time = stop - start;
    //printf("Eigenvector timing = %lf\n", time);

    free(T2);
    free(V2);
    free(TAU2);
    free(E);
    free(A_band);
}
