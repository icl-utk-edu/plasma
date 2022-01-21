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
#include "core_lapack.h"
#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_types.h"
#include "plasma_workspace.h"
#include "bulge.h"

#include <omp.h>
#include <string.h>

#define COMPLEX

/***************************************************************************//**
 *
 * @ingroup plasma_gesdd
 *
 *  Computes the singular value decomposition (SVD) of a complex
 *  m-by-n matrix A, optionally computing the left and/or right singular
 *  vectors. The SVD is written
 *
 *    \f[ A = U \times \Sigma \times V^H \f],
 *
 *  where \f$ \Sigma \f$ is an m-by-n matrix which is zero except for its
 *  min(m, n) diagonal elements, U is an m-by-m unitary matrix, and
 *  V is an n-by-n unitary matrix. The diagonal elements of \f$ \Sigma \f$
 *  are the singular values of A; they are real and non-negative, and
 *  are returned in descending order. The first min(m, n) columns of
 *  U and V are the left and right singular vectors of A.
 *
 *  Note that the routine returns V^H, not V.
 *******************************************************************************
 *
 * @param[in] jobu
 *          Specifies options for computing all or part of the matrix U.
 *          - PlasmaAllVec:  all m columns of U are returned in the array U.
 *          - PlasmaSomeVec: the first min(m, n) columns of U (the left
 *                           singular vectors) are returned in the array U.
 *          - PlasmaNoVec:   no columns of U are computed.
 *
 * @param[in] jobvt
 *          Specifies options for computing all or part of the matrix V^H.
 *          - PlasmaAllVec:  all n rows of V^H are returned in the array VT;
 *          - PlasmaSomeVec: the first min(m, n) rows of V^H (the right
 *                           singular vectors) are returned in the array VT.
 *          - PlasmaNoVec:   no rows of V^H are computed.
 *
 *          NOTE: currently this requires jobu = jobvt.
 *
 * @param[in] m
 *          The number of rows of the matrix A. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix A. n >= 0.
 *
 * @param[in,out] pA
 *          On entry, pointer to the m-by-n matrix A.
 *          On exit, the contents of A are destroyed.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, m).
 *
 * @param[out] S
 *          The double precision singular values of A,
 *          sorted so that S(i) >= S(i + 1).
 *
 * @param[in,out] T
 *          On exit, contains auxiliary factorization data.
 *          Matrix in T is allocated inside this function and needs to be
 *          destroyed by plasma_desc_destroy.
 *
 * @param[out] U
 *          Pointer to the left singular vectors matrix U.
 *          - If jobu = PlasmaAllVec, U is ldu-by-m.
 *            On exit, U contains the m-by-m unitary matrix U.
 *          - If jobu = PlasmaSomeVec, U is ldu-by-min(m, n).
 *            On exit, U contains the m-by-min(m, n) unitary matrix U.
 *          - If jobu = PlasmaNoVec, U is not referenced.
 *
 * @param[in] ldu
 *          The leading dimension of the array U. ldu >= 1;
 *          if jobu = PlasmaAllVec or PlasmaSomeVec, ldu >= m.
 *
 * @param[out] VT
 *         Pointer to the right singular vectors matrix VT
 *         - If jobvt = PlasmaAllVec, VT is ldvt-by-n.
 *           On exit, VT contains the n-by-n unitary matrix V^H.
 *         - If jobvt = PlasmaSomeVec, VT is ldvt-by-n.
 *           On exit, VT contains the first min(m, n) rows of
 *           V^H (the right singular vectors, stored rowwise).
 *         - If jobvt = PlasmaNoVec, VT is not referenced.
 *
 * @param[in] ldvt
 *         The leading dimension of the array VT. ldvt >= 1;
 *         if jobvt = PlasmaAllVec, ldvt >= n;
 *         if jobvt = PlasmaSomeVec, ldvt >= min(m, n).
 *
 *******************************************************************************
 * @retval PlasmaSuccess successful exit
 *
 *******************************************************************************
 *
 * @sa PLASMA_omp_zgesdd
 * @sa PLASMA_cgesdd
 * @sa PLASMA_dgesdd
 * @sa PLASMA_sgesdd
 *
 ******************************************************************************/
int plasma_zgesdd(plasma_enum_t jobu, plasma_enum_t jobvt,
                  int m, int n,
                  plasma_complex64_t *pA, int lda,
                  plasma_desc_t *T,
                  double *S,
                  plasma_complex64_t *pU,  int ldu,
                  plasma_complex64_t *pVT, int ldvt)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    // Check input arguments.
    int minmn = imin(m, n);
    if (jobu != PlasmaNoVec && jobu != PlasmaAllVec && jobu != PlasmaSomeVec) {
        plasma_error("illegal value of jobu");
        return -1;
    }
    if (jobvt != PlasmaNoVec && jobvt != PlasmaAllVec && jobvt != PlasmaSomeVec) {
        plasma_error("illegal value of jobvt");
        return -2;
    }
    if (jobvt != jobu) {
        plasma_error("in this version: jobu should be equal jobvt");
        return -2;
    }
    if (m < 0) {
        plasma_error("illegal value of m");
        return -3;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -4;
    }
    if (lda < imax(1, m)) {
        plasma_error("illegal value of lda");
        return -6;
    }
    if (ldu < 1 || (jobu != PlasmaNoVec && ldu < m)) {
        plasma_error("illegal value of ldu");
        return -9;
    }
    if (ldvt < 1 || (jobvt == PlasmaAllVec && ldvt < n)
                 || (jobvt == PlasmaSomeVec && ldvt < minmn)) {
        plasma_error("illegal value of ldvt");
        return -11;
    }

    // quick return
    if (minmn == 0)
        return PlasmaSuccess;

    // Set tiling parameters.
    int ib = plasma->ib;
    int nb = plasma->nb;

    if (minmn < nb) {
        plasma_error("nb < imin(m, n) not supported");
        return -12;
    }

    // Create tile matrix.
    plasma_desc_t A;
    int retval;
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        m, n, 0, 0, m, n, &A);
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
    size_t lwork = ib*nb + 4*nb*nb;
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

    // Warning !!! plasma_omp_zgesdd is not fully async function.
    // It contains both async and sync functions.
    plasma_omp_zgesdd(jobu, jobvt, A, *T, S, pU, ldu, pVT, ldvt,
                      work, &sequence, &request);

    #pragma omp parallel
    #pragma omp master
    {
        // Translate back to LAPACK layout.
        plasma_omp_zdesc2ge(A, pA, lda, &sequence, &request);
    }

    // implicit synchronization
    plasma_workspace_destroy(&work);

    // Free matrix A in tile layout.
    plasma_desc_destroy(&A);

    // Return status.
    return sequence.status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_gesdd
 *
 *  Computes the singular value decomposition (SVD) of  an
 *  m-by-n matrix A, and optionally the left and right singular vectors.
 *  Tile equivalent of plasma_zgesdd().
 *  May return before the computation is finished.
 *  Operates on matrices stored by tiles.
 *  All matrices are passed through descriptors.
 *  All dimensions are taken from the descriptors.
 *  Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] jobu
 *          Specifies options for computing all or part of the matrix U.
 *          - PlasmaAllVec:  all m columns of U are returned in the array U.
 *          - PlasmaSomeVec: the first min(m, n) columns of U (the left
 *                           singular vectors) are returned in the array U.
 *          - PlasmaNoVec:   no columns of U are computed.
 *
 * @param[in] jobvt
 *          Specifies options for computing all or part of the matrix V^H.
 *          - PlasmaAllVec   all n rows of V^H are returned in the array VT;
 *          - PlasmaSomeVec: the first min(m, n) rows of V^H (the right
 *                           singular vectors) are returned in the array VT.
 *          - PlasmaNoVec:   no rows of V^H are computed.
 *
 *          NOTE: currently this requires jobu = jobvt.
 *
 * @param[in,out] A
 *          Descriptor of matrix A.
 *          A is stored in the tile layout.
 *          On exit, the contents of A are destroyed.
 *
 * @param[out] T
 *          Descriptor of matrix T.
 *          TODO
 *
 * @param[out] S
 *          The double precision singular values of A,
 *          sorted so that S(i) >= S(i + 1).
 *
 * @param[out] U
 *          Pointer to the left singular vectors matrix U.
 *          - If jobu = PlasmaAllVec, U is ldu-by-m.
 *            On exit, U contains the m-by-m unitary matrix U.
 *          - If jobu = PlasmaSomeVec, U is ldu-by-min(m, n).
 *            On exit, U contains the m-by-min(m, n) unitary matrix U.
 *          - If jobu = PlasmaNoVec, U is not referenced.
 *
 * @param[in] ldu
 *          The leading dimension of the array U. ldu >= 1;
 *          if jobu = PlasmaAllVec or PlasmaSomeVec, ldu >= m.
 *
 * @param[out] VT
 *         Pointer to the right singular vectors matrix VT
 *         - If jobvt = PlasmaAllVec, VT is ldvt-by-n.
 *           On exit, VT contains the n-by-n unitary matrix V^H.
 *         - If jobvt = PlasmaSomeVec, VT is ldvt-by-n.
 *           On exit, VT contains the first min(m, n) rows of
 *           V^H (the right singular vectors, stored rowwise).
 *         - If jobvt = PlasmaNoVec, VT is not referenced.
 *
 * @param[in] ldvt
 *         The leading dimension of the array VT. ldvt >= 1;
 *         if jobvt = PlasmaAllVec, ldvt >= n;
 *         if jobvt = PlasmaSomeVec, ldvt >= min(m, n).
 *
 * @param[out] work
 *          Workspace for the auxiliary arrays needed by some coreblas kernels.
 *          Allocated by the plasma_workspace_create function.
 *
 * @param[in] sequence
 *          Identifies the sequence of function calls that this call belongs to
 *          (for completion checks and exception handling purposes).
 *
 * @param[out] request
 *          Identifies this function call (for exception handling purposes).
 *
 * @retval void
 *          Errors are returned by setting sequence->status and
 *          request->status to error values. The sequence->status and
 *          request->status should never be set to PlasmaSuccess (the
 *          initial values) since another async call may be setting a
 *          failure value at the same time.
 *
 *******************************************************************************
 *
 * @sa plasma_zgesdd
 * @sa plasma_omp_cgesdd
 * @sa plasma_omp_dgesdd
 * @sa plasma_omp_sgesdd
 *
 ******************************************************************************/
void plasma_omp_zgesdd(plasma_enum_t jobu, plasma_enum_t jobvt,
                       plasma_desc_t A, plasma_desc_t T,
                       double *S,
                       plasma_complex64_t *pU,  int ldu,
                       plasma_complex64_t *pVT, int ldvt,
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
    if (jobu != PlasmaNoVec && jobu != PlasmaAllVec && jobu != PlasmaSomeVec) {
        plasma_error("illegal value of jobu");
        return;
    }
    if (jobvt != PlasmaNoVec && jobvt != PlasmaAllVec && jobvt != PlasmaSomeVec) {
        plasma_error("illegal value of jobvt");
        return;
    }
    if (jobvt != jobu) {
        plasma_error("in this version: jobu should be equal jobvt");
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

    plasma_enum_t uplo = A.m >= A.n ? PlasmaUpper : PlasmaLower;
    int m = A.m;
    int n = A.n;
    int minmn = imin(m, n);
    int nb = imin(A.mb, minmn);
    int lapack_info;
    char lapack_uplo = A.m >= A.n ? 'U' : 'L';
    int lda_band = 3*nb + 1;

    double *E = NULL;
    plasma_complex64_t *pA_band = NULL;
    plasma_complex64_t *VQ2   = NULL;
    plasma_complex64_t *VP2   = NULL;
    plasma_complex64_t *tauQ2 = NULL;
    plasma_complex64_t *tauP2 = NULL;
    plasma_complex64_t *TQ2   = NULL;
    plasma_complex64_t *TP2   = NULL;

    //===================
    // Overview to factor A = U Sigma V^H
    //
    // optional reduction to square (not yet implemented)
    // if (m >> n)
    //     Q0 R = A  // QR factorization
    //     Ahat = R
    // elif (m << n)
    //     L P0 = A  // LQ factorization
    //     Ahat = L
    // else
    //     Ahat = A
    //
    // Q1 Band   P1^H = Ahat    // reduction to band (ge2gb)
    // Q2 Bidiag P2^H = Band    // bulge chasing (gbbrd)
    // U0 Sigma  V0^H = Bidiag  // bidiagonal SVD (bdsdc)
    // U   = Q0 Q1 Q2 U0        // various unmqr
    // V^H = V0^H P2^H P1^H P0  // various unmlq
    //===================

    //===================
    // TODO
    // if m >> n, initial QR factorization
    // if m << n, initial LQ factorization
    //===================

    int Un  = (jobu  == PlasmaAllVec ? m : minmn);
    int VTm = (jobvt == PlasmaAllVec ? n : minmn);

    // Allocate workspace for band storage.
    // pA_band looks like:
    //       __________________________________
    // NB   |               zero               |
    //       ----------------------------------
    // NB+1 |               band A             |
    //       ----------------------------------
    // NB   |_______________zero_______________|
    //
    pA_band = (plasma_complex64_t*)
        malloc((size_t)lda_band*minmn*sizeof(plasma_complex64_t));
    if (pA_band == NULL) {
        plasma_error("malloc(pA_band) failed");
        goto cleanup;
    }
    memset(pA_band, 0, (size_t)lda_band*minmn*sizeof(plasma_complex64_t));

    // Allocate E for the off-diagonal elements of B
    E = (double*) malloc(minmn*sizeof(double));
    if (E == NULL) {
        plasma_error("malloc(E) failed");
        goto cleanup;
    }

    //===================
    // Reduction to band
    //===================
    plasma_time_t time = omp_get_wtime();
    #pragma omp parallel
    #pragma omp master
    {
        plasma_pzge2gb(A, T, work, sequence, request);

        // Copy tile band to lapack band
        plasma_pzgecpy_tile2lapack_band(uplo, A,
                                        &pA_band[nb], lda_band,
                                        sequence, request);
    }
    time = omp_get_wtime() - time;
    //printf("\n N=%d:  1-stage= %lf\t", n, time);

    //====================
    // Setup for bulge chasing
    //====================
    int vblksiz; // blocking used when applying V2 to the matrix U
    int blkcnt;  // number of diamonds or tiles of Vs
    int ldt, ldv;
    int wantz = 0;

    if (jobu == PlasmaNoVec && jobvt == PlasmaNoVec)
        wantz = 0;
    else
        wantz = 2;

    vblksiz = nb/4; // equivalent if ib
    ldt     = vblksiz;

    // Data for U
    if (wantz) {
        findVTsiz(minmn, nb, vblksiz, &blkcnt, &ldv);
        tauQ2 = (plasma_complex64_t*)
            malloc((size_t)blkcnt*vblksiz*sizeof(plasma_complex64_t));
        VQ2 = (plasma_complex64_t*)
            malloc((size_t)ldv*blkcnt*vblksiz*sizeof(plasma_complex64_t));
        TQ2 = (plasma_complex64_t*)
            malloc((size_t)ldt*blkcnt*vblksiz*sizeof(plasma_complex64_t));

        if (tauQ2 == NULL || VQ2 == NULL || TQ2 == NULL) {
            plasma_error("malloc tauQ2 or VQ2 or TQ2 failed");
            goto cleanup;
        }
        memset(tauQ2, 0, (size_t)    blkcnt*vblksiz*sizeof(plasma_complex64_t));
        memset(VQ2,   0, (size_t)ldv*blkcnt*vblksiz*sizeof(plasma_complex64_t));
        memset(TQ2,   0, (size_t)ldt*blkcnt*vblksiz*sizeof(plasma_complex64_t));
    }
    else {
        tauQ2 = (plasma_complex64_t*)
            malloc((size_t)2*minmn*sizeof(plasma_complex64_t));
        VQ2 = (plasma_complex64_t*)
            malloc((size_t)2*minmn*sizeof(plasma_complex64_t));

        if (tauQ2 == NULL || VQ2 == NULL) {
            plasma_error("malloc tauQ2 or VQ2 failed");
            goto cleanup;
        }
        memset(tauQ2, 0, (size_t)2*minmn*sizeof(plasma_complex64_t));
        memset(VQ2,   0, (size_t)2*minmn*sizeof(plasma_complex64_t));
    }

    // Data for VT
    if (wantz) {
        findVTsiz(minmn, nb, vblksiz, &blkcnt, &ldv);
        tauP2 = (plasma_complex64_t*)
            malloc((size_t)blkcnt*vblksiz*sizeof(plasma_complex64_t));
        VP2 = (plasma_complex64_t*)
            malloc((size_t)ldv*blkcnt*vblksiz*sizeof(plasma_complex64_t));
        TP2 = (plasma_complex64_t*)
            malloc((size_t)ldt*blkcnt*vblksiz*sizeof(plasma_complex64_t));

        if (tauP2 == NULL || VP2 == NULL || TP2 == NULL) {
            plasma_error("malloc tauP2 or VP2 or TP2 failed");
            goto cleanup;
        }
        memset(tauP2, 0, (size_t)    blkcnt*vblksiz*sizeof(plasma_complex64_t));
        memset(VP2,   0, (size_t)ldv*blkcnt*vblksiz*sizeof(plasma_complex64_t));
        memset(TP2,   0, (size_t)ldt*blkcnt*vblksiz*sizeof(plasma_complex64_t));
    }
    else {
        tauP2 = (plasma_complex64_t*)
            malloc((size_t)2*minmn*sizeof(plasma_complex64_t));
        VP2 = (plasma_complex64_t*)
            malloc((size_t)2*minmn*sizeof(plasma_complex64_t));

        if (tauP2 == NULL || VP2 == NULL) {
            plasma_error("malloc tauP2 or VP2 failed");
            goto cleanup;
        }
        memset(tauP2, 0, (size_t)2*minmn*sizeof(plasma_complex64_t));
        memset(VP2,   0, (size_t)2*minmn*sizeof(plasma_complex64_t));
    }

    //=======================================
    // Bulge chasing
    //=======================================
    time = omp_get_wtime();
    plasma_pzgbbrd_static(uplo, minmn, nb, vblksiz,
                          pA_band, lda_band,
                          VQ2, tauQ2, VP2, tauP2,
                          S, E, wantz,
                          work,
                          sequence, request);
    time = omp_get_wtime() - time;
    //printf("2-stage= %lf\t", time);

    //=======================================
    // SVD solver
    //=======================================
    // Use lapack D&C routine on the resulting bidiag [S E]
    double rdummy[1];
    int idummy[1];
    time = omp_get_wtime();
    if (jobu == PlasmaNoVec && jobvt == PlasmaNoVec) {
        lapack_info = LAPACKE_dbdsdc(LAPACK_COL_MAJOR, lapack_uplo,
                                     'N', minmn, S, E,
                                     rdummy, ldu,
                                     rdummy, ldvt,
                                     rdummy, idummy);
    }
    else {
        // Let Uhat, Vhat be the min(m, n)-by-min(m, n) outputs of bdsdc.
        //
        // For job = PlasmaAllVec:
        // U0 = [ Uhat  0 ],  VT0 = [ VThat  0 ]
        //      [  0    I ]         [   0    I ]
        // where U is m-by-m, VT is n-by-n.
        //
        // For job = PlasmaSomeVec (min(m, n) vectors):
        // U0 = [ Uhat ],  VT0 = [ VThat  0 ]
        //      [  0   ]
        // where U is m-by-min(m, n), VT is min(m, n)-by-n.

        // set pU and pVT to zero
        memset(pU,  0, (size_t)ldu  * Un * sizeof(plasma_complex64_t));
        memset(pVT, 0, (size_t)ldvt * n  * sizeof(plasma_complex64_t));

        // Initialize pU(n+1:m, n+1:m) and pVT(m+1:n, m+1:n) to Identity
        if (jobu == PlasmaAllVec) {
            for (int i = n; i < m; i++) {
                pU[i + (size_t)ldu*i] = 1.0;
            }
        }
        if (jobvt == PlasmaAllVec) {
            for (int i = m; i < n; i++) {
                pVT[i + (size_t)ldvt*i] = 1.0;
            }
        }

        #if defined COMPLEX
            // Allocate real matrices RU and RVT to procces B[minmn-by-minmn]
            double* RU  = (double*) malloc((size_t)minmn*minmn*sizeof(double));
            double* RVT = (double*) malloc((size_t)minmn*minmn*sizeof(double));
            if (RU == NULL || RVT == NULL) {
                plasma_error("malloc RU or RVT failed");
                free(RU);
                free(RVT);
                goto cleanup;
            }

            // Call D&C singular value kernel
            lapack_info = LAPACKE_dbdsdc(LAPACK_COL_MAJOR, lapack_uplo, 'I',
                                         minmn, S, E, RU, minmn, RVT, minmn,
                                         rdummy, idummy);

            // Copy real matrices RU and RVT to complex matrices pU and pVT.
            // TODO: use zlacp2
            for (int j = 0; j < minmn; j++) {
                for (int i = 0; i < minmn; i++) {
                    pU[i + (size_t)ldu*j] = RU[i + (size_t)minmn*j];
                }
            }
            for (int j = 0; j < minmn; j++) {
                for (int i = 0; i < minmn; i++) {
                    pVT[i + (size_t)ldvt*j] = RVT[i + (size_t)minmn*j];
                }
            }
            free(RU);
            free(RVT);
        #else
            // Call D&C singular value kernel
            lapack_info = LAPACKE_dbdsdc(LAPACK_COL_MAJOR, lapack_uplo, 'I',
                                         minmn, S, E, pU, ldu, pVT, ldvt,
                                         rdummy, idummy);
        #endif
    }
    time = omp_get_wtime() - time;
    //printf("SVD= %lf\t", time);

    if (lapack_info != 0) {
        plasma_error("bdsdc() failed");
        goto cleanup;
    }

    //=======================================
    // Generate U = [Q0] Q1 Q2 U0
    //=======================================
    time = omp_get_wtime();
    if (jobu == PlasmaAllVec || jobu == PlasmaSomeVec) {
        // compute T2
        #pragma omp parallel
        {
            plasma_pzlarft_blgtrd(minmn, nb, vblksiz,
                                  VQ2, TQ2, tauQ2,
                                  sequence, request);
        }

        // apply Q2 from bulge chasing
        #pragma omp parallel
        {
            plasma_pzunmqr_blgtrd(PlasmaLeft, PlasmaNoTrans,
                                  minmn, nb, minmn, vblksiz, wantz,
                                  VQ2, TQ2, tauQ2, pU, ldu,
                                  work, sequence, request);
        }

        plasma_desc_t U;
        plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                   m, Un, 0, 0, m, Un, &U);

        #pragma omp parallel
        #pragma omp master
        {
            // Translate U to tile layout.
            plasma_pzge2desc(pU, ldu, U, sequence, request);

            // apply Q1 from the reduction to band
            if (m < n) {
                plasma_pzunmqr(PlasmaLeft, PlasmaNoTrans,
                               plasma_desc_view(A, A.mb, 0, A.m-A.mb, A.n-A.nb),
                               plasma_desc_view(T, T.mb, 0, T.m-T.mb, T.n-T.nb),
                               plasma_desc_view(U, U.mb, 0, U.m-U.mb, U.n),
                               work, sequence, request);
            }
            else {
                plasma_pzunmqr(PlasmaLeft, PlasmaNoTrans,
                               A, T, U,
                               work, sequence, request);
            }

            // TODO: apply Q0 from initial QR factorization

            // Translate U to lapack layout.
            plasma_pzdesc2ge(U, pU, ldu, sequence, request);
        }

        plasma_desc_destroy(&U);
    }

    //=======================================
    // Generate VT = V^H = V0^H P2^H P1^H P0
    //=======================================
    if (jobvt == PlasmaAllVec || jobvt == PlasmaSomeVec) {
        // compute T2
        #pragma omp parallel
        {
            plasma_pzlarft_blgtrd(minmn, nb, vblksiz,
                                  VP2, TP2, tauP2,
                                  sequence, request);
        }

        // apply P2 from bulge chasing
        #pragma omp parallel
        {
            plasma_pzunmqr_blgtrd(PlasmaRight, PlasmaConjTrans,
                                  minmn, nb, minmn, vblksiz, wantz,
                                  VP2, TP2, tauP2,
                                  pVT, ldvt,
                                  work, sequence, request);
        }

        plasma_desc_t VT;
        plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                   VTm, n, 0, 0, VTm, n, &VT);

        #pragma omp parallel
        #pragma omp master
        {
            // Translate VT to tile layout.
            plasma_pzge2desc(pVT, ldvt, VT, sequence, request);

            // apply P1 from the reduction to band
            if (m < n) {
                plasma_pzunmlq(PlasmaRight, PlasmaNoTrans,
                               A, T, VT,
                               work, sequence, request);
            }
            else {
                plasma_pzunmlq(PlasmaRight, PlasmaNoTrans,
                               plasma_desc_view(A,  0, A.nb,  A.m-A.mb, A.n-A.nb),
                               plasma_desc_view(T,  0, T.nb,  T.m-T.mb, T.n-T.nb),
                               plasma_desc_view(VT, 0, VT.nb, VT.m,     VT.n-VT.nb),
                               work, sequence, request);
            }

            // TODO: apply P0 from initial LQ factorization

            // Translate VT to lapack layout.
            plasma_pzdesc2ge(VT, pVT, ldvt, sequence, request);
        }

        plasma_desc_destroy(&VT);
    }
    time = omp_get_wtime() - time;
    //printf("Vect= %lf\n", time);

cleanup:
    // Free all arrays.
    // If an array wasn't allocated, it's NULL, so free does nothing.
    free(E);
    free(pA_band);
    free(VQ2);
    free(VP2);
    free(tauQ2);
    free(tauP2);
    free(TQ2);
    free(TP2);
}
