/**
 *
 * @file zgemm.c
 *
 *  PLASMA computational routine.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver.
 *
 * @version 3.0.0
 * @author Jakub Kurzak
 * @date 2016-01-01
 * @precisions normal z -> s d c
 *
 **/

#include "../control/async.h"
#include "../control/context.h"
#include "../control/descriptor.h"
#include "../control/internal.h"
#include "../include/plasma_z.h"
#include "../include/plasmatypes.h"

/***************************************************************************//**
 *
 * @ingroup PLASMA_Complex64_t
 *
 *  Performs one of the matrix-matrix operations
 *
 *          \f[ C = \alpha [op( A )\times op( B )] + \beta C, \f]
 *
 *  where op( X ) is one of:
 *          - op( X ) = X  or
 *          - op( X ) = X' or
 *          - op( X ) = conjg( X' ),
 *
 *  alpha and beta are scalars, and A, B and C are matrices, with op( A )
 *  an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
 *
 *******************************************************************************
 *
 * @param[in] transA
 *          - PlasmaNoTrans:   A is not transposed,
 *          - PlasmaTrans:     A is transposed,
 *          - PlasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] transB
 *          - PlasmaNoTrans:   B is not transposed,
 *          - PlasmaTrans:     B is transposed,
 *          - PlasmaConjTrans: B is conjugate transposed.
 *
 * @param[in] m
 *          The number of rows of the matrix op( A ) and of the matrix C.
 *          m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix op( B ) and of the matrix C.
 *          n >= 0.
 *
 * @param[in] k
 *          The number of columns of the matrix op( A ) and the number of rows
 *          of the matrix op( B ). k >= 0.
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          An lda-by-ka matrix, where ka is k when transA = PlasmaNoTrans,
 *          and is m otherwise.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,m).
 *
 * @param[in] B
 *          An ldb-by-kb matrix, where kb is n when transB = PlasmaNoTrans,
 *          and is k otherwise.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1,n).
 *
 * @param[in] beta
 *          The scalar beta.
 *
 * @param[in,out] C
 *          An ldc-by-n matrix. On exit, the array is overwritten by the m-by-n
 *          matrix ( alpha*op( A )*op( B ) + beta*C ).
 *
 * @param[in] ldc
 *          The leading dimension of the array C. ldc >= max(1,m).
 *
 *******************************************************************************
 *
 * @retval PLASMA_SUCCESS successful exit
 *
 *******************************************************************************
 *
 * @sa PLASMA_zgemm_Tile
 * @sa PLASMA_zgemm_Tile_Async
 * @sa PLASMA_cgemm
 * @sa PLASMA_dgemm
 * @sa PLASMA_sgemm
 *
 ******************************************************************************/
int PLASMA_zgemm(PLASMA_enum transA, PLASMA_enum transB,
                 int m, int n, int k,
                 PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int lda,
                                           PLASMA_Complex64_t *B, int ldb,
                 PLASMA_Complex64_t beta,  PLASMA_Complex64_t *C, int ldc)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA_zgemm", "PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }

    // Check input arguments
    if ((transA != PlasmaNoTrans) &&
        (transA != PlasmaTrans) &&
        (transA != PlasmaConjTrans)) {
        plasma_error("PLASMA_zgemm", "illegal value of transA");
        return -1;
    }
    if ((transB != PlasmaNoTrans) &&
        (transB != PlasmaTrans) &&
        (transB != PlasmaConjTrans)) {
        plasma_error("PLASMA_zgemm", "illegal value of transB");
        return -2;
    }
    if (m < 0) {
        plasma_error("PLASMA_zgemm", "illegal value of M");
        return -3;
    }
    if (n < 0) {
        plasma_error("PLASMA_zgemm", "illegal value of N");
        return -4;
    }
    if (k < 0) {
        plasma_error("PLASMA_zgemm", "illegal value of N");
        return -5;
    }
    int Am, An;
    int Bm, Bn;
    if (transA == PlasmaNoTrans) {
        Am = m;
        An = k;
    }
    else {
        Am = k;
        An = m;
    }
    if (transB == PlasmaNoTrans) {
        Bm = k;
        Bn = n;
    }
    else {
        Bm = n;
        Bn = k;
    }
    if (lda < imax(1, Am)) {
        plasma_error("PLASMA_zgemm", "illegal value of lda");
        return -8;
    }
    if (ldb < imax(1, Bm)) {
        plasma_error("PLASMA_zgemm", "illegal value of ldb");
        return -10;
    }
    if (ldc < imax(1, m)) {
        plasma_error("PLASMA_zgemm", "illegal value of ldc");
        return -13;
    }
    // Quick return
    if (m == 0 || n == 0 || ((alpha == (PLASMA_Complex64_t)0.0 || k == 0) &&
        beta == (PLASMA_Complex64_t)1.0))
        return PLASMA_SUCCESS;

    // Tune nb.
    // if (plasma_tune(PLASMA_FUNC_ZGEMM, m, n, 0) != PLASMA_SUCCESS) {
    //     plasma_error("PLASMA_zgemm", "plasma_tune() failed");
    //     return status;
    // }
    int nb = plasma->nb;

    // Create sequence, initialize request.
    PLASMA_sequence *sequence = NULL;
    plasma_sequence_create(&sequence);
    PLASMA_request request = PLASMA_REQUEST_INITIALIZER;

    PLASMA_desc descA;
    PLASMA_desc descB;
    PLASMA_desc descC;

    if (plasma->translation == PLASMA_OUTOFPLACE) {

        plasma_zooplap2tile(descA, A, nb, nb, lda, An, 0, 0, Am, An,
                            sequence, &request);

        plasma_zooplap2tile(descB, B, nb, nb, ldb, Bn, 0, 0, Bm, Bn,
                            sequence, &request);

        plasma_zooplap2tile(descC, C, nb, nb, ldc, n,  0, 0, m,  n,
                            sequence, &request);

        if (descA.mat == NULL || descB.mat == NULL || descC.mat == NULL) {
            plasma_desc_mat_free(&descA);
            plasma_desc_mat_free(&descB);
            plasma_desc_mat_free(&descC);
            return PLASMA_ERR_OUT_OF_RESOURCES;
        }
    }
    else {
        // plasma_ziplap2tile(descA, A, nb, nb, lda, An, 0, 0, Am, An,
        //                    sequence, &request);

        // plasma_ziplap2tile(descB, B, nb, nb, ldb, Bn, 0, 0, Bm, Bn,
        //                    sequence, &request);

        // plasma_ziplap2tile(descC, C, nb, nb, ldc, n,  0, 0, m,  n,
        //                    sequence, &request);
    }

    /* Call the tile interface. */
    PLASMA_zgemm_Tile_Async(transA, transB,
                            alpha, &descA, &descB, beta, &descC,
                            sequence, &request);

    if (plasma->translation == PLASMA_OUTOFPLACE) {
        plasma_zooptile2lap(descC, C, nb, nb, ldc, n, sequence, &request);
        plasma_desc_mat_free(&descA);
        plasma_desc_mat_free(&descB);
        plasma_desc_mat_free(&descC);
    }
    else {
        // plasma_ziptile2lap(descA, A, nb, nb, lda, An, sequence, &request);
        // plasma_ziptile2lap(descB, B, nb, nb, ldb, Bn, sequence, &request);
        // plasma_ziptile2lap(descC, C, nb, nb, ldc, n,  sequence, &request);
    }

    int status = sequence->status;
    plasma_sequence_destroy(sequence);
    return status;
}

/***************************************************************************//**
 *
 * @ingroup PLASMA_Complex64_t_Tile
 *
 *  Performs matrix multiplication.
 *  Tile equivalent of PLASMA_zgemm().
 *  Operates on matrices stored by tiles.
 *  All matrices are passed through descriptors.
 *  All dimensions are taken from the descriptors.
 *
 *******************************************************************************
 *
 * @param[in] A
 *          Descriptor of matrix A.
 *
 * @param[in] B
 *          Descriptor of matrix B.
 *
 * @param[in,out] C
 *          Descriptor of matrix C.
 *
 *******************************************************************************
 *
 * @retval PLASMA_SUCCESS successful exit
 *
 *******************************************************************************
 *
 * @sa PLASMA_zgemm
 * @sa PLASMA_zgemm_Tile_Async
 * @sa PLASMA_cgemm_Tile
 * @sa PLASMA_dgemm_Tile
 * @sa PLASMA_sgemm_Tile
 *
 ******************************************************************************/
int PLASMA_zgemm_Tile(PLASMA_enum transA, PLASMA_enum transB,
                      PLASMA_Complex64_t alpha, PLASMA_desc *A,
                                                PLASMA_desc *B,
                      PLASMA_Complex64_t beta,  PLASMA_desc *C)
{
    plasma_context_t *plasma;
    PLASMA_sequence *sequence = NULL;
    PLASMA_request request = PLASMA_REQUEST_INITIALIZER;
    int status;

    plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA_zgemm_Tile", "PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }
    plasma_sequence_create(&sequence);
    PLASMA_zgemm_Tile_Async(transA, transB,
                            alpha, A, B, beta, C,
                            sequence, &request);

    status = sequence->status;
    plasma_sequence_destroy(sequence);
    return status;
}

/***************************************************************************//**
 *
 * @ingroup PLASMA_Complex64_t_Tile_Async
 *
 *  Performs matrix multiplication.
 *  Non-blocking equivalent of PLASMA_zgemm_Tile().
 *  May return before the computation is finished.
 *  Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
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
 * @sa PLASMA_zgemm
 * @sa PLASMA_zgemm_Tile
 * @sa PLASMA_cgemm_Tile_Async
 * @sa PLASMA_dgemm_Tile_Async
 * @sa PLASMA_sgemm_Tile_Async
 *
 ******************************************************************************/
int PLASMA_zgemm_Tile_Async(PLASMA_enum transA, PLASMA_enum transB,
                            PLASMA_Complex64_t alpha, PLASMA_desc *A,
                                                      PLASMA_desc *B,
                            PLASMA_Complex64_t beta,  PLASMA_desc *C,
                            PLASMA_sequence *sequence, PLASMA_request *request)
{
    plasma_context_t *plasma;
    PLASMA_desc descA;
    PLASMA_desc descB;
    PLASMA_desc descC;
    int m, n, k;
    int Am, An, Ai, Aj, Amb, Anb;
    int Bm, Bn, Bi, Bj, Bmb, Bnb;

    plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA_zgemm_Tile_Async", "PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }
    if (sequence == NULL) {
        plasma_error("PLASMA_zgemm_Tile_Async", "NULL sequence");
        return PLASMA_ERR_UNALLOCATED;
    }
    if (request == NULL) {
        plasma_error("PLASMA_zgemm_Tile_Async", "NULL request");
        return PLASMA_ERR_UNALLOCATED;
    }
    /* Check sequence status */
    if (sequence->status == PLASMA_SUCCESS)
        request->status = PLASMA_SUCCESS;
    else
        return plasma_request_fail(sequence, request,
                                   PLASMA_ERR_SEQUENCE_FLUSHED);

    /* Check descriptors for correctness */
    if (plasma_desc_check(A) != PLASMA_SUCCESS) {
        plasma_error("PLASMA_zgemm_Tile_Async", "invalid first descriptor");
        return plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
    }
    else {
        descA = *A;
    }
    if (plasma_desc_check(B) != PLASMA_SUCCESS) {
        plasma_error("PLASMA_zgemm_Tile_Async", "invalid second descriptor");
        return plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
    }
    else {
        descB = *B;
    }
    if (plasma_desc_check(C) != PLASMA_SUCCESS) {
        plasma_error("PLASMA_zgemm_Tile_Async", "invalid third descriptor");
        return plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
    }
    else {
        descC = *C;
    }
    /* Check input arguments */
    if ((transA != PlasmaNoTrans) && (transA != PlasmaTrans) &&
        (transA != PlasmaConjTrans)) {
        plasma_error("PLASMA_zgemm_Tile_Async", "illegal value of transA");
        return plasma_request_fail(sequence, request, -1);
    }
    if ((transB != PlasmaNoTrans) && (transB != PlasmaTrans) &&
        (transB != PlasmaConjTrans)) {
        plasma_error("PLASMA_zgemm_Tile_Async", "illegal value of transB");
        return plasma_request_fail(sequence, request, -2);
    }

    if (transA == PlasmaNoTrans) {
        Am  = descA.m;
        An  = descA.n;
        Amb = descA.mb;
        Anb = descA.nb;
        Ai  = descA.i;
        Aj  = descA.j;
    }
    else {
        Am  = descA.n;
        An  = descA.m;
        Amb = descA.nb;
        Anb = descA.mb;
        Ai  = descA.j;
        Aj  = descA.i;
    }

    if (transB == PlasmaNoTrans) {
        Bm  = descB.m;
        Bn  = descB.n;
        Bmb = descB.mb;
        Bnb = descB.nb;
        Bi  = descB.i;
        Bj  = descB.j;
    }
    else {
        Bm  = descB.n;
        Bn  = descB.m;
        Bmb = descB.nb;
        Bnb = descB.mb;
        Bi  = descB.j;
        Bj  = descB.i;
    }

    if ((Amb != descC.mb) || (Anb != Bmb) || (Bnb != descC.nb)) {
        plasma_error("PLASMA_zgemm_Tile_Async", "tile sizes have to match");
        return plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
    }
    if ((Am != descC.m) || (An != Bm) || (Bn != descC.n)) {
        plasma_error("PLASMA_zgemm_Tile_Async",
                     "sizes of matrices have to match");
        return plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
    }
    if ((Ai%Amb != descC.i%descC.mb) || (Aj%Anb != Bi%Bmb) ||
        (Bj%Bnb != descC.j%descC.nb) ) {
        plasma_error("PLASMA_zgemm_Tile_Async", "start indexes have to match");
        return plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
    }

    m = descC.m;
    n = descC.n;
    k = An;

    /* Quick return */
    if (m == 0 || n == 0 || ((alpha == (PLASMA_Complex64_t)0.0 || k == 0) &&
        beta == (PLASMA_Complex64_t)1.0))
        return PLASMA_SUCCESS;

    plasma_pzgemm(transA, transB,
                  alpha, descA, descB, beta, descC,
                  sequence, request);

    return PLASMA_SUCCESS;
}
