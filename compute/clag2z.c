/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions mixed zc -> ds
 *
 **/

#include "plasma_types.h"
#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_zc.h"

/***************************************************************************//**
 *
 * @ingroup plasma_lag2
 *
 *  Converts m-by-n matrix As from complex single to complex double precision.
 *
 *******************************************************************************
 *
 * @param[in] m
 *          The number of rows of the matrix As. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix As. n >= 0.
 *
 * @param[in] As
 *          The ldas-by-n matrix As in single complex precision.
 *
 * @param[in] ldas
 *          The leading dimension of the array As. ldas >= max(1,m).
 *
 * @param[out] A
 *          On exit, the lda-by-n matrix A in double complex precision.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,m).
 *
 *******************************************************************************
 *
 * @retval PlasmaSuccess successful exit
 *
 *******************************************************************************
 *
 * @sa plasma_omp_clag2z
 * @sa PLASMA_zlag2c
 * @sa PLASMA_dlag2s
 * @sa PLASMA_slag2d
 *
 ******************************************************************************/
int PLASMA_clag2z(int m, int n,
                  plasma_complex32_t *As, int ldas,
                  plasma_complex64_t *A,  int lda)
{
    int nb;
    int retval;
    int status;
    plasma_context_t  *plasma;
    plasma_sequence_t *sequence = NULL;
    plasma_request_t   request  = PLASMA_REQUEST_INITIALIZER;
    plasma_desc_t descA, descB;

    // Get PLASMA context
    plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    // Check input arguments
    if (m < 0) {
        plasma_error("illegal value of m");
        return -1;
    }

    if (n < 0) {
        plasma_error("illegal value of n");
        return -2;
    }

    if (ldas < imax(1, m)) {
        plasma_error("illegal value of ldas");
        return -4;
    }

    if (lda < imax(1, m)) {
        plasma_error("illegal value of lda");
        return -6;
    }

    // Quick return
    if (imin(n, m) == 0)
      return PlasmaSuccess;

    // Tune
    // if (plasma_tune(PLASMA_FUNC_CLAG2Z, m, n, 0) != PlasmaSuccess) {
    //     plasma_error("plasma_tune() failed");
    //     return status;
    // }
    nb = plasma->nb;

    // Initialize tile matrix descriptors
    descAs = plasma_desc_init(PlasmaComplexReal,   nb, nb, nb*nb,
                              m, n, 0, 0, m, n);

    descA  = plasma_desc_init(PlasmaComplexDouble, nb, nb, nb*nb,
                              m, n, 0, 0, m, n);

    // Allocate matrices in tile layout
    retval = plasma_desc_mat_alloc(&descAs);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_mat_alloc() failed");
        return retval;
    }

    retval = plasma_desc_mat_alloc(&descA);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_mat_alloc() failed");
        plasma_desc_mat_free(&descAs);
        return retval;
    }

    // Create sequence
    retval = plasma_sequence_create(&sequence);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_sequence_create() failed");
        return retval;
    }

    // Asynchronous block
    #pragma omp parallel
    #pragma omp master
    {
        // Translate to tile layout
        PLASMA_zcm2ccrb_Async(As, ldas, &descAs, sequence, &request);
        PLASMA_zcm2ccrb_Async(A,  lda,  &descA,  sequence, &request);

        // Call tile async function
        if (sequence->status == PlasmaSuccess) {
            plasma_omp_clag2z(&descAs, &descA, sequence, &request);
        }

        // Revert to LAPACK layout
        PLASMA_zccrb2cm_Async(&descAs, As, ldas, sequence, &request);
        PLASMA_zccrb2cm_Async(&descA,  A,  lda,  sequence, &request);
    }
    // Implicit synchronization

    // Deallocate memory in tile layout
    plasma_desc_mat_free(&descAs);
    plasma_desc_mat_free(&descA);

    // Destroy sequence
    plasma_sequence_destroy(sequence);

    // Return status
    status = sequence->status;
    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_lag2
 *
 *  Converts m-by-n matrix A from single complex to double complex precision.
 *  Non-blocking tile version of PLASMA_clag2z(). May return before the
 *  computation is finished. Operates on matrices stored by tiles. All matrices
 *  are passed through descriptors. All dimensions are taken from the
 *  descriptors. Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] As
 *          Descriptor of matrix As.
 *
 * @param[out] A
 *          Descriptor of matrix A.
 *
 * @param[in] sequence
 *          Identifies the sequence of function calls that this call belongs to
 *          (for completion checks and exception handling purposes). Check the
 *          sequence->status for errors.
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
 * @sa PLASMA_clag2z
 * @sa plasma_omp_zlag2c
 * @sa plasma_omp_dlag2s
 * @sa plasma_omp_slag2d
 *
 ******************************************************************************/
void plasma_omp_clag2z(plasma_desc_t *As, plasma_desc_t *A,
                       plasma_sequence_t *sequence, plasma_request_t *request)
{
    plasma_desc_t descAs;
    plasma_desc_t descA;
    plasma_context_t *plasma;

    // Get PLASMA context
    plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }

    // Check descriptors for correctness
    if (plasma_desc_check(As) != PlasmaSuccess) {
        plasma_error("invalid As");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    else {
        descAs = *As;
    }

    if (plasma_desc_check(A) != PlasmaSuccess) {
        plasma_error("invalid A");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    else {
        descA = *A;
    }

    if (descAs.nb != descAs.mb) {
        plasma_error("only square tiles supported");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }

    if (sequence == NULL) {
        plasma_error("NULL sequence");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }

    if (request == NULL) {
        plasma_error("NULL request");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }

    // Quick return
    if (imin(descAs.m, descAs.n) == 0)
        return;

    // Call parallel function
    plasma_pclag2z(descAs, descA, sequence, request);
}
