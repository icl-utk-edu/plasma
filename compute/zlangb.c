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
#include "plasma_types.h"

/***************************************************************************//**
 *
 * @ingroup plasma_langb
 *
 *  Returns the norm of a general band matrix as
 *
 *     zlange = ( max(abs(A(i,j))), NORM = PlasmaMaxNorm
 *              (
 *              ( norm1(A),         NORM = PlasmaOneNorm
 *              (
 *              ( normI(A),         NORM = PlasmaInfNorm
 *              (
 *              ( normF(A),         NORM = PlasmaFrobeniusNorm
 *
 *  where norm1 denotes the one norm of a matrix (maximum column sum),
 *  normI denotes the infinity norm of a matrix (maximum row sum) and
 *  normF denotes the Frobenius norm of a matrix (square root of sum
 *  of squares). Note that max(abs(A(i,j))) is not a consistent matrix
 *  norm.
 *
 *******************************************************************************
 *
 * @param[in] norm
 *          - PlasmaMaxNorm: max norm
 *          - PlasmaOneNorm: one norm
 *          - PlasmaInfNorm: infinity norm
 *          - PlasmaFrobeniusNorm: Frobenius norm
 *
 * @param[in] m
 *          The number of rows of the matrix A. m >= 0. When m = 0,
 *          the returned value is set to zero.
 *
 * @param[in] n
 *          The number of columns of the matrix A. n >= 0. When n = 0,
 *          the returned value is set to zero.
 *
 * @param[in] kl
 *          The number of subdiagonals within the band of A. kl >= 0.
 *
 * @param[in] ku
 *          The number of superdiagonals within the band of A. ku >= 0.
 *
 * @param[in] pAB
 *          The band matrix AB.
 *
 * @param[in] ldab
 *          The leading dimension of the array AB. lda >= max(1,m).
 *
 *******************************************************************************
 *
 * @retval double
 *         The specified norm of the general band matrix A.
 *
 *******************************************************************************
 *
 * @sa plasma_omp_zlangb
 * @sa plasma_clangb
 * @sa plasma_dlangb
 * @sa plasma_slangb
 *
 ******************************************************************************/
double plasma_zlangb(plasma_enum_t norm,
                     int m, int n, int kl, int ku,
                     plasma_complex64_t *pAB, int ldab)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    // Check input arguments.
    if ((norm != PlasmaMaxNorm) && (norm != PlasmaOneNorm) &&
        (norm != PlasmaInfNorm) && (norm != PlasmaFrobeniusNorm) ) {
        plasma_error("illegal value of norm");
        return -1;
    }
    if (m < 0) {
        plasma_error("illegal value of m");
        return -2;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -3;
    }
    if (kl < 0) {
        plasma_error("illegal value of kl");
        return -4;
    }
    if (ku < 0) {
        plasma_error("illegal value of ku");
        return -5;
    }

    if (ldab < imax(1, 1+kl+ku)) {
        //printf("%d\n", ldab);
        plasma_error("illegal value of lda");
        return -7;
    }

    // quick return
    if (imin(n, m) == 0)
      return 0.0;

    // Set tiling parameters.
    int nb = plasma->nb;

    // Create tile matrices.
    plasma_desc_t AB;
    int tku = (ku+kl+nb-1)/nb; // number of tiles in upper band (not including diagonal)
    int tkl = (kl+nb-1)/nb;    // number of tiles in lower band (not including diagonal)
    int lm = (tku+tkl+1)*nb;
    int retval;
    retval = plasma_desc_general_band_create(PlasmaComplexDouble, PlasmaGeneral,
                                             nb, nb, lm, n, 0, 0, m, n, kl, ku, &AB);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        return retval;
    }

    // Allocate workspace.
    double *work = NULL;
    switch (norm) {
    case PlasmaMaxNorm:
        work = (double*)malloc((size_t)(AB.klt+AB.kut-1)*AB.nt*sizeof(double));
        break;

    case PlasmaOneNorm:
        work = (double*)calloc(((size_t)AB.n*(tku+tkl+1)+AB.n), sizeof(double)); //TODO: too much space.
        break;

    case PlasmaInfNorm:
        work = (double*)calloc(((size_t)AB.nt*AB.mt*AB.mb+AB.mb*AB.mt), sizeof(double));
        break;

    case PlasmaFrobeniusNorm:
        work = (double*)calloc((size_t)2*(tku+tkl+1)*AB.nt, sizeof(double));
        break;

    default:
        assert(0);
    }
    if (work == NULL) {
        plasma_error("malloc() failed");
        return PlasmaErrorOutOfMemory;
    }

    // Create sequence.
    plasma_sequence_t sequence;
    retval = plasma_sequence_init(&sequence);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_sequence_create() failed");
        return retval;
    }

    // Initialize request.
    plasma_request_t request;
    retval = plasma_request_init(&request);

    double value;

    // asynchronous block
    #pragma omp parallel
    #pragma omp master
    {
        // Translate to tile layout.
        plasma_omp_zpb2desc(pAB, ldab, AB, &sequence, &request);
        // Call tile async function.
        plasma_omp_zlangb(norm, AB, work, &value, &sequence, &request);
    }
    // implicit synchronization

    free(work);

    // Free matrix in tile layout.
    plasma_desc_destroy(&AB);

    // Return the norm.
    //printf("[plasma_zlangb]: value=%.3f\n", value);
    return value;
}

/***************************************************************************//**
 *
 * @ingroup plasma_langb
 *
 *  Calculates the max, one, infinity or Frobenius norm of a general band matrix.
 *  Non-blocking equivalent of plasma_zlangb(). May return before the
 *  computation is finished. Operates on matrices stored by tiles. All matrices
 *  are passed through descriptors. All dimensions are taken from the
 *  descriptors. Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] norm
 *          - PlasmaMaxNorm: Max norm
 *          - PlasmaOneNorm: One norm
 *          - PlasmaInfNorm: Infinity norm
 *          - PlasmaFrobeniusNorm: Frobenius norm
 *
 * @param[in] AB
 *          The descriptor of matrix A.
 *
 * @param[out] work
 *          Workspace of size:
 *          - PlasmaMaxNorm: (AB.klt+AB.kut-1)*A.nt
 *          - PlasmaOneNorm: AB.n*(tku+tkl+1)+AB.n
 *          - PlasmaInfNorm: AB.nt*AB.mt*AB.mb+AB.mb*AB.mt
 *          - PlasmaFrobeniusNorm: 2*(tku+tkl+1)*AB.nt
 *
 * @param[out] value
 *          The calculated value of the norm requested.
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
 * @sa plasma_zlangb
 * @sa plasma_omp_clangb
 * @sa plasma_omp_dlangb
 * @sa plasma_omp_slangb
 *
 ******************************************************************************/
void plasma_omp_zlangb(plasma_enum_t norm, plasma_desc_t AB,
                       double *work, double *value,
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
    if ((norm != PlasmaMaxNorm) && (norm != PlasmaOneNorm) &&
        (norm != PlasmaInfNorm) && (norm != PlasmaFrobeniusNorm)) {
        plasma_error("illegal value of norm");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (plasma_desc_check(AB) != PlasmaSuccess) {
        plasma_error("invalid descriptor AB");
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

    // quick return
    if (imin(AB.m, AB.n) == 0) {
        *value = 0.0;
        return;
    }

    // Call the parallel function.
    plasma_pzlangb(norm, AB, work, value, sequence, request);
}
