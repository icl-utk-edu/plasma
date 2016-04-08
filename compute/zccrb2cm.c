/**
 *
 * @file zccrb2cm.c
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

/******************************************************************************/
int PLASMA_zccrb2cm_Async(PLASMA_desc *A, PLASMA_Complex64_t *Af77, int lda,
                          PLASMA_sequence *sequence, PLASMA_request *request)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        plasma_error("PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }

    // Check input arguments.
    if (plasma_desc_check(A) != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        plasma_error("invalid A");
        return -1;
    }
    if (Af77 == NULL) {
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        plasma_error("NULL A");
        return -2;
    }
    if (sequence == NULL) {
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        plasma_error("NULL sequence");
        return -4;
    }
    if (request == NULL) {
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        plasma_error("NULL request");
        return -5;
    }

    // Check sequence status.
    if (sequence->status == PLASMA_SUCCESS)
        request->status = PLASMA_SUCCESS;
    else
        return plasma_request_fail(sequence, request,
                                   PLASMA_ERR_SEQUENCE_FLUSHED);

    // quick return
    if (A->m == 0 || A->n == 0)
        return PLASMA_SUCCESS;

    // Call the parallel function.
    plasma_pzooccrb2cm(*A, Af77, lda, sequence, request);

    return PLASMA_SUCCESS;
}
