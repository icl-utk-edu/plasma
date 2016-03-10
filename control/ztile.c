/**
 *
 * @file ztile.h
 *
 *  PLASMA control routines.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver.
 *
 * @version 3.0.0
 * @author Jakub Kurzak
 * @date 2016-01-01
 * @precisions normal z -> s d c
 *
 **/

#include "async.h"
#include "descriptor.h"

/******************************************************************************/
void plasma_zooplap2tile(PLASMA_desc *descA, PLASMA_Complex64_t *A,
                        int mb, int nb, int lm, int ln,
                        int i, int j, int m, int n,
                        PLASMA_sequence sequence, PLASMA_request request)
{
    // Create the descriptor.
    *descA = plasma_desc_init(PlasmaComplexDouble,
                              mb, nb, mb*nb, lm, ln, i, j, m, n);

    // Allocate and call the translation.
    if (plasma_desc_mat_alloc(descA) == PLASMA_SUCCESS)
        plasma_pzlapack_to_tile(A, lm, descA, sequence, request);
}
