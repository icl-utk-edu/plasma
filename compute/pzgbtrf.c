/**
 *
 * @file pzpotrf.c
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> s d c
 *
 **/

#include "plasma_async.h"
#include "plasma_descriptor.h"
#include "plasma_types.h"
#include "plasma_internal.h"
#include "core_blas_z.h"

// number of tiles within band (panel needs 1+KLT tiles, KLT stores KL elements for the last column)
#define KUT ((A.ku+A.kl+A.nb-1)/A.nb) // number of tiles in upper band (not including diagonal)
#define KLT ((A.kl+A.nb-1)/A.nb)      // number of tiles in lower band (not including diagonal)

#define A(m,n) ((PLASMA_Complex64_t*)plasma_getaddr(A, KUT+(m)-(n), n))
#define BLKLDD_BAND(A, m, n) BLKLDD((A), KUT+(m)-(n))

#define fill(k) (&(fill[(k)]))
#define ipiv(k) (&(ipiv[A.mb*(k)]))

/***************************************************************************//**
 *  Parallel tile LU factorization of a band matrix.
 * @see PLASMA_zgbtrf_Tile_Async
 ******************************************************************************/
void plasma_pzgbtrf(PLASMA_desc A, int *ipiv, int *fill, int *fake,
                    PLASMA_sequence *sequence, PLASMA_request *request)
{
    int k, m, n, minmn;
    int tempm, tempkm, tempkn, tempmm, tempnn;

    PLASMA_Complex64_t zone  = (PLASMA_Complex64_t)1.0;
    PLASMA_Complex64_t mzone = (PLASMA_Complex64_t)-1.0;

    if (sequence->status != PLASMA_SUCCESS)
        return;

    minmn = imin(A.mt, A.nt);
    for (k = 0; k < minmn; k++)
    {
        tempkm = k == A.mt-1 ? A.m-k*A.mb : A.mb;
        tempkn = k == A.nt-1 ? A.n-k*A.nb : A.nb;
        tempm  = imin(A.m - k * A.mb, tempkm+A.kl);

        /*
         * Factorize the panel
         */
        CORE_OMP_zgetrf_tile(
            // sub-matrix for panel, A(k:mt, k)
            A, k, tempm, tempkn, 
            ipiv(k),
            // to keep track of fills
            A.ku, A.n, (k == 0 ? fill(k) : fill(k-1)), fill(k),
            // error tracking..
            k*A.mb,
            // fake dependency on k-th panel (output)
            &fake[k]);

        /*
         * Update the trailing submatrix
         */
        for (n = k+1; n < imin(A.nt, k+A.kut); n++)
        {
            /*
             * Apply row interchange after the panel (work on the panel)
             */
            tempnn = n == A.nt-1 ? A.n-n*A.nb : A.nb;
            CORE_OMP_zswptr_ontile_fill(
                // sub-matrix for n-th panel
                A, k, n, tempm, tempnn,
                // for laswp
                1, tempkm, ipiv(k), 1,
                // for trsm
                A(k, k), BLKLDD_BAND(A, k, k),
                // to keep track of current fill 
                n, fill(k),
                // fake dependency on n-th panel (input)
                &fake[n]);

            /*
             * Update after the row interchange
             */
            for (m = imax(k+1,n-A.kut); m < imin(k+A.klt, A.mt); m++)
            {
                tempmm = m == A.mt-1 ? A.m-m*A.mb : A.mb;
                CORE_OMP_zgemm_fill(
                    PlasmaNoTrans, PlasmaNoTrans,
                    tempmm, tempnn, A.nb, 
                    mzone, A(m, k), BLKLDD_BAND(A, m, k),
                           A(k, n), BLKLDD_BAND(A, k, n),
                    zone,  A(m, n), BLKLDD_BAND(A, m, n),
                    /* to keep track of current fill */
                    k, m, n, fill(k),
                    /* Dependency on next swap, or panel when k+1 == n 
                     (all gemms on the n-th column need to be done before) */
                    &fake[n]);
                    //A(k+1, n), A.mb*A.nb, INOUT | GATHERV);
            }
        }
    }
}

