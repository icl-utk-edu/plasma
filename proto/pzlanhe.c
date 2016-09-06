/**
 *
 * @file pzlanhe.c
 *
 *  PLASMA auxiliary routines
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of Manchester, Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 3.0.0
 * @author  Emmanuel Agullo
 * @author  Mathieu Faverge
 * @author  Maksims Abalenkovs
 * @date    2016-07-21
 * @precisions normal z -> c
 *
 **/

#include "plasma_async.h"
#include "plasma_descriptor.h"
#include "plasma_types.h"
#include "plasma_internal.h"
#include "core_blas_z.h"

#define A(m, n, i, j, ldt)  (BLKADDR(A, PLASMA_Complex64_t, m, n)+((j)*(ldt)+(i)))

/***************************************************************************//**
 *
 **/
void plasma_pzlanhe(PLASMA_enum norm, PLASMA_enum uplo, PLASMA_desc A,
                    double *work, double *result,
                    PLASMA_sequence *sequence, PLASMA_request *request)
{
    plasma_context_t *plasma;

    double* lwork;
    int X, X1, X2, Y, Y1;
    int ldam;
    int m, n, k;
    int szeW, pos;

    plasma = plasma_context_self();
    if (sequence->status != PLASMA_SUCCESS)
        return;

    *result = 0.0;
    switch ( norm ) {
    //===============
    // PlasmaMaxNorm
    //===============
    case PlasmaMaxNorm:
        szeW = (A.mt*(A.mt+1))/2;
        pos = 0;
        lwork = (double *)plasma_shared_alloc(plasma, szeW, PlasmaRealDouble);
        memset(lwork, 0, szeW*sizeof(double));
        for (m = 0; m < A.mt; m++) {
            X1 = m == 0      ?  A.i       %A.mb   : 0;
            X2 = m == A.mt-1 ? (A.i+A.m-1)%A.mb+1 : A.mb;
            X = X2 - X1;
            ldam = BLKLDD(A, m);
            OMP_CORE_zlanhe_f1(
                plasma->quark, &task_flags,
                PlasmaMaxNorm, uplo, X,
                A(m, m, X1, X1, ldam), ldam, ldam*X,
                0, lwork + pos,
                result, 1);
            pos++;
            //=============
            // PlasmaLower
            //=============
            if (uplo == PlasmaLower) {
                for (n = 0; n < m; n++) {
                    Y1 = n == 0 ? A.j%A.nb : 0;
                    Y  = A.nb - Y1;
                    OMP_CORE_zlange_f1(
                        plasma->quark, &task_flags,
                        PlasmaMaxNorm, X, Y,
                        A(m, n, X1, Y1, ldam), ldam, ldam*Y,
                        0, lwork + pos,
                        result, 1);
                    pos++;
                }
            }
            //=============
            // PlasmaUpper
            //=============
            else {
                for (n = m+1; n < A.mt; n++) {
                    Y = n == A.nt-1 ? (A.j+A.n-1)%A.nb+1 : A.nb;
                    OMP_CORE_zlange_f1(
                        plasma->quark, &task_flags,
                        PlasmaMaxNorm, X, Y,
                        A(m, n, X1, 0, ldam), ldam, ldam*Y,
                        0, lwork + pos,
                        result, 1);
                    pos++;
                }
            }
        }
        OMP_CORE_dlange(
            plasma->quark, &task_flags,
            PlasmaMaxNorm, szeW, 1,
            lwork, 1, szeW,
            0, result);

        OMP_CORE_free(
            plasma->quark, &task_flags,
            lwork, szeW*sizeof(PLASMA_Complex64_t));
        break;
    //===============================
    // PlasmaOneNorm / PlasmaInfNorm
    //===============================
    case PlasmaOneNorm:
    case PlasmaInfNorm:
        lwork = (double *)plasma_shared_alloc(plasma, A.m, PlasmaRealDouble);
        memset(lwork, 0, A.m*sizeof(double));
        for (m = 0; m < A.mt; m++) {
            X1 = m == 0      ?  A.i       %A.mb   : 0;
            X2 = m == A.mt-1 ? (A.i+A.m-1)%A.mb+1 : A.mb;
            X = X2 - X1;
            ldam = BLKLDD(A, m);
            OMP_CORE_dzasum_f1(
                plasma->quark, &task_flags,
                PlasmaRowwise, uplo, X, X,
                A(m, m, X1, X1, ldam), ldam, ldam*X,
                lwork + m*A.mb, A.mb,
                result, 1);
            //=============
            // PlasmaLower
            //=============
            if (uplo == PlasmaLower) {
                for (n = 0; n < m; n++) {
                    Y1 = n == 0 ? A.j%A.nb : 0;
                    Y = A.nb - Y1;
                    OMP_CORE_dzasum_f1(
                        plasma->quark, &task_flags,
                        PlasmaRowwise, PlasmaUpperLower, X, Y,
                        A(m, n, X1, Y1, ldam), ldam, ldam*Y,
                        lwork + m*A.mb, A.mb,
                        result, 1);

                    OMP_CORE_dzasum_f1(
                        plasma->quark, &task_flags,
                        PlasmaColumnwise, PlasmaUpperLower, X, Y,
                        A(m, n, X1, Y1, ldam), ldam, ldam*Y,
                        lwork + n*A.mb, A.mb,
                        result, 1);
                }
            }
            //=============
            // PlasmaUpper
            //=============
            else {
                for (n = m+1; n < A.mt; n++) {
                    Y = n == A.nt-1 ? (A.j+A.n-1)%A.nb+1 : A.nb;
                    OMP_CORE_dzasum_f1(
                        plasma->quark, &task_flags,
                        PlasmaRowwise, PlasmaUpperLower, X, Y,
                        A(m, n, X1, 0, ldam), ldam, ldam*Y,
                        lwork + m*A.mb, A.mb,
                        result, 1);

                    OMP_CORE_dzasum_f1(
                        plasma->quark, &task_flags,
                        PlasmaColumnwise, PlasmaUpperLower, X, Y,
                        A(m, n, X1, 0, ldam), ldam, ldam*Y,
                        lwork + n*A.mb, A.mb,
                        result, 1);
                }
            }
        }
        OMP_CORE_dlange(
            plasma->quark, &task_flags,
            PlasmaMaxNorm, A.m, 1,
            lwork, 1, A.m,
            0, result);

        OMP_CORE_free(
            plasma->quark, &task_flags,
            lwork, A.m*sizeof(PLASMA_Complex64_t));
        break;
    //=====================
    // PlasmaFrobeniusNorm
    //=====================
    case PlasmaFrobeniusNorm:
        szeW = 2*(PLASMA_SIZE);
        lwork = (double*)plasma_shared_alloc(plasma, szeW, PlasmaRealDouble);

        for (m = 0; m < PLASMA_SIZE; m++) {
            lwork[2*m  ] = 0.;
            lwork[2*m+1] = 1.;
        }

        k = 0;
        for (m = 0; m < A.mt; m++) {
            X1 = m == 0      ?  A.i       %A.mb   : 0;
            X2 = m == A.mt-1 ? (A.i+A.m-1)%A.mb+1 : A.mb;
            X = X2 - X1;
            ldam = BLKLDD(A, m);

            OMP_CORE_zhessq_f1(
                plasma->quark, &task_flags,
                uplo, X,
                A(m, m, X1, X1, ldam), ldam,
                lwork + 2*k,
                lwork + 2*k + 1,
                result, 1, INOUT | GATHERV );
            k++;
            k = k % PLASMA_SIZE;

            //=============
            // PlasmaLower
            //=============
            if (uplo == PlasmaLower) {
                for (n = 0; n < m; n++) {
                    Y1 = n == 0 ? A.j%A.nb : 0;
                    Y = A.nb - Y1;

                    OMP_CORE_zgessq_f1(
                        plasma->quark, &task_flags,
                        X, Y,
                        A(m, n, X1, Y1, ldam), ldam,
                        lwork + 2*k,
                        lwork + 2*k + 1,
                        result, 1, INOUT | GATHERV );

                    OMP_CORE_zgessq_f1(
                        plasma->quark, &task_flags,
                        X, Y,
                        A(m, n, X1, Y1, ldam), ldam,
                        lwork + 2*k,
                        lwork + 2*k + 1,
                        result, 1, INOUT | GATHERV );
                    k++;
                    k = k % PLASMA_SIZE;

                }
            }
            //=============
            // PlasmaUpper
            //=============
            else {
                for (n = m+1; n < A.mt; n++) {
                    Y = n == A.nt-1 ? (A.j+A.n-1)%A.nb+1 : A.nb;

                    OMP_CORE_zgessq_f1(
                        plasma->quark, &task_flags,
                        X, Y,
                        A(m, n, X1, 0, ldam), ldam,
                        lwork + 2*k,
                        lwork + 2*k + 1,
                        result, 1, INOUT | GATHERV );

                    OMP_CORE_zgessq_f1(
                        plasma->quark, &task_flags,
                        X, Y,
                        A(m, n, X1, 0, ldam), ldam,
                        lwork + 2*k,
                        lwork + 2*k + 1,
                        result, 1, INOUT | GATHERV );
                    k++;
                    k = k % PLASMA_SIZE;

                }
            }
        }
        OMP_CORE_dplssq(
            plasma->quark, &task_flags,
            PLASMA_SIZE, lwork, result );

        OMP_CORE_free(
            plasma->quark, &task_flags,
            lwork, szeW*sizeof(double));
    default:;
    }
}
