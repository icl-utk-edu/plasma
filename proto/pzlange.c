/**
 *
 * @file pzlange.c
 *
 *  PLASMA auxiliary routines
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of Manchester, Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 3.0.0
 * @author  Emmanuel Agullo
 * @author  Mathieu Faverge
 * @author  Maksims Abalenkovs
 * @date    2016-07-22
 * @precisions normal z -> s d c
 *
 **/
#include <stdlib.h>
#include <math.h>
#include "common.h"

#define A(m, n, i, j, ldt)  (BLKADDR(A, PLASMA_Complex64_t, m, n)+((j)*(ldt)+(i)))

/***************************************************************************//**
 *
 **/
void plasma_pzlange_quark(PLASMA_enum norm, PLASMA_desc A, double *work, double *result,
                          PLASMA_sequence *sequence, PLASMA_request *request)
{
    plasma_context_t *plasma;
    Quark_Task_Flags task_flags = Quark_Task_Flags_Initializer;

    double *lwork;
    int X, X1, X2, Y, Y1, Y2;
    int ldam;
    int m, n, k;
    int szeW;

    plasma = plasma_context_self();
    if (sequence->status != PLASMA_SUCCESS)
        return;
    QUARK_Task_Flag_Set(&task_flags, TASK_SEQUENCE, (intptr_t)sequence->quark_sequence);
    plasma_profile_by_function( &task_flags, LANGE );

    *result = 0.0;
    switch ( norm ) {
    /*
     *  PlasmaMaxNorm
     */
    case PlasmaMaxNorm:
        szeW = A.mt*A.nt;
        lwork = (double*)plasma_shared_alloc(plasma, szeW, PlasmaRealDouble);
        memset(lwork, 0, szeW*sizeof(double));
        for (m = 0; m < A.mt; m++) {
            X1 = m == 0      ?  A.i       %A.mb   : 0;
            X2 = m == A.mt-1 ? (A.i+A.m-1)%A.mb+1 : A.mb;
            X = X2 - X1;
            ldam = BLKLDD(A, m);
            for (n = 0; n < A.nt; n++) {
                Y1 = n == 0      ?  A.j       %A.nb   : 0;
                Y2 = n == A.nt-1 ? (A.j+A.n-1)%A.nb+1 : A.nb;
                Y = Y2 - Y1;
                QUARK_CORE_zlange_f1(
                    plasma->quark, &task_flags,
                    PlasmaMaxNorm, X, Y,
                    A(m, n, X1, Y1, ldam), ldam, ldam*Y, 0,
                    lwork + A.mt * n + m,
                    result, 1);
            }
        }
        QUARK_CORE_dlange(
            plasma->quark, &task_flags,
            PlasmaMaxNorm, A.mt, A.nt,
            lwork, A.mt, szeW,
            0, result);

        QUARK_CORE_free(
            plasma->quark, &task_flags,
            lwork, szeW*sizeof(PLASMA_Complex64_t));
        break;

    /*
     *  PlasmaOneNorm
     */
    case PlasmaOneNorm:
        lwork = (double*)plasma_shared_alloc(plasma, A.n, PlasmaRealDouble);
        memset(lwork, 0, A.n*sizeof(double));
        for (m = 0; m < A.mt; m++) {
            X1 = m == 0      ?  A.i       %A.mb   : 0;
            X2 = m == A.mt-1 ? (A.i+A.m-1)%A.mb+1 : A.mb;
            X = X2 - X1;
            ldam = BLKLDD(A, m);
            for (n = 0; n < A.nt; n++) {
                Y1 = n == 0      ?  A.j       %A.nb   : 0;
                Y2 = n == A.nt-1 ? (A.j+A.n-1)%A.nb+1 : A.nb;
                Y = Y2 - Y1;
                QUARK_CORE_dzasum_f1(
                    plasma->quark, &task_flags,
                    PlasmaColumnwise, PlasmaUpperLower, X, Y,
                    A(m, n, X1, Y1, ldam), ldam, ldam*Y,
                    lwork + n*A.nb, A.nb,
                    result, 1);
            }
        }
        QUARK_CORE_dlange(
            plasma->quark, &task_flags,
            PlasmaMaxNorm, A.n, 1,
            lwork, 1, A.n,
            0, result);

        QUARK_CORE_free(
            plasma->quark, &task_flags,
            lwork, A.n*sizeof(PLASMA_Complex64_t));
        break;
    /*
     *  PlasmaInfNorm
     */
    case PlasmaInfNorm:
        lwork = (double*)plasma_shared_alloc(plasma, A.m, PlasmaRealDouble);
        memset(lwork, 0, A.m*sizeof(double));
        for (m = 0; m < A.mt; m++) {
            X1 = m == 0      ?  A.i       %A.mb   : 0;
            X2 = m == A.mt-1 ? (A.i+A.m-1)%A.mb+1 : A.mb;
            X = X2 - X1;
            ldam = BLKLDD(A, m);
            for (n = 0; n < A.nt; n++) {
                Y1 = n == 0      ?  A.j       %A.nb   : 0;
                Y2 = n == A.nt-1 ? (A.j+A.n-1)%A.nb+1 : A.nb;
                Y = Y2 - Y1;
                QUARK_CORE_dzasum_f1(
                    plasma->quark, &task_flags,
                    PlasmaRowwise, PlasmaUpperLower, X, Y,
                    A(m, n, X1, Y1, ldam), ldam, ldam*Y,
                    lwork + m*A.mb, A.mb,
                    result, 1);
            }
        }
        QUARK_CORE_dlange(
            plasma->quark, &task_flags,
            PlasmaMaxNorm, A.m, 1,
            lwork, 1, A.m,
            0, result);

        QUARK_CORE_free(
            plasma->quark, &task_flags,
            lwork, A.m*sizeof(PLASMA_Complex64_t));
        break;
    /*
     *  PlasmaFrobeniusNorm
     */
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
            for (n = 0; n < A.nt; n++) {
                Y1 = n == 0      ?  A.j       %A.nb   : 0;
                Y2 = n == A.nt-1 ? (A.j+A.n-1)%A.nb+1 : A.nb;
                Y = Y2 - Y1;
                QUARK_CORE_zgessq_f1(
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
        QUARK_CORE_dplssq(
            plasma->quark, &task_flags,
            PLASMA_SIZE, lwork, result );

        QUARK_CORE_free(
            plasma->quark, &task_flags,
            lwork, szeW*sizeof(double));
        break;
    default:;
    }
}
