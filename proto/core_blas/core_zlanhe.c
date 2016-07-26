/**
 *
 * @file qwrapper_zlanhe.c
 *
 *  PLASMA core_blas quark wrapper
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of Manchester, Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 3.0.0
 * @author  Julien Langou
 * @author  Henricus Bouwmeester
 * @author  Mathieu Faverge
 * @author  Maksims Abalenkovs
 * @date    2016-07-22
 * @precisions normal z -> c
 *
 **/

#include "core_blas.h"
#include "plasma_types.h"

#ifdef PLASMA_WITH_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

/***************************************************************************//**
 *
 **/
void CORE_zlanhe(int norm, PLASMA_enum uplo, int n,
                 PLASMA_Complex64_t *A, int lda, double *work, double *normA)
{
    *normA = LAPACKE_zlanhe_work(
        LAPACK_COL_MAJOR,
        lapack_const(norm), lapack_const(uplo),
        n, A, lda, work);
}

/***************************************************************************//**
 *
 **/
void CORE_OMP_zlanhe(int norm, PLASMA_enum uplo, int n,
                     const PLASMA_Complex64_t *A, int lda, int szeA,
                     int szeW, double *result)
{
    szeW = max(1, szeW);

    double *work = (double *) malloc(sizeof(double)*szeW);

#pragma omp task depend(in:A[0:szeA]) depend(out:result)
    CORE_zlanhe(norm, uplo, n, A, lda, work, result);

    free(work);
}

/***************************************************************************//**
 *
 **/
void CORE_zlanhe_f1(double *normA, int norm, PLASMA_enum uplo, int n,
                    PLASMA_Complex64_t *A, int lda, double *work, double *fake)
{
    *normA = LAPACKE_zlanhe_work(
        LAPACK_COL_MAJOR,
        lapack_const(norm), lapack_const(uplo),
        n, A, lda, work);
}

/***************************************************************************//**
 *
 **/
void CORE_OMP_zlanhe_f1(PLASMA_enum norm, PLASMA_enum uplo, int n,
                        const PLASMA_Complex64_t *A, int lda, int szeA,
                        int szeW, double *result,
                        double *fake, int szeF)
{
    szeW = max(1, szeW);

    double *work = (double *) malloc(sizeof(double)*szeW);

    if (result == fake) {

#pragma omp task depend(in:A[0:szeA]) depend(out:result[0:szeF]) reduction(+:result)
        CORE_zlanhe(norm, uplo, n, A, lda, work, result);

    } else {

        double *fake = (double *) malloc(sizeof(double)*szeF);

#pragma omp task depend(in:A[0:szeA]) depend(out:result) depend(out:fake[0:szeF]) reduction(+:fake)
        CORE_zlanhe_f1(result, norm, uplo, n, A, lda, work, fake);

        free(fake);
    }

    free(work);
}
