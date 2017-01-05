/**
 *
 * @file
 *
 *  PLASMA header.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of Manchester, Univ. of California Berkeley and
 *  Univ. of Colorado Denver.
 *
 * @precisions normal z -> s d c
 *
 **/
#ifndef ICL_CORE_LAPACK_Z_H
#define ICL_CORE_LAPACK_Z_H

#ifdef __cplusplus
extern "C" {
#endif

// LAPACK_GLOBAL is Fortran name mangling macro from LAPACKE

#ifndef LAPACK_zlassq
#define LAPACK_zlassq LAPACK_GLOBAL(zlassq, ZLASSQ)
#endif

void LAPACK_zlassq(int *n, const plasma_complex64_t *x, int *incx,
                   double *scale, double *sumsq);

// LAPACKE_zlascl not available in LAPACKE < 3.6.0
#ifndef LAPACK_zlascl
#define LAPACK_zlascl LAPACK_GLOBAL(zlascl, ZLASCL)
#endif

void LAPACK_zlascl(char* type, int* kl, int* ku,
                   double* cfrom, double* cto,
                   int* m, int* n,
                   plasma_complex64_t* A, int* lda,
                   int *info);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_CORE_LAPACK_Z_H
