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

#ifndef ICL_CORE_LAPACK_Z_H
#define ICL_CORE_LAPACK_Z_H

#ifdef __cplusplus
extern "C" {
#endif

// LAPACK_GLOBAL is Fortran name mangling macro from LAPACKE

// LAPACKE_zlantr broken (returns 0) in LAPACKE < 3.6.1
#ifndef LAPACK_zlantr
#define LAPACK_zlantr LAPACK_GLOBAL(zlantr, ZLANTR)
double LAPACK_zlantr(const char *norm, const char *uplo, const char *diag,
                     const int *m, const int *n,
                     const plasma_complex64_t *A, const int *lda,
                     double *work);
#endif

// LAPACKE_zlascl not available in LAPACKE < 3.6.0
#ifndef LAPACK_zlascl
#define LAPACK_zlascl LAPACK_GLOBAL(zlascl, ZLASCL)
void LAPACK_zlascl(const char *type, const int *kl, const int *ku,
                   const double *cfrom, const double *cto,
                   const int *m, const int *n,
                   plasma_complex64_t *A, const int *lda,
                   int *info);
#endif

// LAPACKE_zlassq not available yet
#ifndef LAPACK_zlassq
#define LAPACK_zlassq LAPACK_GLOBAL(zlassq, ZLASSQ)
void LAPACK_zlassq(const int *n, const plasma_complex64_t *x, const int *incx,
                   double *scale, double *sumsq);
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_CORE_LAPACK_Z_H
