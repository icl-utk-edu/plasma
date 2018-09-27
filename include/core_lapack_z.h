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

#ifndef PLASMA_CORE_LAPACK_Z_H
#define PLASMA_CORE_LAPACK_Z_H

#ifdef __cplusplus
extern "C" {
#endif

// LAPACK_GLOBAL is Fortran name mangling macro from LAPACKE

// LAPACKE_zlantr broken (returns 0) in LAPACKE < 3.6.1
#ifndef LAPACK_zlantr
#define LAPACK_zlantr LAPACK_GLOBAL(zlantr, ZLANTR)
double LAPACK_zlantr(const char *norm, const char *uplo, const char *diag,
                     const lapack_int *m, const lapack_int *n,
                     const plasma_complex64_t *A, const lapack_int *lda,
                     double *work);
#endif

// LAPACKE_zlascl not available in LAPACKE < 3.6.0
#ifndef LAPACK_zlascl
#define LAPACK_zlascl LAPACK_GLOBAL(zlascl, ZLASCL)
void LAPACK_zlascl(const char *type, const lapack_int *kl, const lapack_int *ku,
                   const double *cfrom, const double *cto,
                   const lapack_int *m, const lapack_int *n,
                   plasma_complex64_t *A, const lapack_int *lda,
                   lapack_int *info);
#endif

// LAPACKE_zlassq not available yet
#ifndef LAPACK_zlassq
#define LAPACK_zlassq LAPACK_GLOBAL(zlassq, ZLASSQ)
void LAPACK_zlassq(const lapack_int *n, const plasma_complex64_t *x, const lapack_int *incx,
                   double *scale, double *sumsq);
#endif

// LAPACKE_zlangb not available yet
#ifndef LAPACK_zlangb
#define LAPACK_zlangb LAPACK_GLOBAL(zlangb, ZLANGB)
double LAPACK_zlangb(const char *norm,
                     const lapack_int *n, const lapack_int *kl, const lapack_int *ku,
                     const plasma_complex64_t *A, const lapack_int *lda,
                     double *work);

#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // PLASMA_CORE_LAPACK_Z_H
