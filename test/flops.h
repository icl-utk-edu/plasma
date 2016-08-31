/**
 *
 * @file flops.h
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#ifndef ICL_PLASMA_FLOPS_H
#define ICL_PLASMA_FLOPS_H

#include "plasma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// Level 2 BLAS
// Generic formulas come from LAWN 41
//==============================================================================

//------------------------------------------------------------ gemv
static double fmuls_gemv(double m, double n)
    { return m*n + 2.*m; }

static double fadds_gemv(double m, double n)
    { return m*n; }

static double  flops_zgemv(double m, double n)
    { return 6.*fmuls_gemv(m, n) + 2.*fadds_gemv(m, n); }

static double  flops_cgemv(double m, double n)
    { return 6.*fmuls_gemv(m, n) + 2.*fadds_gemv(m, n); }

static double  flops_dgemv(double m, double n)
    { return    fmuls_gemv(m, n) +    fadds_gemv(m, n); }

static double  flops_sgemv(double m, double n)
    { return    fmuls_gemv(m, n) +    fadds_gemv(m, n); }

//------------------------------------------------------------ symv/hemv
static double fmuls_symv(double n)
    { return fmuls_gemv(n, n); }

static double fadds_symv(double n)
    { return fadds_gemv(n, n); }

static double fmuls_hemv(double n)
    { return fmuls_symv(n); }

static double fadds_hemv(double n)
    { return fadds_symv(n); }

static double  flops_zhemv(double n)
    { return 6.*fmuls_hemv(n) + 2.*fadds_hemv(n); }

static double  flops_chemv(double n)
    { return 6.*fmuls_hemv(n) + 2.*fadds_hemv(n); }

static double  flops_zsymv(double n)
    { return 6.*fmuls_symv(n) + 2.*fadds_symv(n); }

static double  flops_csymv(double n)
    { return 6.*fmuls_symv(n) + 2.*fadds_symv(n); }

static double  flops_dsymv(double n)
    { return    fmuls_symv(n) +    fadds_symv(n); }

static double  flops_ssymv(double n)
    { return    fmuls_symv(n) +    fadds_symv(n); }

//==============================================================================
// Level 3 BLAS
//==============================================================================

//------------------------------------------------------------ gemm
static double fmuls_gemm(double m, double n, double k)
    { return m*n*k; }

static double fadds_gemm(double m, double n, double k)
    { return m*n*k; }

static double  flops_zgemm(double m, double n, double k)
    { return 6.*fmuls_gemm(m, n, k) + 2.*fadds_gemm(m, n, k); }

static double  flops_cgemm(double m, double n, double k)
    { return 6.*fmuls_gemm(m, n, k) + 2.*fadds_gemm(m, n, k); }

static double  flops_dgemm(double m, double n, double k)
    { return    fmuls_gemm(m, n, k) +    fadds_gemm(m, n, k); }

static double  flops_sgemm(double m, double n, double k)
    { return    fmuls_gemm(m, n, k) +    fadds_gemm(m, n, k); }

//------------------------------------------------------------ symm/hemm
static double fmuls_symm(PLASMA_enum side, double m, double n)
{
    return (side == PlasmaLeft)
        ? fmuls_gemm(m, m, n)
        : fmuls_gemm(m, n, n);
}

static double fadds_symm(PLASMA_enum side, double m, double n)
{
    return (side == PlasmaLeft)
        ? fadds_gemm(m, m, n)
        : fadds_gemm(m, n, n);
}

static double fmuls_hemm(PLASMA_enum side, double m, double n)
    { return fmuls_symm(side, m, n); }

static double fadds_hemm(PLASMA_enum side, double m, double n)
    { return fadds_symm(side, m, n); }

static double  flops_zhemm(PLASMA_enum side, double m, double n)
    { return 6.*fmuls_hemm(side, m, n) + 2.*fadds_hemm(side, m, n); }

static double  flops_chemm(PLASMA_enum side, double m, double n)
    { return 6.*fmuls_hemm(side, m, n) + 2.*fadds_hemm(side, m, n); }

static double  flops_zsymm(PLASMA_enum side, double m, double n)
    { return 6.*fmuls_symm(side, m, n) + 2.*fadds_symm(side, m, n); }

static double  flops_csymm(PLASMA_enum side, double m, double n)
    { return 6.*fmuls_symm(side, m, n) + 2.*fadds_symm(side, m, n); }

static double  flops_dsymm(PLASMA_enum side, double m, double n)
    { return    fmuls_symm(side, m, n) +    fadds_symm(side, m, n); }

static double  flops_ssymm(PLASMA_enum side, double m, double n)
    { return    fmuls_symm(side, m, n) +    fadds_symm(side, m, n); }

//------------------------------------------------------------ syrk/herk
static double fmuls_syrk(double k, double n)
    { return 0.5*k*n*(n + 1); }

static double fadds_syrk(double k, double n)
    { return 0.5*k*n*(n + 1); }

static double fmuls_herk(double k, double n)
    { return fmuls_syrk(k, n); }

static double fadds_herk(double k, double n)
    { return fadds_syrk(k, n); }

static double  flops_zherk(double k, double n)
    { return 6.*fmuls_herk(k, n) + 2.*fadds_herk(k, n); }

static double  flops_cherk(double k, double n)
    { return 6.*fmuls_herk(k, n) + 2.*fadds_herk(k, n); }

static double  flops_zsyrk(double k, double n)
    { return 6.*fmuls_syrk(k, n) + 2.*fadds_syrk(k, n); }

static double  flops_csyrk(double k, double n)
    { return 6.*fmuls_syrk(k, n) + 2.*fadds_syrk(k, n); }

static double  flops_dsyrk(double k, double n)
    { return    fmuls_syrk(k, n) +    fadds_syrk(k, n); }

static double  flops_ssyrk(double k, double n)
    { return    fmuls_syrk(k, n) +    fadds_syrk(k, n); }

//------------------------------------------------------------ syr2k/her2k
static double fmuls_syr2k(double k, double n)
    { return k*n*n; }

static double fadds_syr2k(double k, double n)
    { return k*n*n + n; }

static double fmuls_her2k(double k, double n)
    { return fmuls_syr2k(k, n); }

static double fadds_her2k(double k, double n)
    { return fadds_syr2k(k, n); }

static double  flops_zher2k(double k, double n)
    { return 6.*fmuls_her2k(k, n) + 2.*fadds_her2k(k, n); }

static double  flops_cher2k(double k, double n)
    { return 6.*fmuls_her2k(k, n) + 2.*fadds_her2k(k, n); }

static double  flops_zsyr2k(double k, double n)
    { return 6.*fmuls_syr2k(k, n) + 2.*fadds_syr2k(k, n); }

static double  flops_csyr2k(double k, double n)
    { return 6.*fmuls_syr2k(k, n) + 2.*fadds_syr2k(k, n); }

static double  flops_dsyr2k(double k, double n)
    { return    fmuls_syr2k(k, n) +    fadds_syr2k(k, n); }

static double  flops_ssyr2k(double k, double n)
    { return    fmuls_syr2k(k, n) +    fadds_syr2k(k, n); }

//------------------------------------------------------------ trmm
static double fmuls_trmm_2(double m, double n)
    { return 0.5*n*m*(m + 1); }

static double fadds_trmm_2(double m, double n)
    { return 0.5*n*m*(m - 1); }

static double fmuls_trmm(PLASMA_enum side, double m, double n)
{
    return (side == PlasmaLeft)
        ? fmuls_trmm_2(m, n)
        : fmuls_trmm_2(n, m);
}

static double fadds_trmm(PLASMA_enum side, double m, double n)
{
    return (side == PlasmaLeft)
        ? fadds_trmm_2(m, n)
        : fadds_trmm_2(n, m);
}

static double  flops_ztrmm(PLASMA_enum side, double m, double n)
    { return 6.*fmuls_trmm(side, m, n) + 2.*fadds_trmm(side, m, n); }

static double  flops_ctrmm(PLASMA_enum side, double m, double n)
    { return 6.*fmuls_trmm(side, m, n) + 2.*fadds_trmm(side, m, n); }

static double  flops_dtrmm(PLASMA_enum side, double m, double n)
    { return    fmuls_trmm(side, m, n) +    fadds_trmm(side, m, n); }

static double  flops_strmm(PLASMA_enum side, double m, double n)
    { return    fmuls_trmm(side, m, n) +    fadds_trmm(side, m, n); }

//------------------------------------------------------------ trsm
static double fmuls_trsm(PLASMA_enum side, double m, double n)
    { return fmuls_trmm(side, m, n); }

static double fadds_trsm(PLASMA_enum side, double m, double n)
    { return fadds_trmm(side, m, n); }

static double  flops_ztrsm(PLASMA_enum side, double m, double n)
    { return 6.*fmuls_trsm(side, m, n) + 2.*fadds_trsm(side, m, n); }

static double  flops_ctrsm(PLASMA_enum side, double m, double n)
    { return 6.*fmuls_trsm(side, m, n) + 2.*fadds_trsm(side, m, n); }

static double  flops_dtrsm(PLASMA_enum side, double m, double n)
    { return    fmuls_trsm(side, m, n) +    fadds_trsm(side, m, n); }

static double  flops_strsm(PLASMA_enum side, double m, double n)
    { return    fmuls_trsm(side, m, n) +    fadds_trsm(side, m, n); }

//==============================================================================
// LAPACK
//==============================================================================

//------------------------------------------------------------ getrf
static double fmuls_getrf(double m, double n)
{
    return (m < n)
        ? (0.5*m*(m*(n - 1./3.*m - 1.) + n) + 2./3.*m)
        : (0.5*n*(n*(m - 1./3.*n - 1.) + m) + 2./3.*n);
}

static double fadds_getrf(double m, double n)
{
    return (m < n)
        ? (0.5*m*(m*(n - 1./3.*m) - n) + 1./6.*m)
        : (0.5*n*(n*(m - 1./3.*n) - m) + 1./6.*n);
}

static double  flops_zgetrf(double m, double n)
    { return 6.*fmuls_getrf(m, n) + 2.*fadds_getrf(m, n); }

static double  flops_cgetrf(double m, double n)
    { return 6.*fmuls_getrf(m, n) + 2.*fadds_getrf(m, n); }

static double  flops_dgetrf(double m, double n)
    { return    fmuls_getrf(m, n) +    fadds_getrf(m, n); }

static double  flops_sgetrf(double m, double n)
    { return    fmuls_getrf(m, n) +    fadds_getrf(m, n); }

//------------------------------------------------------------ getri
static double fmuls_getri(double n)
    { return n*(5./6. + n*(2./3.*n + 0.5)); }

static double fadds_getri(double n)
    { return n*(5./6. + n*(2./3.*n - 1.5)); }

static double  flops_zgetri(double n)
    { return 6.*fmuls_getri(n) + 2.*fadds_getri(n); }

static double  flops_cgetri(double n)
    { return 6.*fmuls_getri(n) + 2.*fadds_getri(n); }

static double  flops_dgetri(double n)
    { return    fmuls_getri(n) +    fadds_getri(n); }

static double  flops_sgetri(double n)
    { return    fmuls_getri(n) +    fadds_getri(n); }

//------------------------------------------------------------ getrs
static double fmuls_getrs(double n, double nrhs)
    { return nrhs*n*n; }

static double fadds_getrs(double n, double nrhs)
    { return nrhs*n*(n - 1); }

static double  flops_zgetrs(double n, double nrhs)
    { return 6.*fmuls_getrs(n, nrhs) + 2.*fadds_getrs(n, nrhs); }

static double  flops_cgetrs(double n, double nrhs)
    { return 6.*fmuls_getrs(n, nrhs) + 2.*fadds_getrs(n, nrhs); }

static double  flops_dgetrs(double n, double nrhs)
    { return    fmuls_getrs(n, nrhs) +    fadds_getrs(n, nrhs); }

static double  flops_sgetrs(double n, double nrhs)
    { return    fmuls_getrs(n, nrhs) +    fadds_getrs(n, nrhs); }

//------------------------------------------------------------ potrf
static double fmuls_potrf(double n)
    { return n*((1./6.*n + 0.5)*n + 1./3.); }

static double fadds_potrf(double n)
    { return n*((1./6.*n)*n - 1./6.); }

static double  flops_zpotrf(double n)
    { return 6.*fmuls_potrf(n) + 2.*fadds_potrf(n); }

static double  flops_cpotrf(double n)
    { return 6.*fmuls_potrf(n) + 2.*fadds_potrf(n); }

static double  flops_dpotrf(double n)
    { return    fmuls_potrf(n) +    fadds_potrf(n); }

static double  flops_spotrf(double n)
    { return    fmuls_potrf(n) +    fadds_potrf(n); }

//------------------------------------------------------------ potri
static double fmuls_potri(double n)
    { return n*(2./3. + n*(1./3.*n + 1.0)); }

static double fadds_potri(double n)
    { return n*(1./6. + n*(1./3.*n - 0.5)); }

static double  flops_zpotri(double n)
    { return 6.*fmuls_potri(n) + 2.*fadds_potri(n); }

static double  flops_cpotri(double n)
    { return 6.*fmuls_potri(n) + 2.*fadds_potri(n); }

static double  flops_dpotri(double n)
    { return    fmuls_potri(n) +    fadds_potri(n); }

static double  flops_spotri(double n)
    { return    fmuls_potri(n) +    fadds_potri(n); }

//------------------------------------------------------------ potrs
static double fmuls_potrs(double n, double nrhs)
    { return nrhs*n*(n + 1); }

static double fadds_potrs(double n, double nrhs)
    { return nrhs*n*(n - 1); }

static double  flops_zpotrs(double n, double nrhs)
    { return 6.*fmuls_potrs(n, nrhs) + 2.*fadds_potrs(n, nrhs); }

static double  flops_cpotrs(double n, double nrhs)
    { return 6.*fmuls_potrs(n, nrhs) + 2.*fadds_potrs(n, nrhs); }

static double  flops_dpotrs(double n, double nrhs)
    { return    fmuls_potrs(n, nrhs) +    fadds_potrs(n, nrhs); }

static double  flops_spotrs(double n, double nrhs)
    { return    fmuls_potrs(n, nrhs) +    fadds_potrs(n, nrhs); }

//------------------------------------------------------------ geqrf
static double fmuls_geqrf(double m, double n)
{
    return (m > n)
        ? (n*(n*( 0.5 - 1./3.*n + m) +    m + 23./6.))
        : (m*(m*(-0.5 - 1./3.*m + n) + 2.*n + 23./6.));
}

static double fadds_geqrf(double m, double n)
{
    return (m > n)
        ? (n*(n*( 0.5 - 1./3.*n + m)        +  5./6.))
        : (m*(m*(-0.5 - 1./3.*m + n) +    n +  5./6.));
}

static double  flops_zgeqrf(double m, double n)
    { return 6.*fmuls_geqrf(m, n) + 2.*fadds_geqrf(m, n); }

static double  flops_cgeqrf(double m, double n)
    { return 6.*fmuls_geqrf(m, n) + 2.*fadds_geqrf(m, n); }

static double  flops_dgeqrf(double m, double n)
    { return    fmuls_geqrf(m, n) +    fadds_geqrf(m, n); }

static double  flops_sgeqrf(double m, double n)
    { return    fmuls_geqrf(m, n) +    fadds_geqrf(m, n); }

//------------------------------------------------------------ geqrt
static double fmuls_geqrt(double m, double n)
    { return 0.5*m*n; }

static double fadds_geqrt(double m, double n)
    { return 0.5*m*n; }

static double  flops_zgeqrt(double m, double n)
    { return 6.*fmuls_geqrt(m, n) + 2.*fadds_geqrt(m, n); }

static double  flops_cgeqrt(double m, double n)
    { return 6.*fmuls_geqrt(m, n) + 2.*fadds_geqrt(m, n); }

static double  flops_dgeqrt(double m, double n)
    { return    fmuls_geqrt(m, n) +    fadds_geqrt(m, n); }

static double  flops_sgeqrt(double m, double n)
    { return    fmuls_geqrt(m, n) +    fadds_geqrt(m, n); }

//------------------------------------------------------------ geqlf
static double fmuls_geqlf(double m, double n)
    { return fmuls_geqrf(m, n); }

static double fadds_geqlf(double m, double n)
    { return fadds_geqrf(m, n); }

static double  flops_zgeqlf(double m, double n)
    { return 6.*fmuls_geqlf(m, n) + 2.*fadds_geqlf(m, n); }

static double  flops_cgeqlf(double m, double n)
    { return 6.*fmuls_geqlf(m, n) + 2.*fadds_geqlf(m, n); }

static double  flops_dgeqlf(double m, double n)
    { return    fmuls_geqlf(m, n) +    fadds_geqlf(m, n); }

static double  flops_sgeqlf(double m, double n)
    { return    fmuls_geqlf(m, n) +    fadds_geqlf(m, n); }

//------------------------------------------------------------ gerqf
static double fmuls_gerqf(double m, double n)
{
    return (m > n)
        ? (n*(n*( 0.5 - 1./3.*n + m) +    m + 29./6.))
        : (m*(m*(-0.5 - 1./3.*m + n) + 2.*n + 29./6.));
}

static double fadds_gerqf(double m, double n)
{
    return (m > n)
        ? (n*(n*(-0.5 - 1./3.*n + m) +    m +  5./6.))
        : (m*(m*( 0.5 - 1./3.*m + n) +      +  5./6.));
}

static double  flops_zgerqf(double m, double n)
    { return 6.*fmuls_gerqf(m, n) + 2.*fadds_gerqf(m, n); }

static double  flops_cgerqf(double m, double n)
    { return 6.*fmuls_gerqf(m, n) + 2.*fadds_gerqf(m, n); }

static double  flops_dgerqf(double m, double n)
    { return    fmuls_gerqf(m, n) +    fadds_gerqf(m, n); }

static double  flops_sgerqf(double m, double n)
    { return    fmuls_gerqf(m, n) +    fadds_gerqf(m, n); }

//------------------------------------------------------------ gelqf
static double fmuls_gelqf(double m, double n)
    { return  fmuls_gerqf(m, n); }

static double fadds_gelqf(double m, double n)
    { return  fadds_gerqf(m, n); }

static double  flops_zgelqf(double m, double n)
    { return 6.*fmuls_gelqf(m, n) + 2.*fadds_gelqf(m, n); }

static double  flops_cgelqf(double m, double n)
    { return 6.*fmuls_gelqf(m, n) + 2.*fadds_gelqf(m, n); }

static double  flops_dgelqf(double m, double n)
    { return    fmuls_gelqf(m, n) +    fadds_gelqf(m, n); }

static double  flops_sgelqf(double m, double n)
    { return    fmuls_gelqf(m, n) +    fadds_gelqf(m, n); }

//------------------------------------------------------------ ungqr
static double fmuls_ungqr(double m, double n, double k)
    { return k*(2.*m*n +  2.*n - 5./3. + k*(2./3.*k - (m + n) - 1.)); }

static double fadds_ungqr(double m, double n, double k)
    { return k*(2.*m*n + n - m + 1./3. + k*(2./3.*k - (m + n))); }

static double  flops_zungqr(double m, double n, double k)
    { return 6.*fmuls_ungqr(m, n, k) + 2.*fadds_ungqr(m, n, k); }

static double  flops_cungqr(double m, double n, double k)
    { return 6.*fmuls_ungqr(m, n, k) + 2.*fadds_ungqr(m, n, k); }

static double  flops_dorgqr(double m, double n, double k)
    { return    fmuls_ungqr(m, n, k) +    fadds_ungqr(m, n, k); }

static double  flops_sorgqr(double m, double n, double k)
    { return    fmuls_ungqr(m, n, k) +    fadds_ungqr(m, n, k); }

//------------------------------------------------------------ ungql
static double fmuls_ungql(double m, double n, double k)
    { return  fmuls_ungqr(m, n, k); }

static double fadds_ungql(double m, double n, double k)
    { return fadds_ungqr(m, n, k); }

static double  flops_zungql(double m, double n, double k)
    { return 6.*fmuls_ungql(m, n, k) + 2.*fadds_ungql(m, n, k); }

static double  flops_cungql(double m, double n, double k)
    { return 6.*fmuls_ungql(m, n, k) + 2.*fadds_ungql(m, n, k); }

static double  flops_dorgql(double m, double n, double k)
    { return    fmuls_ungql(m, n, k) +    fadds_ungql(m, n, k); }

static double  flops_sorgql(double m, double n, double k)
    { return    fmuls_ungql(m, n, k) +    fadds_ungql(m, n, k); }

//------------------------------------------------------------ ungrq
static double fmuls_ungrq(double m, double n, double k)
    { return k*(2.*m*n + m + n - 2./3. + k*(2./3.*k - (m + n) - 1.)); }

static double fadds_ungrq(double m, double n, double k)
    { return k*(2.*m*n + m - n + 1./3. + k*(2./3.*k - (m + n))); }

static double  flops_zungrq(double m, double n, double k)
    { return 6.*fmuls_ungrq(m, n, k) + 2.*fadds_ungrq(m, n, k); }

static double  flops_cungrq(double m, double n, double k)
    { return 6.*fmuls_ungrq(m, n, k) + 2.*fadds_ungrq(m, n, k); }

static double  flops_dorgrq(double m, double n, double k)
    { return    fmuls_ungrq(m, n, k) +    fadds_ungrq(m, n, k); }

static double  flops_sorgrq(double m, double n, double k)
    { return    fmuls_ungrq(m, n, k) +    fadds_ungrq(m, n, k); }

//------------------------------------------------------------ unglq
static double fmuls_unglq(double m, double n, double k)
    { return fmuls_ungrq(m, n, k); }

static double fadds_unglq(double m, double n, double k)
    { return fadds_ungrq(m, n, k); }

static double  flops_zunglq(double m, double n, double k)
    { return 6.*fmuls_unglq(m, n, k) + 2.*fadds_unglq(m, n, k); }

static double  flops_cunglq(double m, double n, double k)
    { return 6.*fmuls_unglq(m, n, k) + 2.*fadds_unglq(m, n, k); }

static double  flops_dorglq(double m, double n, double k)
    { return    fmuls_unglq(m, n, k) +    fadds_unglq(m, n, k); }

static double  flops_sorglq(double m, double n, double k)
    { return    fmuls_unglq(m, n, k) +    fadds_unglq(m, n, k); }

//------------------------------------------------------------ geqrs
static double fmuls_geqrs(double m, double n, double nrhs)
    { return nrhs*(n*(2.*m - 0.5*n + 2.5)); }

static double fadds_geqrs(double m, double n, double nrhs)
    { return nrhs*(n*(2.*m - 0.5*n + 0.5)); }

static double  flops_zgeqrs(double m, double n, double nrhs)
    { return 6.*fmuls_geqrs(m, n, nrhs) + 2.*fadds_geqrs(m, n, nrhs); }

static double  flops_cgeqrs(double m, double n, double nrhs)
    { return 6.*fmuls_geqrs(m, n, nrhs) + 2.*fadds_geqrs(m, n, nrhs); }

static double  flops_dgeqrs(double m, double n, double nrhs)
    { return    fmuls_geqrs(m, n, nrhs) +    fadds_geqrs(m, n, nrhs); }

static double  flops_sgeqrs(double m, double n, double nrhs)
    { return    fmuls_geqrs(m, n, nrhs) +    fadds_geqrs(m, n, nrhs); }

//------------------------------------------------------------ unmqr
static double fmuls_unmqr(PLASMA_enum side, double m, double n, double k)
{
    return (side == PlasmaLeft)
        ? (2.*n*m*k - n*k*k + 2.*n*k)
        : (2.*n*m*k - m*k*k + m*k + n*k - 0.5*k*k + 0.5*k);
}

static double fadds_unmqr(PLASMA_enum side, double m, double n, double k)
{
    return (side == PlasmaLeft)
        ? (2.*n*m*k - n*k*k + n*k)
        : (2.*n*m*k - m*k*k + m*k);
}

static double  flops_zunmqr(PLASMA_enum side, double m, double n, double k)
    { return 6.*fmuls_unmqr(side, m, n, k) + 2.*fadds_unmqr(side, m, n, k); }

static double  flops_cunmqr(PLASMA_enum side, double m, double n, double k)
    { return 6.*fmuls_unmqr(side, m, n, k) + 2.*fadds_unmqr(side, m, n, k); }

static double  flops_dormqr(PLASMA_enum side, double m, double n, double k)
    { return    fmuls_unmqr(side, m, n, k) +    fadds_unmqr(side, m, n, k); }

static double  flops_sormqr(PLASMA_enum side, double m, double n, double k)
    { return    fmuls_unmqr(side, m, n, k) +    fadds_unmqr(side, m, n, k); }

//------------------------------------------------------------ unmql
static double fmuls_unmql(PLASMA_enum side, double m, double n, double k)
    { return fmuls_unmqr(side, m, n, k); }

static double fadds_unmql(PLASMA_enum side, double m, double n, double k)
    { return fadds_unmqr(side, m, n, k); }

static double  flops_zunmql(PLASMA_enum side, double m, double n, double k)
    { return 6.*fmuls_unmql(side, m, n, k) + 2.*fadds_unmql(side, m, n, k); }

static double  flops_cunmql(PLASMA_enum side, double m, double n, double k)
    { return 6.*fmuls_unmql(side, m, n, k) + 2.*fadds_unmql(side, m, n, k); }

static double  flops_dormql(PLASMA_enum side, double m, double n, double k)
    { return    fmuls_unmql(side, m, n, k) +    fadds_unmql(side, m, n, k); }

static double  flops_sormql(PLASMA_enum side, double m, double n, double k)
    { return    fmuls_unmql(side, m, n, k) +    fadds_unmql(side, m, n, k); }

//------------------------------------------------------------ unmrq
static double fmuls_unmrq(PLASMA_enum side, double m, double n, double k)
    { return fmuls_unmqr(side, m, n, k); }

static double fadds_unmrq(PLASMA_enum side, double m, double n, double k)
    { return fadds_unmqr(side, m, n, k); }

static double  flops_zunmrq(PLASMA_enum side, double m, double n, double k)
    { return 6.*fmuls_unmrq(side, m, n, k) + 2.*fadds_unmrq(side, m, n, k); }

static double  flops_cunmrq(PLASMA_enum side, double m, double n, double k)
    { return 6.*fmuls_unmrq(side, m, n, k) + 2.*fadds_unmrq(side, m, n, k); }

static double  flops_dormrq(PLASMA_enum side, double m, double n, double k)
    { return    fmuls_unmrq(side, m, n, k) +    fadds_unmrq(side, m, n, k); }

static double  flops_sormrq(PLASMA_enum side, double m, double n, double k)
    { return    fmuls_unmrq(side, m, n, k) +    fadds_unmrq(side, m, n, k); }

//------------------------------------------------------------ unmlq
static double fmuls_unmlq(PLASMA_enum side, double m, double n, double k)
    { return fmuls_unmqr(side, m, n, k); }

static double fadds_unmlq(PLASMA_enum side, double m, double n, double k)
    { return fadds_unmqr(side, m, n, k); }

static double  flops_zunmlq(PLASMA_enum side, double m, double n, double k)
    { return 6.*fmuls_unmlq(side, m, n, k) + 2.*fadds_unmlq(side, m, n, k); }

static double  flops_cunmlq(PLASMA_enum side, double m, double n, double k)
    { return 6.*fmuls_unmlq(side, m, n, k) + 2.*fadds_unmlq(side, m, n, k); }

static double  flops_dormlq(PLASMA_enum side, double m, double n, double k)
    { return    fmuls_unmlq(side, m, n, k) +    fadds_unmlq(side, m, n, k); }

static double  flops_sormlq(PLASMA_enum side, double m, double n, double k)
    { return    fmuls_unmlq(side, m, n, k) +    fadds_unmlq(side, m, n, k); }

//------------------------------------------------------------ trtri
static double fmuls_trtri(double n)
    { return n*(n*(1./6.*n + 0.5) + 1./3.); }

static double fadds_trtri(double n)
    { return n*(n*(1./6.*n - 0.5) + 1./3.); }

static double  flops_ztrtri(double n)
    { return 6.*fmuls_trtri(n) + 2.*fadds_trtri(n); }

static double  flops_ctrtri(double n)
    { return 6.*fmuls_trtri(n) + 2.*fadds_trtri(n); }

static double  flops_dtrtri(double n)
    { return    fmuls_trtri(n) +    fadds_trtri(n); }

static double  flops_strtri(double n)
    { return    fmuls_trtri(n) +    fadds_trtri(n); }

//------------------------------------------------------------ gehrd
static double fmuls_gehrd(double n)
    { return n*(n*(5./3. *n + 0.5) - 7./6.) - 13.; }

static double fadds_gehrd(double n)
    { return n*(n*(5./3. *n - 1.0) - 2./3.) -  8.; }

static double  flops_zgehrd(double n)
    { return 6.*fmuls_gehrd(n) + 2.*fadds_gehrd(n); }

static double  flops_cgehrd(double n)
    { return 6.*fmuls_gehrd(n) + 2.*fadds_gehrd(n); }

static double  flops_dgehrd(double n)
    { return    fmuls_gehrd(n) +    fadds_gehrd(n); }

static double  flops_sgehrd(double n)
    { return    fmuls_gehrd(n) +    fadds_gehrd(n); }

//------------------------------------------------------------ sytrd
static double fmuls_sytrd(double n)
    { return n*(n*(2./3.*n + 2.5) - 1./6.) - 15.; }

static double fadds_sytrd(double n)
    { return n*(n*(2./3.*n + 1.0) - 8./3.) -  4.; }

static double fmuls_hetrd(double n)
    { return fmuls_sytrd(n); }

static double fadds_hetrd(double n)
    { return fadds_sytrd(n); }

static double  flops_zhetrd(double n)
    { return 6.*fmuls_hetrd(n) + 2.*fadds_hetrd(n); }

static double  flops_chetrd(double n)
    { return 6.*fmuls_hetrd(n) + 2.*fadds_hetrd(n); }

static double  flops_dsytrd(double n)
    { return    fmuls_sytrd(n) +    fadds_sytrd(n); }

static double  flops_ssytrd(double n)
    { return    fmuls_sytrd(n) +    fadds_sytrd(n); }

//------------------------------------------------------------ gebrd
static double fmuls_gebrd(double m, double n)
{
    return (m >= n)
        ? (n*(n*(2.*m - 2./3.*n + 2.) + 20./3.))
        : (m*(m*(2.*n - 2./3.*m + 2.) + 20./3.));
}

static double fadds_gebrd(double m, double n)
{
    return (m >= n)
        ? (n*(n*(2.*m - 2./3.*n + 1.) - m +  5./3.))
        : (m*(m*(2.*n - 2./3.*m + 1.) - n +  5./3.));
}

static double  flops_zgebrd(double m, double n)
    { return 6.*fmuls_gebrd(m, n) + 2.*fadds_gebrd(m, n); }

static double  flops_cgebrd(double m, double n)
    { return 6.*fmuls_gebrd(m, n) + 2.*fadds_gebrd(m, n); }

static double  flops_dgebrd(double m, double n)
    { return    fmuls_gebrd(m, n) +    fadds_gebrd(m, n); }

static double  flops_sgebrd(double m, double n)
    { return    fmuls_gebrd(m, n) +    fadds_gebrd(m, n); }

//------------------------------------------------------------ larfg
static double fmuls_larfg(double n)
    { return 2*n; }

static double fadds_larfg(double n)
    { return   n; }

static double  flops_zlarfg(double n)
    { return 6.*fmuls_larfg(n) + 2.*fadds_larfg(n); }

static double  flops_clarfg(double n)
    { return 6.*fmuls_larfg(n) + 2.*fadds_larfg(n); }

static double  flops_dlarfg(double n)
    { return    fmuls_larfg(n) +    fadds_larfg(n); }

static double  flops_slarfg(double n)
    { return    fmuls_larfg(n) +    fadds_larfg(n); }

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_PLASMA_FLOPS_H
