/**
 *
 * @file flops.h
 *
 *  PLASMA test routines.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver.
 *
 * @version 3.0.0
 * @author Jakub Kurzak
 * @date 2016-01-01
 *
 **/
#ifndef FLOPS_H
#define FLOPS_H

//==============================================================================
// FLOPS counts
//==============================================================================
static double flops_zgemm(int m, int n, int k) { return 8.0*m*n*k; }
static double flops_cgemm(int m, int n, int k) { return 8.0*m*n*k; }
static double flops_dgemm(int m, int n, int k) { return 2.0*m*n*k; }
static double flops_sgemm(int m, int n, int k) { return 2.0*m*n*k; }

#endif // FLOPS_H
