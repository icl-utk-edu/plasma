/**
 *
 * @file internal.h
 *
 *  PLASMA control routines.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver.
 *
 * @version 3.0.0
 * @author Jakub Kurzak
 * @date 2016-01-01
 *
 **/
#ifndef INTERNAL_H
#define INTERNAL_H

#include <stdio.h>

/******************************************************************************/
static inline int imax(int a, int b)
{
    if (a > b)
        return a;
    else
        return b;
}

/******************************************************************************/
#define plasma_warning(msg) \
        plasma_warning_func_line_file(__func__, __LINE__, __FILE__, msg)

#define plasma_error(msg) \
        plasma_error_func_line_file(__func__, __LINE__, __FILE__, msg)

#define plasma_fatal_error(msg) \
        plasma_fatal_error_func_line_file(__func__, __LINE__, __FILE__, msg)

/******************************************************************************/
static inline void plasma_warning_func_line_file(
    char *func, int line, char *file, char *msg)
{
    fprintf(stderr,
            "PLASMA WARNING at %d of %s() in %s: %s\n",
            line, func, file, msg);
}

/******************************************************************************/
static inline void plasma_error_func_line_file(
    char *func, int line, char *file, char *msg)
{
    fprintf(stderr,
            "PLASMA ERROR at %d of %s() in %s: %s\n",
            line, func, file, msg);
}

/******************************************************************************/
static inline void plasma_fatal_error_func_line_file(
    char *func, int line, char *file, char *msg)
{
    fprintf(stderr,
            "PLASMA FATAL ERROR at %d of %s() in %s: %s\n",
            line, func, file, msg);
    exit(0);
}

#include "internal_s.h"
#include "internal_d.h"
#include "internal_c.h"
#include "internal_z.h"

#endif // INTERNAL_H
