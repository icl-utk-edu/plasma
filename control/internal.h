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
static inline void plasma_warning(const char *func_name, char* msg_text)
{
    fprintf(stderr, "PLASMA WARNING: %s(): %s\n", func_name, msg_text);
}

/******************************************************************************/
static inline void plasma_error(const char *func_name, char* msg_text)
{
    fprintf(stderr, "PLASMA ERROR: %s(): %s\n", func_name, msg_text);
}

/******************************************************************************/
static inline void plasma_fatal_error(const char *func_name, char* msg_text)
{
    fprintf(stderr, "PLASMA FATAL ERROR: %s(): %s\n", func_name, msg_text);
    exit(0);
}

#endif // INTERNAL_H
