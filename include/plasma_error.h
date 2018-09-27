/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#ifndef PLASMA_ERROR_H
#define PLASMA_ERROR_H

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************/
#define plasma_warning(msg) \
    plasma_warning_func_line_file(__func__, __LINE__, __FILE__, msg)

#define plasma_error(msg) \
    plasma_error_func_line_file(__func__, __LINE__, __FILE__, msg)

#define plasma_error_with_code(msg, code) \
    plasma_error_func_line_file_code(__func__, __LINE__, __FILE__, msg, \
                                         code)

#define plasma_fatal_error(msg) \
    plasma_fatal_error_func_line_file(__func__, __LINE__, __FILE__, msg)

/******************************************************************************/
static inline void plasma_warning_func_line_file(
    char const *func, int line, const char *file, const char *msg)
{
    fprintf(stderr,
            "PLASMA WARNING at %d of %s() in %s: %s\n",
            line, func, file, msg);
}

/******************************************************************************/
static inline void plasma_error_func_line_file(
    char const *func, int line, const char *file, const char *msg)
{
    fprintf(stderr,
            "PLASMA ERROR at %d of %s() in %s: %s\n",
            line, func, file, msg);
}

/******************************************************************************/
static inline void plasma_error_func_line_file_code(
    char const *func, int line, const char *file, const char *msg, int code)
{
    fprintf(stderr,
            "PLASMA ERROR at %d of %s() in %s: %s %d\n",
            line, func, file, msg, code);
}

/******************************************************************************/
static inline void plasma_fatal_error_func_line_file(
    char const *func, int line, const char *file, const char *msg)
{
    fprintf(stderr,
            "PLASMA FATAL ERROR at %d of %s() in %s: %s\n",
            line, func, file, msg);
    exit(EXIT_FAILURE);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // PLASMA_ERROR_H
