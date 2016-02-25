/**
 *
 * @file context.h
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
#ifndef CONTEXT_H
#define CONTEXT_H

/***************************************************************************//**
 *  PLASMA context.
 **/
typedef struct {
    int nb;                  ///< tile size
    PLASMA_enum translation; ///< in-place or out-of-place layout translation
} plasma_context_t;

#endif CONTEXT_H
