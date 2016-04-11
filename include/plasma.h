/**
 *
 * @file plasma.h
 *
 *  PLASMA header.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver.
 *
 * @version 3.0.0
 * @author Jakub Kurzak
 * @date 2016-01-01
 *
 **/
#ifndef PLASMA_H
#define PLASMA_H

#ifdef PLASMA_WITH_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

#include "async.h"
#include "descriptor.h"
#include "context.h"

#include "plasma_s.h"
#include "plasma_d.h"
#include "plasma_c.h"
#include "plasma_z.h"

#endif // PLASMA_H
