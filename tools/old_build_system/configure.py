#!/usr/bin/env python
#
# PLASMA is a software package provided by:
# University of Tennessee, US,
# University of Manchester, UK.

from __future__ import print_function

import sys
import re

import config
from   config import Error, red, font_bold, font_normal

print(
font_bold + red +
'''
                               Welcome to PLASMA!
''' + font_normal + '''
This script, ''' + sys.argv[0] + ''', will create a make.inc configuration file,
which you can then edit if needed.

PLASMA requires a C compiler that supports C99 and OpenMP 4.0 with task depend.
This will search for available compilers. Alternatively, set $CC to your C
compiler. Variables can be set in your environment:
    export CC=gcc  # in sh/bash
or on the make or configure command line:
    make CC=gcc
    python ''' + sys.argv[0] + ''' CC=gcc

PLASMA can optionally compile Fortran 2008 interfaces, if a suitable Fortran
compiler is available. You can set $FC to your Fortran compiler.

PLASMA requires BLAS, CBLAS, LAPACK, and LAPACKE libraries. These are often
provided by optimized vendor math libraries such as AMD ACML, Cray LibSci,
IBM ESSL, Intel MKL, or MacOS Accelerate; or open source libraries such as
ATLAS or OpenBLAS. The open source reference version of LAPACK, with CBLAS
and LAPACKE, is available at:
    http://www.netlib.org/lapack/

If the (C)BLAS and LAPACK(E) libraries are not in the compiler's default path,
specify their location using one or more of the variables below, again set
either in the environment or on the configure command line:
    $ACML_DIR, $ATLAS_DIR, $MKLROOT, $OPENBLAS_DIR, $CBLAS_DIR, $LAPACK_DIR
    $MKLROOT is often set in ~/.bash_profile or ~/.cshrc by one of:
        source /path/to/intel/compilers/bin/compilervars.sh  intel64
        source /path/to/intel/compilers/bin/compilervars.csh intel64

Alternatively, specify the necessary flags with:
    $LAPACK_CFLAGS    Include paths, e.g., -I/opt/lapack/include
    $LAPACK_LIBS      Library paths and libraries, e.g.,
                      -L/opt/lapack/lib -llapacke -llapack -lcblas -lblas

[return to continue, q to quit] ''', end='' )
reply = raw_input()
if (re.search( 'q|quit', reply, re.I )):
	exit(1)

try:
	config.init()
	
	config.prog_cc()
	#config.prog_cxx()
	config.prog_fortran( required=False )
	#config.prog_f77( required=False )
	
	config.blas()
	print()
	config.blas_return_float( required=False )
	config.blas_return_complex( required=False )
	
	config.cblas()
	config.cblas_enum()
	config.lapack()
	
	#config.set_verbose()
	config.lapacke()
	print()
	config.lapacke_dlascl( required=False )
	config.lapacke_dlantr( required=False )
	config.lapacke_dlassq( required=False )
	
	#config.output_headers( 'config.h' )
	config.output_files( 'make.inc' )
	config.print_header( '' )
	
except Error, e:
	config.print_error( 'Configuration aborted: ' + str(e) )
	exit(1)
