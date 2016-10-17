# PLASMA example make.inc, using Intel MKL and gcc
#
# PLASMA is a software package provided by:
# University of Tennessee, US,
# University of Manchester, UK.

# --------------------
# programs

CC        = gcc

ARCH      = ar
ARCHFLAGS = cr
RANLIB    = ranlib


# --------------------
# flags

# Use -fPIC to make shared (.so) and static (.a) libraries;
# can be commented out if making only static libraries.
FPIC      = -fPIC

CFLAGS    = -fopenmp $(FPIC) -O3 -std=c99 -Wall -pedantic -Wshadow -Wno-unused-function
LDFLAGS   = -fopenmp $(FPIC)

# options for MKL
CFLAGS   += -DPLASMA_WITH_MKL \
            -DMKL_Complex16="double _Complex" \
            -DMKL_Complex8="float _Complex"


# --------------------
# libraries
# This assumes $MKLROOT is set in your environment.
# Add these to your .cshrc or .bashrc, adjusting for where MKL is installed:
# in .cshrc:   source /opt/intel/bin/compilervars.csh intel64
# in .bashrc:  source /opt/intel/bin/compilervars.sh  intel64

# With gcc OpenMP (libgomp), use -lmkl_sequential or (-lmkl_gnu_thread   with MKL_NUM_THREADS=1).
# With icc OpenMP (liomp5),  use -lmkl_sequential or (-lmkl_intel_thread with MKL_NUM_THREADS=1).
LIBS      = -L$(MKLROOT)/lib -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lm

INC       = -I$(MKLROOT)/include
