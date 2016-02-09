
SRC = $(wildcard *.c)
OBJ = $(SRC:.c=.o)
EXE	= test

CFLAGS = -std=c99 -fopenmp -O3 -DPLASMA_WITH_MKL -w

MKL = /opt/intel/mkl

INC = -I../include -I$(MKL)/include

LIB = -L$(MKL)/lib

LIBS = -lmkl_intel_lp64 -lmkl_core -lmkl_sequential

all:
	gcc $(CFLAGS) $(INC) $(SRC) -o $(EXE) $(LIB) $(LIBS)

clean:
	rm -f $(EXE) $(OBJ)
