
SRC = $(wildcard *.c)
OBJ = $(SRC:.c=.o)
EXE	= test

all:
	gcc -std=c99 -O3 $(SRC) -o $(EXE)

clean:
	rm -f $(EXE) $(OBJ)
