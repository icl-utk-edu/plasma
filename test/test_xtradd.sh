#!/bin/bash
#
# @author  Maksims Abalenkovs
# @email   m.abalenkovs@manchester.ac.uk
# @date    Sep 9, 2016
# @version 0.1
#
# Implements an exhaustive test of PLASMA ZTRADD routine.

source ~/.profile

x="c"
uplo="l"
transa="n"
m=5
n=5
nb=5
routine="tradd"
alpha=2.7182818284
beta=3.1415926535

# for x in "z" "c" "d" "s"; do
  # for uplo in "f" "u" "l"; do
    # for transa in "n" "t" "c"; do
      # for m in "750" "1000"; do
        # for n in "750" "1000"; do
          # for nb in "125" "250" "500"; do
            ./test $x$routine --iter=1 --outer=y --uplo=$uplo --transa=$transa \
	                      --m=$m --n=$n --alpha=$alpha --beta=$beta --nb=$nb
          # done
        # done
      # done
    # done
  # done
# done
