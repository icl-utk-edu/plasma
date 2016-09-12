#!/bin/bash
#
# @author  Maksims Abalenkovs
# @email   m.abalenkovs@manchester.ac.uk
# @date    Aug 31, 2016
# @version 0.1
#
# Implements an exhaustive test of PLASMA ZTRMM routine.

source ~/.profile

m=1000
n=750
alpha=2.7182818284

for side in "l" "r"; do
  for uplo in "u" "l"; do
    for transa in "n" "t" "c"; do
      for diag in "n" "u"; do
        ./test ztrmm --iter=10 --outer=y --side=$side --uplo=$uplo --transa=$transa --diag=$diag --m=$m --n=$n --alpha=$alpha
      done
    done
  done
done
