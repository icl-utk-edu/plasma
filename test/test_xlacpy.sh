#!/bin/bash
#
# @author  Maksims Abalenkovs
# @email   m.abalenkovs@manchester.ac.uk
# @date    Oct 17, 2016
# @version 0.3
#
# Implements an exhaustive test of PLASMA xLACPY routine.

# source ~/.profile

routine="lacpy"

uplo="g,u,l"
m=500,750,1000
n=500,750,1000
nb=64,128,256

for x in "z" "c" "d" "s"; do
  ./test $x$routine --iter=1 --outer=y --uplo=$uplo --m=$m --n=$n --nb=$nb
done
