#!/bin/bash -x

maker=$1
device=$2

mydir=$(dirname $0)
source ${mydir}/setup_env.sh

# Instead of exiting on the first failed test (bash -e),
# run all the tests and accumulate failures into $err.
err=0

export OMP_NUM_THREADS=8

print "======================================== Tests"
cd test

args="--quick"

./run_tests.py ${args}
(( err += $? ))

print "======================================== Finished test"
exit ${err}
