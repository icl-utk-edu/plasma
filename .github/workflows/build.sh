#!/bin/bash -x

maker=$1
device=$2

mydir=$(dirname $0)
source ${mydir}/setup_env.sh

print "======================================== Build"
make -j8 || exit 10

print "======================================== Install"
make -j8 install || exit 11
ls -R ${top}/install

print "======================================== Verify build"
ldd_result=$(ldd plasmatest) || exit 12
echo "${ldd_result}"

print "======================================== Finished build"
exit 0
