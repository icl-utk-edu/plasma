#!/bin/bash -x

maker=$1
device=$2

# CMake build directory. cd is in setup_env.sh.
rm -rf build
mkdir -p build

mydir=$(dirname $0)
source ${mydir}/setup_env.sh

print "======================================== Environment"
# Show environment variables, excluding functions.
(set -o posix; set)

print "======================================== Modules"
quiet module list -l

print "======================================== Setup build"
# Note: set all env variables in setup_env.sh,
# else build.sh and test.sh won't see them.

rm -rf ${top}/install

cmake -DCMAKE_INSTALL_PREFIX=${top}/install .. || exit 12

print "======================================== Finished configure"
exit 0
