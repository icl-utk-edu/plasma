#!/bin/bash
#
# Sourced from configure.sh, build.sh, test.sh
# to load modules and setup environment.

#-------------------------------------------------------------------------------
# Functions

# Suppress echo (-x) output of commands executed with `quiet`.
# Useful for sourcing files, loading modules, spack, etc.
# set +x, set -x are not echo'd.
quiet() {
    { set +x; } 2> /dev/null;
    $@;
    set -x
}

# `print` is like `echo`, but suppresses output of the command itself.
# https://superuser.com/a/1141026
echo_and_restore() {
    builtin echo "$*"
    date
    case "${save_flags}" in
        (*x*)  set -x
    esac
}
alias print='{ save_flags="$-"; set +x; } 2> /dev/null; echo_and_restore'


#-------------------------------------------------------------------------------
quiet source /etc/profile

hostname && pwd
export top=$(pwd)

shopt -s expand_aliases

quiet module load intel-oneapi-mkl
print "MKLROOT=${MKLROOT}"

quiet module load python
quiet which python
quiet which python3
python  --version
python3 --version

quiet module load pkgconf
quiet which pkg-config

#----------------------------------------------------------------- Compiler
print "======================================== Load GNU compiler"
quiet module load gcc@11.3

print "---------------------------------------- Verify compiler"
print "CXX = $CXX"
print "CC  = $CC"
print "FC  = $FC"
${CXX} --version
${CC}  --version
${FC}  --version

#----------------------------------------------------------------- CMake
print "======================================== Load cmake"
quiet module load cmake
quiet which cmake
cmake --version
cd build
