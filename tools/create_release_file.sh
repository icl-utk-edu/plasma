#! /bin/bash

MJR=`grep -r PLASMA_VERSION_MAJOR include | awk '{print $NF;}'`
MNR=`grep -r PLASMA_VERSION_MINOR include | awk '{print $NF;}'`
PTC=`grep -r PLASMA_VERSION_PATCH include | awk '{print $NF;}'`

# this can come from include/plasma_types.h
VERSION=${MJR}.${MNR}.$PTC

DIR=plasma-${VERSION}

if test ! -e $DIR ; then
    ln -s . $DIR
fi

echo Preparing $DIR ...

find -H ${DIR} -maxdepth 1 -type f | \
    xargs echo ${DIR}/*/*.[ch] ${DIR}/*/*.py ${DIR}/*/*.hin ${DIR}/*/*.cmake ${DIR}/*/*.f90 ${DIR}/*/*.lua ${DIR}/*/doxygen* |  \
    xargs tar --exclude=.hgtags --exclude=plasma_config.h --exclude=Makefile.\*.gen --owner=root --group=root --mtime=1970-01-01 -cshof ${DIR}.tar
gzip --best --rsyncable --verbose ${DIR}.tar

rm -r $DIR
