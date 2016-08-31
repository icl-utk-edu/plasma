#!/bin/sh
#
# Finds doxygen groups that are in use,  sorts & puts in file "ingroup"
# Finds doxygen groups that are defined, sorts & puts in file "defgroup"
# Doing
#     diff ingroup defgroup
# provides an easy way to see what groups are used vs. defined.
#
# Usage, from top level plasma directory:
#     ./tools/doxygen_groups.sh
#
# On MacOS, uses opendiff; otherwise uses diff.

egrep -h '@ingroup' */*.{h,c} | \
	perl -pe 's/^ *\*//;  s@^ *///@@;  s/^ +//;  s/\@ingroup/\@group/;' | \
	sort --unique > ingroup

egrep -h '^ *@defgroup' docs/doxygen/groups.dox | \
    egrep -v 'group_|core_blas' | \
    perl -pe 's/^ *\@defgroup +(\w+).*/\@group $1/;' | \
    sort > defgroup

which opendiff > /dev/null
if [ $? == 0 ]; then
    opendiff ingroup defgroup
else
    diff ingroup defgroup
fi
