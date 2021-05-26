#!/bin/sh -u

exit_stat=0

rout_list=$(mktemp)
grep '{ "[cdisz]' test.c | cut -d '"' -f 2 | sort -u > "${rout_list}"

while read -r rout
do
  echo "test ${rout}:"
  ./test "${rout}" || exit_stat=1
done < "${rout_list}"

if test ${exit_stat} -ne 0
then
  echo "FAILED" >&2
else
  echo "ok"
fi

rm -f "${rout_list}"

exit ${exit_stat}
