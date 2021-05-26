#!/bin/sh -u

exit_stat=0

rout_list=$(mktemp)
find . -name 'test_*.f90' | sed -e 's%\.f90$%%g' | sort -u > "${rout_list}"

test_output=$(mktemp)

while read -r rout
do
  echo test "$(echo "${rout}" | cut -d "_" -f 2)":
  "${rout}" | tee "${test_output}"
  if grep WRONG "${test_output}"
  then
    exit_stat=1
  fi
done < "${rout_list}"

if test ${exit_stat} -ne 0
then
  echo "FAILED" >&2
else
  echo "ok"
fi

rm -f "${rout_list}" "${test_output}"

exit ${exit_stat}
