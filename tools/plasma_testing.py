#! /usr/bin/env python

###############################################################################
# This testing suite is derived from the plasma_testing.py script of the PLASMA
# project for the purpose of INTERTWinE project.
# Its main goal is to test the correctness of developments of the runtimes from
# the point of view of functions of solving systems of equations with dense
# matrices.
# The benchmark suite comprises:
# 1. DPOSV routine - solving problems with symmetric positive definite matrices.
#    (based on Cholesky decomposition of the matrix)
# 2. DGELS routine - solving problems with general matrices, which can be square
#    or rectangular, with more rows than columns or vice-versa. Both these cases
#    are included. (based on QR decomposition of the matrix)
# 3. DGESV routine - solving problems with general square matrices.
#    (based on LU factorization of the matrix)
# Example:
#     ./plasma_testing_intertwine.py
# The script was tested under Linux/UNIX environment.
###############################################################################

from subprocess import Popen, STDOUT, PIPE, TimeoutExpired
import os, sys, math
import getopt

Plasma_Tester = "plasmatest"
#Plasma_Tester smaset path to PLASMA testing directory (part of the distribution, not the installed one)
plasma_dir = ".."
path_to_plasma_tests = os.path.realpath(plasma_dir) + "/test"
if os.path.exists(Plasma_Tester) and os.access(Plasma_Tester, os.X_OK):
    path_to_plasma_tests = os.path.abspath(os.curdir)

#print path_to_plasma_tests

# Add current directory to the path for subshells of this shell
# Allows the popen to find local files in both windows and unixes
os.environ["PATH"] = os.environ["PATH"]+":"+path_to_plasma_tests

os.environ["PLASMA_TUNING_FILENAME"] = plasma_dir + "/tuning/default.lua"

# Define a function to open the executable (different filenames on unix and Windows)
def local_popen(f, cmdline, timeout=10*60):
   p=Popen(cmdline, shell=True, stdout=PIPE, stderr=STDOUT)

   try:
     r = p.wait(timeout=timeout)
   except TimeoutExpired as exc:
     print("Timed out after {} seconds for command {}".format(exc.timeout, repr(exc.cmd)))
     return 127

   pipe=p.stdout

   if r != 0:
      print("---- TESTING " + cmdline.split()[3] + "... FAILED(" + str(p.returncode) +") !")
      err = p.returncode
      for line in pipe.readlines():
         f.write(str(line))
   else:
      found=0
      err = "Error parsing output."
      for line in pipe.readlines():
         f.write(str(line))
         if b"pass" in line :
            found = 1
            #print line,
            err = 0
      if found == 0:
         print(cmdline.split()[0] + " " + cmdline.split()[3] + ": FAILED(Unexpected error)")
         f.flush();
         err = "Unexpected error"

   f.flush();
   return err


# plot the results right to the stdout
f = sys.stdout

print(" ")
print("----- Testing the subset of double precision PLASMA Routines related to INTERTWinE project -----")
print(" ")

# only double precision routines are part of the benchmark
dtypes = (
("s", "d", "c", "z"),
("Single", "Double", "Complex", "Double Complex"),
("sor", "dor", "cun", "zun"),
("t", "t", "c", "c"),
)

errors =[];

for dtype in range(4):
   letter = dtypes[0][dtype]
   name = dtypes[1][dtype]
   namemqrlq = dtypes[2][dtype]
   transpose = dtypes[3][dtype]


   print(" ")
   print("------------------------- %s ------------------------" % name)
   print(" ")

   # play the strings game
   #binary = path_to_plasma_tests + "/" + "%stesting" % letter
   binary = path_to_plasma_tests + "/" + Plasma_Tester

   geqrf  = "%sgeqrf"  % letter
   potrf  = "%spotrf"  % letter
   getrf  = "%sgetrf"  % letter
   gels   = "%sgels"   % letter

   # check the binary exists
   if (os.path.exists(binary) and os.access(binary,os.X_OK)):
      # geqrf
      print("------------------------- %s ------------------------" % geqrf)
      test = local_popen(f, binary + " " + geqrf + " --outer=y --dim=1000"); errors.append(test);

      # potrf
      print("------------------------- %s ------------------------" % potrf)
      test = local_popen(f, binary + " " + potrf + " --outer=y --dim=1000"); errors.append(test);

      # getrf
      print("------------------------- %s ------------------------" % getrf)
      test = local_popen(f, binary + " " + getrf + "  --outer=y --dim=2000x1000"); errors.append(test);

      # gels
      print("------------------------- %s ------------------------" % gels)
      for h in ("f", "t"):
          test = local_popen(f, binary + " " + gels  + " --outer=y --dim=2000x1000 --trans=n," + transpose + " --hmode=" + h); errors.append(test);

      sys.stdout.flush()
   else:
      print("The file for testing:", binary, "does not exist or does not have execute rights.")
      print("Have you build PLASMA by make in the PLASMA directory:", plasma_dir, "?")

print("----- Testing the subset of PLASMA Routines -----")
if (any(errors)):
   print("At least one test FAILED.")
   sys.exit(1)
else:
   if (len(errors) == 0):
      print("Tests FAILED, no test were actually run for some reason.")
      sys.exit(2)
   else:
      print("All tests were SUCCESSFUL.")
      sys.exit(0)

sys.exit(3)
