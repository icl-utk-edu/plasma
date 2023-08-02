#! /usr/bin/env python3


import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import time


def log(*args):
    print(*args)


def trashit(filename):
    trash = "trash"
    if not os.path.exists(trash):
        os.mkdir(trash)
    if os.path.exists(filename):
        os.rename(filename, trash + "/" + filename + "_" + str(int(time.time())))


def main(argv):
    package_name = os.path.split(os.path.abspath(os.path.curdir))[1]

    for line in open("CMakeLists.txt"):
        package_name + "version"
    filename = argv[1]
    version = filename.replace(package_name, "").replace("-", "").replace(".tar.gz", "")
    log("Processing", package_name, "package version", version)

    if 0 and os.path.exists(filename): # FIXME: add proper creation of new file
        if "--remove" in argv:
            trashit(filename)
    else:
        if 0:subprocess.call(["tar", "--exclude=.git", "--exclude=tools", "-chzf", filename, "xsdk-examples-0.2.0"])
        if 0:subprocess.call(["sh", "tools/create_release_file.sh"])

    dgst = hashlib.sha256(open(filename,"rb").read()).hexdigest()
    dstdir = pathlib.Path(os.environ["SPACK_ROOT"]) / pathlib.Path("var/spack/cache/_source-cache/archive/"+dgst[:2])
    log("New SHA256", dgst)
    if not dstdir.exists():
        log("Created ", dstdir)
        os.makedirs(dstdir, exist_ok=True)
    dstfn = dstdir / pathlib.Path(dgst+".tar.gz")
    if not os.path.exists(dstfn):
        log("Copying ", filename)
        shutil.copy(filename, dstfn)

    package = pathlib.Path(os.environ["SPACK_ROOT"]) / pathlib.Path("var/spack/repos/builtin/packages/{}/package.py".format(package_name))

    op = ""
    pkgdt = open(package).read()
    idx = pkgdt.find("version('{}'".format(version))
    if idx > 0:
        eqidx = pkgdt.find("=", idx)     # =
        qtidx = pkgdt.find("'", eqidx)   # '
        q2idx = pkgdt.find("'", qtidx+1) # '
        sha256=pkgdt[qtidx+1:q2idx]
        if sha256 == dgst:
            log("{} already available".format(sha256))
        else:
            with open(package, "w") as fd:
                fd.write(pkgdt.replace(sha256, dgst))
            op = "Replacing"

    elif -1 == idx:
        # find first version spec
        idx = pkgdt.find("version(")
        indent = 4 * " "
        if idx > 0:
            with open(package, "w") as fd:
                fd.write(pkgdt[:idx] + "version('{}', sha256='{}')\n".format(version, dgst) + indent + pkgdt[idx:])
            op = "Adding"
            sha256 = dgst

        else:
            raise RuntimeError("Didn't find spot for version {}".format(version))

    if op:
        log("{} SHA256={} in Spack's {}".format(op, sha256, package))

    if "--remove" in argv:
        trashit(filename)

    return 0


if "__main__" == __name__:
    sys.exit(main(sys.argv))
