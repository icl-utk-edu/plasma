.

     _ \ |      \    __|  \  |   \
     __/ |     _ \ \__ \ |\/ |  _ \
    _|  ____|_/  _\____/_|  _|_/  _\

* * *

**Parallel Linear Algebra Software for Multicore Architectures**

**University of Tennessee (US)**

**University of Manchester (UK)**

* * *

[Download PLASMA Software](https://github.com/icl-utk-edu/plasma/tags)

* * *

[TOC]

* * *

About
=====

PLASMA is a software package for solving problems in dense linear algebra
using OpenMP.
PLASMA provides implementations of state-of-the-art algorithms
using cutting-edge task scheduling techniques.
PLASMA currently offers a collection of routines
for solving linear systems of equations, least squares problems,
eigenvalue problems, and singular value problems.

PLASMA was ported from [QUARK](http://icl.cs.utk.edu/quark/)
to [OpenMP](http://www.openmp.org/) using the modern features of the latter.
At the same time, PLASMA was moved from its ICL SVN repository
to this Bitbucket Mercurial repository (a move to Git is forthcoming).
The content of this repository reflects the progress of the transition.
Before the transition is complete, the last release of the old PLASMA
is available here:
https://bitbucket.org/icl/plasma/downloads/plasma-2.8.tar.gz

More information about the old PLASMA based on QUARK is included below.

Installation
============

Installing PLASMA on a Linux-like systems requires a compiler (PLASMA is
written in C with OpenMP and also includes Fortran interface) and several
linear algebra libraries: BLAS and CBLAS (implementation exposing Fortran
bindings and its C interface) with accompanying LAPACK and LAPACKE
(implementation exposing Fortran bindings and its C interface).

The regularly tested compilers include [GNU
GCC](https://gcc.gnu.org/projects/gomp/),
[LLVM](https://openmp.llvm.org/), Intel, and HPE/Cray.
The compatible linear algebra libraries include Apple Accelerate/VecLib, ATLAS,
HPE/Cray LibSci, IBM ESSL, Intel MKL (and oneMKL),
Netlib
[BLAS](https://www.netlib.org/blas)/CBLAS/[LAPACK](https://www.netlib.org/lapack)/LAPACKE
suite, and OpenBLAS. Both ATLAS and Netlib
implementations provided the functionality required by PLASMA in multiple
library files. OpenBLAS provides all of its functionality in a single
library. This difference is handled automatically during installation.

The main supported method of installing PLASMA is through CMake and mos
package managers, such as Spack, or module systems, such as Tcl Modules
or Lmod, provide enough information to CMake making the installation a
three-step process:

    cmake /path/to/plasma
    cmake --build .
    cmake --install .

PLASMA manages thread-level parallelism internally through OpenMP and
thus the BLAS and LAPACK should not use multiple threads for
parallelism or the resulting performance will suffer. This can be
achieved by building the BLAS and LAPACK with threading disabled or use
specific configuration options such as environment variables, for
example `MKL_NUM_THREADS=1` for MKL.

Documentation
=============

Doxygen-generated PLASMA documentation is available at:
http://icl.bitbucket.io/plasma/

Getting Assistance
==================

To get assistance with PLASMA, join the *PLASMA User* Google group by going to
https://groups.google.com/a/icl.utk.edu/forum/#!forum/plasma-user and clicking
`Apply to join group`.
Then email your questions and comments to `plasma-user@icl.utk.edu`.

Citing
======

Feel free to use the following publications to reference PLASMA:

* Asim YarKhan, Jakub Kurzak, Piotr Luszczek, Jack Dongarra,
  **Porting the PLASMA Numerical Library to the OpenMP Standard**,
  *International Journal of Parallel Programming*,
  [First Online: 14 June 2016](http://dx.doi.org/10.1007/s10766-016-0441-6).

* Simplice Donfack, Jack Dongarra, Mathieu Faverge, Mark Gates,
  Jakub Kurzak, Piotr Luszczek, Ichitaro Yamazaki,
  **A survey of recent developments in parallel implementations
  of Gaussian elimination**,
  *Concurrency and Computation: Practice and Experience*,
  [Volume 27, Issue 5, April 2015, Pages 1292–1309](http://dx.doi.org/10.1002/cpe.3306).

* Azzam Haidar, Jakub Kurzak, Piotr Luszczek,
  **An improved parallel singular value algorithm and its implementation
  for multicore hardware**,
  *Proceedings of the International Conference on High Performance Computing,
  Networking, Storage and Analysis*
  [Article No. 90](http://dx.doi.org/10.1145/2503210.2503292), ACM, 2013.

* Jakub Kurzak, Hatem Ltaief, Jack Dongarra, Rosa M. Badia,
  **Scheduling dense linear algebra operations on multicore processors**,
  *Concurrency and Computation: Practice and Experience*,
  [Volume 22, Issue 1, January 2010, Pages 15–44](http://dx.doi.org/10.1002/cpe.1467).

* Alfredo Buttari, Julien Langou, Jakub Kurzak, Jack Dongarra,
  **A class of parallel tiled linear algebra algorithms for multicore architectures**,
  *Parallel Computing*,
  [Volume 35, Issue 1, January 2009, Pages 38–53](http://dx.doi.org/10.1016/j.parco.2008.10.002).

Funding
=======

Primary funding for PLASMA was provided by NSF grants:

* [CPA-ACR-T: PLASMA: Parallel Linear Algebra Software for Multiprocessor Architectures](http://www.nsf.gov/awardsearch/showAward?AWD_ID=0811642),
* [Collaborative CPA-ACR-T: PLASMA: Parallel Linear Algebra Software for Multiprocessor Architectures.](http://www.nsf.gov/awardsearch/showAward?AWD_ID=0811520)

Work on PLASMA was also partially funded by NSF grants:

* [SI2-SSI: Collaborative Research: Sustained Innovation for Linear Algebra Software (SILAS)](http://www.nsf.gov/awardsearch/showAward?AWD_ID=1339822),
* [SHF: Small: Empirical Autotuning of Parallel Computation for Scalable Hybrid Systems](http://nsf.gov/awardsearch/showAward?AWD_ID=1527706) (a.k.a. DARE).

Currently, PLASMA is being developed in collaboration with European Commission funded [Horizon 2020](https://ec.europa.eu/programmes/horizon2020/) projects:

* [NLAFET: Parallel Numerical Linear Algebra for Future Extreme Scale Systems](http://www.nlafet.eu), Grant Agreement no. 671633,
* [INTERTWinE: Programming Model INTERoperability ToWards Exascale](http://www.intertwine-project.eu), Grant Agreement no. 671602,

and an [EPSRC](https://www.epsrc.ac.uk/) funded project

* [SERT: Scale-free, Energy-aware, Resilient and Transparent Adaptation of CSE Applications to Mega-core Systems](http://gow.epsrc.ac.uk/NGBOViewGrant.aspx?GrantRef=EP/M01147X/1), EPSRC Reference: EP/M01147X/1.

Additional funding was provided by the following companies:

* Intel Corporation,
* Advanced Micro Devices,
* The MathWorks,
* Fujitsu.

People
======

The following people contributed to the development of PLASMA:

* Maksims Abalenkovs
* Emmanuel Agullo
* Wesley Alvaro
* Dulceneia Becker
* Alfredo Buttari
* Jack Dongarra
* Joseph Dorris
* Mathieu Faverge
* Mark Gates
* Fred Gustavson
* Bilel Hadri
* Azzam Haidar
* Blake Haugen
* Vijay Joshi
* Bo Kågström
* Lars Karlsson
* Jakub Kurzak
* Julien Langou
* Julie Langou
* Hatem Ltaief
* Piotr Luszczek
* Daniel Mishler
* Samuel Relton
* Jakub Sistek
* Stanimire Tomov
* Pedro Valero Lara
* Ichitaro Yamazaki
* Asim YarKhan
* Mawussi Zounon

License
=======

    -- Innovative Computing Laboratory
    -- University of Tennessee
    -- (C) Copyright 2008-2017

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the University of Tennessee, Knoxville nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

    This software is provided by the copyright holders and contributors
    ``as is'' and any express or implied warranties, including, but not
    limited to, the implied warranties of merchantability and fitness for
    a particular purpose are disclaimed. In no event shall the copyright
    holders or contributors be liable for any direct, indirect, incidental,
    special, exemplary, or consequential damages (including, but not
    limited to, procurement of substitute goods or services; loss of use,
    data, or profits; or business interruption) however caused and on any
    theory of liability, whether in contract, strict liability, or tort
    (including negligence or otherwise) arising in any way out of the use
    of this software, even if advised of the possibility of such damage.

Old PLASMA version based on QUARK
=================================

The old version PLASMA is still available for reference but is no longer maintained.

* Old overview: https://icl.utk.edu/plasma/overview/
* Old release news: https://icl.utk.edu/plasma/news/
* Old publications: https://icl.utk.edu/plasma/pubs/
* Old links: https://icl.utk.edu/plasma/links/
* Old list of contributors: https://icl.utk.edu/plasma/people/
* Old documentation: https://icl.utk.edu/plasma/custom/index.html?lid=124&slid=232
* Old releases: http://icl.cs.utk.edu/plasma/software/browse.html
