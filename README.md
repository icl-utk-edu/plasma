
~~~~
 _ \ |      \    __|  \  |   \
 __/ |     _ \ \__ \ |\/ |  _ \
_|  ____|_/  _\____/_|  _|_/  _\
~~~~

**Parallel Linear Algebra Software for Multicore Architectures**

**University of Tennessee,
University of Colorado Denver,
University of California, Berkeley**

PLASMA is a software package for solving problems in dense linear algebra
using multicore processors and Xeon Phi coprocessors.
PLASMA provides implementations of state-of-the-art algorithms
using cutting-edge task scheduling techniques.
PLASMA currently offers a collection of routines
for solving linear systems of equations, least squares problems,
eigenvalue problems, and singular value problems.

PLASMA is in the process of porting form [QUARK](http://icl.cs.utk.edu/quark/)
to [OpenMP](http://openmp.org/wp/).
At the same time, it is moving from its ICL SVN repository
to this Bitbucket Mercurial repository.
The content of this repository reflects the progress of the transition.
Before the transition is complete, the old releases of PLASMA are available at
http://icl.cs.utk.edu/plasma/.

Feel free to use the following publications to reference PLASMA:

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

* Emmanuel Agullo, Jim Demmel, Jack Dongarra, Bilel Hadri, Jakub Kurzak, Julien Langou,
  Hatem Ltaief, Piotr Luszczek and Stanimire Tomov,
  **Numerical linear algebra on emerging architectures: The PLASMA and MAGMA projects**,
  *Journal of Physics: Conference Series*,
  [Volume 180, Number 1](http://dx.doi.org/10.1088/1742-6596/180/1/012037),
  IOP Publishing, 2009.

* Alfredo Buttari, Julien Langou, Jakub Kurzak, Jack Dongarra,
  **A class of parallel tiled linear algebra algorithms for multicore architectures**,
  *Parallel Computing*,
  [Volume 35, Issue 1, January 2009, Pages 38–53](http://dx.doi.org/10.1016/j.parco.2008.10.002).

~~~~
  @article{donfack2015survey,
    title={A survey of recent developments in parallel implementations of Gaussian elimination},
    author={Donfack, Simplice and Dongarra, Jack and Faverge, Mathieu and Gates, Mark and Kurzak, Jakub and Luszczek, Piotr and Yamazaki, Ichitaro},
    journal={Concurrency and Computation: Practice and Experience},
    volume={27},
    number={5},
    pages={1292--1309},
    year={2015},
    publisher={Wiley Online Library}
  }

@inproceedings{haidar2013improved,
  title={An improved parallel singular value algorithm and its implementation for multicore hardware},
  author={Haidar, Azzam and Kurzak, Jakub and Luszczek, Piotr},
  booktitle={Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis},
  pages={90},
  year={2013},
  organization={ACM}
}

@article{kurzak2010scheduling,
  title={Scheduling dense linear algebra operations on multicore processors},
  author={Kurzak, Jakub and Ltaief, Hatem and Dongarra, Jack and Badia, Rosa M},
  journal={Concurrency and Computation: Practice and Experience},
  volume={22},
  number={1},
  pages={15--44},
  year={2010},
  publisher={Wiley Online Library}
}

@inproceedings{agullo2009numerical,
  title={Numerical linear algebra on emerging architectures: The PLASMA and MAGMA projects},
  author={Agullo, Emmanuel and Demmel, Jim and Dongarra, Jack and Hadri, Bilel and Kurzak, Jakub and Langou, Julien and Ltaief, Hatem and Luszczek, Piotr and Tomov, Stanimire},
  booktitle={Journal of Physics: Conference Series},
  volume={180},
  number={1},
  pages={012037},
  year={2009},
  organization={IOP Publishing}
}

@article{buttari2009class,
  title={A class of parallel tiled linear algebra algorithms for multicore architectures},
  author={Buttari, Alfredo and Langou, Julien and Kurzak, Jakub and Dongarra, Jack},
  journal={Parallel Computing},
  volume={35},
  number={1},
  pages={38--53},
  year={2009},
  publisher={Elsevier}
}
~~~~
