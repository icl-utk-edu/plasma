!>
!! @file
!!
!!  PLASMA is a software package provided by:
!!  University of Tennessee,  US,
!!  University of Manchester, UK
!!
!! @precisions mixed zc -> ds
!!
!! @brief Tests LU factorization-based iterative refinement routine for
!!  solution of linear system A*X=B, where A is n-by-n matrix, X and B
!!  are n-by-nrhs matrices

      PROGRAM TEST_ZCGESV

      USE, INTRINSIC :: ISO_FORTRAN_ENV
      USE            :: OMP_LIB
      USE            :: PLASMA

      IMPLICIT NONE

      ! Precisions
      integer, parameter :: sp = REAL32
      integer, parameter :: dp = REAL64

      ! Matrices A, X, B
      complex(dp), allocatable :: A(:,:), Aref(:,:), X(:,:), B(:,:)

      ! Matrix dimensions
      integer :: n = 1000, nrhs = 1, lda, ldb, ldx

      ! PLASMA variables
      integer :: nb = 120
      integer :: iter, info

      ! LAPACK variables
      integer :: iSeed(8) = [218, 1910, 3542, 2995, 3607, 27, 3488, 593]

      complex(dp) :: zone = 1.0, zmone = -1.0

      real(dp), allocatable :: work(:)
      real(dp)              :: Anorm, Xnorm, Rnorm, relerr

      ! Pivots
      integer, allocatable :: iPiv(:)

      ! Test variables
      real(dp)          :: tol  = 0.0d+0

      ! Performance variables
      real(dp) :: tstart = 0.0d+0, tstop = 0.0d+0, telapsed = 0.0d+0

      ! External functions
      real(dp), external :: dlamch
      real(dp), external :: zlange


      lda = max(1,n)
      ldb = lda
      ldx = lda

      tol = 50.0d+0 * dlamch('E')

      ! Allocate matrices
      allocate(A(lda,n), Aref(lda,n), X(ldx,nrhs), B(ldb,nrhs), &
               iPiv(lda), work(n))

      !=================================================================
      ! Generate and initialize random matrices A and B
      !=================================================================

      ! Generate random matrices A, B
      call zlarnv(1, iSeed(1:4), lda*n, A)

      ! Save initial matrix A into Aref
      Aref = A

      ! Generate random matrix B
      call zlarnv(1, iSeed(5:8), ldb*nrhs, B)

      ! Initialize solution matrix X
      X = 0.0d+0

      !=================================================================
      ! Initialize PLASMA
      !=================================================================
      call plasma_init(info)

      call plasma_set(PlasmaNb, nb, info)

      !=================================================================
      ! Solve linear system A*X=B
      !=================================================================
      tstart   = omp_get_wtime()

      call plasma_zcgesv(n, nrhs, A, lda, iPiv, B, ldb, X, ldx, &
                         iter, info)

      tstop    = omp_get_wtime()
      telapsed = tstop - tstart

      !=================================================================
      ! Finalize PLASMA
      !=================================================================
      call plasma_finalize(info)

      !=================================================================
      ! Perform linear system A*X=B solution check
      !=================================================================

      ! Calculate infinite norms of matrices Aref and X
      Anorm = zlange('I', n, n,    Aref, lda, work)
      Xnorm = zlange('I', n, nrhs, X,    ldx, work)

      ! Calculate residual R = A*X-B, store result in B
      call zgemm('N', 'N', n, nrhs, n, zone, Aref, lda, X, ldx, &
                  zmone, B, ldb)

      ! Calculate infinite norm of residual matrix R
      Rnorm = zlange('I', n, nrhs, B, ldb, work)

      ! Calculate relative error
      relerr = Rnorm / n / Anorm / Xnorm

      print *, "Time:            ", telapsed
      print *, "Iter. to solve:  ", iter
      print *, "Tolerance:       ", tol
      print *, "Relative error:  ", relerr

      if (relerr < tol) then
        print *, "Solution: CORRECT"
      else
        print *, "Solution: WRONG"
      end if
      write(*,'(a)') ""

      ! Deallocate matrices
      deallocate(A, Aref, X, B, iPiv, work, stat=info)

      END PROGRAM TEST_ZCGESV
