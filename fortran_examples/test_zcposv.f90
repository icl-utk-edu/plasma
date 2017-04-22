!>
!! @file
!!
!!  PLASMA is a software package provided by:
!!  University of Tennessee,  US,
!!  University of Manchester, UK
!!
!! @precisions mixed zc -> ds
!!
!! @brief Tests Cholesky factorization-based iterative refinement
!!  routine for solution of linear system A*X=B, where A is
!!  n-by-n Hermitian positive definite matrix, and X, B are n-by-nrhs
!!  matrices

      PROGRAM TEST_ZCPOSV

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
      integer :: uploPlasma = PlasmaLower
      integer :: i, iter, info

      ! LAPACK variables
      integer :: iSeed(8) = [3476, 1940, 2069, 1681, 2151, 822, 3982, 757]

      complex(dp) :: zone = 1.0, zmone = -1.0

      real(dp), allocatable :: work(:)
      real(dp)              :: Anorm, Xnorm, Rnorm, relerr

      ! Test variables
      ! character(len=32) :: frmt ="(10(3X,F7.3,SP,F7.3,'i'))"
      real(dp)          :: tol  = 0.0d+0

      ! Performance variables
      real(dp) :: tstart = 0.0, tstop = 0.0, telapsed = 0.0d+0

      ! External functions
      real(dp), external :: dlamch
      real(dp), external :: zlange


      lda = max(1,n)
      ldb = lda
      ldx = lda

      tol = 50.0d+0 * dlamch('E')

      ! Allocate matrices
      allocate(A(lda,n), Aref(lda,n), X(ldx,nrhs), B(ldb,nrhs), work(n))

      !=================================================================
      ! Generate and initialize random matrices A and B
      !=================================================================

      ! Generate Hermitian positive definite matrix A
      call zlarnv(1, iSeed(1:4), lda*n, A)

      A = 0.5 * ( A+conjg(transpose(A)) )
      ! A = A * conjg(transpose(A))

      do i = 1, n
        A(i,i) = A(i,i) + n
      end do

      ! @test Print out matrix A
!       print *, "Random Hermitian positive definite matrix A:"
!       do i = 1, lda
!         print frmt, A(i,:)
!       end do
!       print *, ""

      ! Save initial matrix A into Aref
      Aref = A

      ! Generate random matrix B
      call zlarnv(1, iSeed(5:8), ldb*nrhs, B)

      ! @test Print out matrix B
!       print *, "Random matrix B:"
!       do i = 1, ldb
!         print frmt, B(i,:)
!       end do
!       print *, ""

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

      call plasma_zcposv(uploPlasma, n, nrhs, A, lda, B, ldb, X, ldx, &
                         iter, info)

      tstop    = omp_get_wtime()
      telapsed = tstop - tstart

      ! @test Print out solution matrix X
!       print *, "Solution matrix X:"
!       do i = 1, ldx
!         print frmt, X(i,:)
!       end do
!       print *, ""

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

      ! Calculate infinite norm of resudual matrix R
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
      deallocate(A, Aref, X, B, work, stat=info)

      END PROGRAM TEST_ZCPOSV
