!>
!> @file
!>
!>  PLASMA is a software package provided by:
!>  University of Tennessee, US,
!>  University of Manchester, UK.
!>
!> @precisions normal z -> s d c
!>
!>    @brief Tests PLASMA linear system (Ax=b) solution based on
!>           Cholesky factorization

      PROGRAM TEST_ZPOTRS

      USE, INTRINSIC :: ISO_FORTRAN_ENV
      USE OMP_LIB
      USE PLASMA

      IMPLICIT NONE

      integer, parameter :: sp = REAL32
      integer, parameter :: dp = REAL64

      ! set working precision, this value is rewritten for different precisions
      integer, parameter :: wp = dp

      integer,     parameter   :: n = 2000, nrhs = 1000
      complex(wp), parameter   :: zone = 1.0, zmone = -1.0
      complex(wp), allocatable :: A(:,:), Aref(:,:), B(:,:), Bref(:,:)
      real(wp),    allocatable :: work(:)
      integer                  :: seed(4) = [0, 0, 0, 1]
      real(wp)                 :: Anorm, Xnorm, Rnorm, error, tol
      character                :: uploLapack ='L'
      integer                  :: uploPlasma = PlasmaLower
      !character(len=32)        :: frmt ="(10(3X,F7.3,SP,F6.3,'i'))"
      integer                  :: lda, ldb, infoPlasma, infoLapack, i
      logical                  :: success = .false.

      ! Performance variables
      real(dp) :: tstart, tstop, telapsed

      ! External functions
      real(wp), external :: dlamch, zlanhe, zlange


      tol = 50.0 * dlamch('E')
      print *, "tol:", tol

      lda = max(1,n);  ldb = max(1,n)

      ! Allocate matrices A, B, Re, Im
      allocate(A(lda,n), B(ldb,nrhs), stat=infoPlasma)

      ! Generate random Hermitian positive definite matrix A
      call zlarnv(1, seed, lda*n, A)
      A = A * conjg(transpose(A))
      do i = 1, n
        A(i,i) = A(i,i) + n
      end do

      !print *, "Random Hermitian positive definite matrix A:"
      !do i = 1, n
      !  print frmt, A(i,:)
      !end do
      !print *, ""

      ! Generate random matrix B
      call zlarnv(1, seed, ldb*nrhs, B)
      !print *, "Random matrix B:"
      !do i = 1, n
      !  print frmt, B(i,:)
      !end do
      !print *, ""

      allocate(Aref(lda,n), Bref(ldb,nrhs), work(n), stat=infoPlasma)
      Aref = A;  Bref = B;  work = 0.0

      ! Factorize matrix A with Cholesky
      call zpotrf(uploLapack, n, A, lda, infoLapack)

      !print *, "Factor " // uploLapack // " of Chol(A):"
      !do i = 1, n
      !  print frmt, A(i,:)
      !end do
      !print *, ""

      !===================
      ! Initialize PLASMA.
      !===================
      call plasma_init(infoPlasma)

      !============================================================
      ! Solve linear system (Ax=b) using factor L or U from Chol(A)
      !============================================================
      tstart = omp_get_wtime()
      call plasma_zpotrs(uploPlasma, n, nrhs, A, lda, B, ldb, infoPlasma)
      tstop  = omp_get_wtime()
      telapsed  = tstop-tstart

      !=================
      ! Finalise PLASMA.
      !=================
      call plasma_finalize(infoPlasma)

      !print *, "Solution matrix X:"
      !do i = 1, n
      !  print frmt, B(i,:)
      !end do
      !print *, ""

      print *, "Time:", telapsed
      print *, ""

      ! Check linear system (Ax=b) solution is correct

      ! Calculate residual (R=b-Ax), Bref := -Aref*B + Bref
      call zgemm('N', 'N', n, nrhs, n, zmone, Aref, lda, B, ldb, &
                  zone, Bref, ldb)

      !print *, "Residual R=b-Ax:"
      !do i = 1, n
      !  print frmt, Bref(i,:)
      !end do
      !print *, ""

      ! Calculate norms |Aref|_I, |X|_I, |R|_I
      Anorm = zlanhe('I', uploLapack, n, Aref, lda, work)

      Xnorm = zlange('I', n, nrhs, B,    ldb, work)
      Rnorm = zlange('I', n, nrhs, Bref, ldb, work)

      print *, "|Aref|_I:", Anorm
      print *, "|X|_I:   ", Xnorm
      print *, "|R|_I:   ", Rnorm
      print *, ""

      if (Anorm > 0.0 .and. Xnorm > 0.0) then

        ! Calculate error |b-Ax|_I / (|A|_I * |x|_I * n)
        error = Rnorm/Anorm/Xnorm/n

        if (error < tol)  success = .true.

      else
          error = -1.0
      end if

      print *, "|b-Ax|_I / (|A|_I * |x|_I * n):", error
      print *, "success:                       ", success

      if (success) then
          write(*,'(a)') "  The result is CORRECT."
      else
          write(*,'(a)') "  The result is WRONG!"
      end if
      write(*,'(a)') ""

      ! Deallocate matrices Aref, Bref
      deallocate(Aref, Bref, work, stat=infoPlasma)

      ! Deallocate matrices A, B
      deallocate(A, B, stat=infoPlasma)

      END PROGRAM TEST_ZPOTRS
