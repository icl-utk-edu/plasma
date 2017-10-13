!>
!> @file
!>
!>  PLASMA is a software package provided by:
!>  University of Tennessee, US,
!>  University of Manchester, UK.
!>
!> @precisions normal z -> s d c
!>
program test_zgetrs
    use, intrinsic :: iso_fortran_env

    use plasma
    implicit none

    integer, parameter :: sp = REAL32
    integer, parameter :: dp = REAL64

    ! set working precision, this value is rewritten for different precisions
    integer, parameter :: wp = dp

    ! matrix
    complex(wp), allocatable :: A(:,:)
    complex(wp), allocatable :: Aref(:,:)

    ! right-hand side
    complex(wp), allocatable :: B(:,:)
    complex(wp), allocatable :: Bref(:,:)

    ! vector of row permutations
    integer, allocatable :: ipiv(:)

    ! dimensions
    integer :: n
    integer :: nrhs

    integer :: lda
    integer :: ldb

    integer :: nb
    integer :: ib

    integer :: num_panel_threads

    real(wp) :: tol

    ! auxiliary variables
    integer :: info

    integer :: seed(4) = [0, 0, 0, 1]

    real(wp) :: normA
    real(wp) :: normX
    real(wp) :: normR

    real(wp), allocatable :: work(:)
    real(wp) :: zlange, dlamch

    integer :: num_panel_threads_check

    ! Set parameters.
    n    = 2000 ! number of rows and columns
    nrhs = 1000 ! number of right-hand sides

    nb   = 256  ! tile size (square tiles)
    ib   = 64   ! inner blocking size within a tile
    num_panel_threads = 2  ! number of threads for panel factorization

    ! tolerance for the correctness check
    tol = 50.0 * dlamch('E')

    ! Initialize arrays.
    lda  = n    ! leading dimension of A
    allocate(A(lda, n), Aref(lda, n))
    call zlarnv(1, seed, lda*n, A)
    Aref = A

    ldb  =  n  ! leading dimension of right-hand side matrix B
    allocate(B(ldb, nrhs), Bref(ldb, nrhs))
    call zlarnv(1, seed, ldb*nrhs, B)
    Bref = B

    allocate(ipiv(n))

    write(*,'(a, i6, a, i6)') " Testing LU solution with a matrix ", n, " x ", n
    write(*,'(a, i6)')        "  tile size:        ", nb
    write(*,'(a, i6)')        "  inner block size: ", ib

    !==============================================
    ! Initialize PLASMA.
    !==============================================
    call plasma_init(info)
    call check_error('plasma_init()', info)

    !==============================================
    ! Set PLASMA parameters.
    !==============================================
    ! set tile size
    call plasma_set(PlasmaNb, nb, info)
    call check_error('plasma_set()', info)

    ! set inner blocking size
    call plasma_set(PlasmaIb, ib, info)
    call check_error('plasma_set()', info)

    ! set number of threads for panel factorization
    call plasma_set(PlasmaNumPanelThreads, num_panel_threads, info)
    call check_error('plasma_set()', info)

    call plasma_get(PlasmaNumPanelThreads, num_panel_threads_check, info)
    if (num_panel_threads_check /= num_panel_threads) then
        write(*,'(a, i2, a, i2)') &
            "Something is wrong - number of threads for panel factorization " &
            //"is not set correctly:", &
            num_panel_threads_check, " not the same as ", num_panel_threads
    end if

    !==============================================
    ! Call LU factorization for A.
    !==============================================
    call plasma_zgetrf(n, n, A, lda, ipiv, info)
    call check_error('plasma_zgetrf()', info)

    !==============================================
    ! Call LU back-substitution for B.
    !==============================================
    call plasma_zgetrs(n, nrhs, A, lda, ipiv, B, ldb, info)
    call check_error('plasma_zgetrs()', info)

    !==============================================
    ! Finalize PLASMA.
    !==============================================
    call plasma_finalize(info)
    call check_error('plasma_finalize()', info)

    allocate(work(n))
    normA = zlange('I', n, n, Aref, lda, work)

    normX = zlange('I', n, nrhs, B, ldb, work)

    ! compute residual B - A*X
    Bref = Bref - matmul(Aref, B)
    normR = zlange('I', n, nrhs, Bref, ldb, work)

    ! normalize the residual
    if (normA /= 0.0 .and. normX /= 0.0) then
       normR = normR / (n * normA * normX)
    end if

    ! print output of the check
    write(*,'(a, e13.6)') &
        "  Residual of solution, measured as |B - A*X|_oo: ", normR
    if (normR < tol) then
        write(*,'(a)') "  The result is CORRECT."
    else
        write(*,'(a)') "  The result is WRONG!"
    end if
    write(*,'(a)') ""

    deallocate(A)
    deallocate(Aref)
    deallocate(B)
    deallocate(Bref)
    deallocate(ipiv)
    deallocate(work)

contains

    subroutine check_error(routine_name, code)
        use plasma
        implicit none
        character(len=*) :: routine_name
        integer :: code

        if (code /= PlasmaSuccess) then
            write(*,'(a, i2, a, a)') "Error ", code, "in a call to ", &
                trim(routine_name)
        end if
    end subroutine

end program test_zgetrs
