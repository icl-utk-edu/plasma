!>
!> @file
!>
!>  PLASMA is a software package provided by:
!>  University of Tennessee, US,
!>  University of Manchester, UK.
!>
!> @precisions normal z -> s d c
!>
program test_zgels
    use, intrinsic :: iso_fortran_env

    use plasma
    implicit none

    integer, parameter :: sp = REAL32
    integer, parameter :: dp = REAL64

    ! set working precision, this value is rewritten for different precisions
    integer, parameter :: wp = dp

    ! matrices
    complex(wp), allocatable :: A(:,:)
    complex(wp), allocatable :: Aref(:,:)
    complex(wp), allocatable :: B(:,:)
    complex(wp), allocatable :: Bref(:,:)

    ! matrix descriptor for storing coefficients of the Householder reflectors
    ! generalization of TAU array known from LAPACK
    type(plasma_desc_t) :: descT

    ! dimensions
    integer :: m
    integer :: n
    integer :: nrhs
    integer :: lda
    integer :: ldb
    integer :: nb
    integer :: ib
    logical :: tree_householder

    real(wp) :: tol

    ! auxiliary variables
    integer :: info

    complex(wp), parameter :: zone  =  1.0
    complex(wp), parameter :: zmone = -1.0
    complex(wp), parameter :: zzero =  0.0
    integer :: seed(4) = [0, 0, 0, 1]

    real(wp) :: Anorm
    real(wp) :: Bnorm
    real(wp) :: Xnorm
    real(wp) :: residual

    real(wp) :: work(1)
    real(wp) :: zlange, dlamch

    ! Set parameters.
    m    = 2000 ! number of rows
    n    = 500  ! number of columns
    nrhs = 1000 ! number of right-hand sides

    nb   = 256  ! tile size (square tiles)
    ib   = 64   ! inner blocking size within a tile

    ! Should tree reduction be used? This is good for tall and skinny matrices.
    tree_householder = .true.

    ! tolerance for the correctness check
    tol = 50.0 * dlamch('E')

    ! Initialize arrays.
    lda  = m    ! leading dimension of A
    allocate(A(lda, n), Aref(lda, n))
    call zlarnv(1, seed, lda*n, A)
    Aref = A

    ldb  = max(m, n)  ! leading dimension of right-hand side matrix B
    allocate(B(ldb, nrhs), Bref(ldb, nrhs))
    call zlarnv(1, seed, ldb*nrhs, B)
    Bref = B

    write(*,'(a, i6, a, i6)') " Testing least-squares solution with a matrix ", m, " x ", n
    write(*,'(a, i6, a)')     " and ", nrhs, " right-hand sides."
    write(*,'(a, i6)')        "  tile size:        ", nb
    write(*,'(a, i6)')        "  inner block size: ", ib
    if (tree_householder) then
        write(*,'(a, i6)')    "  Using tree-based Householder reduction. "
    else
        write(*,'(a, i6)')    "  Using flat-tree Householder reduction. "
    end if

    !==============================================
    ! Initialize PLASMA.
    !==============================================
    call plasma_init(info)
    call check_error('plasma_init()', info)

    !==============================================
    ! Set PLASMA parameters.
    !==============================================
    ! set Householder mode - PlasmaFlatHouseholder is the default
    if (tree_householder) then
        call plasma_set(PlasmaHouseholderMode, PlasmaTreeHouseholder, info)
    else
        call plasma_set(PlasmaHouseholderMode, PlasmaFlatHouseholder, info)
    end if
    call check_error('plasma_set()', info)

    !=============================================================
    ! Solve problem problem A X = B using least-squares if m >= n,
    ! or with minimum norm solution if m < n.
    !=============================================================
    call plasma_zgels(PlasmaNoTrans, m, n, nrhs, A, lda, descT, B, ldb, info)
    call check_error('plasma_zgels()', info)

    !==============================================
    ! Deallocate matrix in descriptor.
    !==============================================
    call plasma_desc_destroy(descT, info)
    call check_error('plasma_desc_destroy()', info)

    !==============================================
    ! Finalize PLASMA.
    !==============================================
    call plasma_finalize(info)
    call check_error('plasma_finalize()', info)


    ! Test results by computing residual.
    ! |A|_F
    Anorm = zlange('F', m, n, Aref, lda, work)

    ! |B|_F
    Bnorm = zlange('F', m, nrhs, Bref, ldb, work)

    ! |X|_F, solution X is now stored in the n-by-nrhs part of B
    Xnorm = zlange('F', n, nrhs, B, ldb, work)

    ! compute residual and store it in B = A*X - B
    call zgemm('No transpose', 'No transpose', m, nrhs, n, &
               zone, Aref, lda, B, ldb, zmone, Bref, ldb)

    ! Compute B = A^H * (A*X - B)
    call zgemm('Conjugate transpose', 'No transpose', n, nrhs, m, &
               zone, Aref, lda, Bref, ldb, zzero, B, ldb)

    ! |RES|_F
    residual = zlange('F', n, nrhs, B, ldb, work)

    ! normalize the result
    residual = residual / ((Anorm*Xnorm+Bnorm)*n);

    ! print output of the check
    write(*,'(a, e13.6)') &
        "  Normalized norm of residual : ", residual
    if (residual < tol) then
        write(*,'(a)') "  The result is CORRECT."
    else
        write(*,'(a)') "  The result is WRONG!"
    end if
    write(*,'(a)') ""

    deallocate(A)
    deallocate(Aref)
    deallocate(B)
    deallocate(Bref)

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

end program test_zgels
