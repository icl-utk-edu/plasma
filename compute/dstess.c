/**
 *
 * @file
 *
 *  plasma is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/

#include "plasma.h"
#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_tuning.h"
#include "plasma_types.h"
#include "plasma_workspace.h"
#include "plasma_core_blas.h"

#include <string.h>
#include <omp.h>
#include <math.h>
#include <float.h>
#include "mkl_lapack.h"

// The debug lines individually conditional, at minimum use DSTESS_DEBUG(1) to
// include code, or DSTESS_DEBUG(0) to not include it.  Braced code can follow.
// The advantage of this approach over some others is the compiler still checks
// the debug code for syntax errors, preventing bit rot, and the code can be as
// complex as you like, but the code will be eliminated by optimization as
// never executed when using DSTESS_DEBUG(0). 
#define  DSTESS_DEBUG(condition) if (condition)

// #include "core_lapack.h" // Original LAPACK Fortran style prototypes.
// void dlatsqr_(int *M, int *N, int *MB, int *NB, double *A, int *LDA, double *T, int *LDT, double *WORK, int *LWORK, int *INFO);
// void dorgtsqr_(int *M, int *N, int *MB, int *NB, double *A, int *LDA, double *T, int *LDT, double *WORK, int *LWORK, int *INFO);
// void dstein_(int *N, double *D, double *E, int *M, double *W, int *IBLOCK, int *ISPLIT,
//            double *Z, int *LDZ, double *WORK, int *IWORK, int *IFAIL, int *INFO);


/*******************************************************************************
 *
 * plasma_dstess: Symmetric Tridiagonal Eigen Spectrum Slicer.
 *
 * Computes a caller-selected range of eigenvalues and, optionally,
 * eigenvectors of a symmetric tridiagonal matrix A.  Eigenvalues and
 * eigenvectors can be selected by specifying either a range of values or a
 * range of indices for the desired eigenvalues.
 *
 * This is similiar to the LAPACK routine dstevx. 
 *
 * Because input matrices are expected to be extremely large and the exact
 * number of eigenvalues is not necessarily known to the caller, this routine
 * provides a way to get the number of eigenvalues in either a value range or
 * an index range; so the caller can allocate the return arrays. There are
 * three; the floating point vector pVal, the integer vector pMul, and the
 * floating point matrix pVec, which is allocated only for PLasmaVec.  It is
 * the user's responsibility to free() this memory.
 *
 * K eigenvalues (or pairs) are found. If K=0, then no allocations are made. The
 * code here will initialize the pVal and pMul arrays to zeros.
 *
 * When the runType is PlasmaCount; the code returns the number of eigenvalues 
 * in the caller-selected range, call that nEigVals. 
 *
 * However, upon return from runType=PlasmaVec or runType=PlasmaNoVec, the code
 * returns the number of unique eigenvalues found, call that nFound. For a
 * symmetric matrix we should have K unique eigenvalues, but due to the limits
 * of machine precision, multiple arithmetically unique eigenvalue may be
 * approximated by the same floating point number. In that case, to the machine
 * this looks like a multiplicity; and we report it that way. We see this
 * phenomenon in large (e.g. N=50000) Wilkinson matrices, for example.
 *
 * Finding eigenvalues alone is much faster than finding eigenpairs; the
 * majority of the time consumed when eigenvectors are found is in
 * orthogonalizing the vectors; an O(N*K^2) operation. 
 *
 * The 3rd and 4th arguments are n,k. n is the length of the 'diag' vector.
 * k is the length the user has allocated for pVal[] and pMul[], and pVec[].
 * If eigenvectors are requested, pVec[] is an (n*k) column major array, otherwise it is 
 * not referenced. If nFound != nEigVals, then entries in pVal, pMul, and pVec
 * beyond the nFound entry or column will contain left-over data.
 *******************************************************************************

 *
 * @param[in] eigt
 *          enum:
 *          = PlasmaNoVec: computes eigenvalues only;
 *          = PlasmaVec:   computes eigenvalues and eigenvectors.
 *          = PlasmaCount: computes pFound as the max number of eigenvalues/pairs
 *                         in the given range, so user can allocate 
 *                         pVal[Found], pMult[Found], pVec[n x Found].
 *
 * @param[in] range
 *          enum:
 *          PlasmaRangeV use vl, vu for range [vl, vu)
 *          PlasmaRangeI use il, iu for range [il, iu]. 1-relative; 1..N. 
 *
 * @param[in] n
 *          int. The order of the matrix A. n >= 0.
 *
 * @param[in] k
 *          int. The space the user has allocated for eigenvalues; as reflected in
 *          pVal, pMul, pVec. 
 *
 * @param[in] diag
 *          double[n]. Vector of [n] diagonal entries of A. 
 *
 * @param[in] offd
 *          double[n-1]. A vector of [n-1] off-diagonal entries of A.
 *
 * @param[in] vl   
 *          double. Lowest eigenvalue in desired range [vl, vu).
 *          if less than Gerschgorin min; we use Gerschgorin min.
 *
 * @param[in] vu
 *          double. Highest eigenvalue in desired range, [vl,vu).
 *          if greater than Gerschgorin max, we use Gerschgorin max.
 *
 * @param[in] il
 *          int. Low Index of range. Must be in range [1,n].
 *
 * @param[in] iu
 *          int. High index of range. Must be in range [1,n], >=il.
 *
 * @param[out] pFound
 *          int*. On exit, the number of distinct eigenvalues (or pairs) found.
 *          Or for eigt=PlasmaCount, the maximum number of distinct
 *          eigenvalues in the interval selected by range, [vl,vu) or [il,iu].
 *
 * @param[out] pVal
 *          double*. expect double Val[k]. The first 'found' elements are the 
 *          found eigenvalues.
 *
 * @param[out] pMul
 *          int*. expect int Mul[k]. The first 'found' elements are the 
 *          multiplicity values.
 *
 * @param[out] pVec
 *          double*. Expect double Vec[n*k]. the first ('n'*'found') elements 
 *          contain an orthonormal set of 'found' eigenvectors, each of 'n'
 *          elements, in column major format. e.g. eigenvector j is found in
 *          Vec[n*j+0] ... Vec[n*j+n-1]. It corresponds to eigenvalue Val[j],
 *          with multiplicity Mul[j].
 *          if eigt=PlasmaNoVec, then pVec is not referenced.
 *
 *******************************************************************************
 *
 * @retval PlasmaSuccess successful exit
 * @retval < 0 if -i, the i-th argument had an illegal value
 *
 *******************************************************************************
*/

// This is a bracket for a range; any children are subdivisions of the range.
// We give each thread a bracket. The thread bisects the bracket until it has
// one eigenvalue, throwing off children in the chain as it does. Empty
// divisions are discarded; they won't be in the chain; e.g. if I divide the
// range in half and one half contains no eigenvalues, then we just update
// the existing bracket appropriately and divide again.

enum {Init, Bisection, GetVector}; 

typedef struct
{
   int    stage;        // stage of operations on this bracket.
   
   double lowerBound;
   double upperBound;
   int    nLT_low;      // # < lowerBound. -1 if it needs to be found.
   int    nLT_hi;       // # < upperBound. -1 if it needs to be found.
   int    numEV;        // number of Eigenvalues in bracket.
   void   *next;        // A bracket subdivides to more brackets.
} EvBracket;

// dstein needs work areas to function. Instead of allocating and 
// deallocating these for every call, we provide a set of work areas
// per thread that can be re-used. So if we have 72 threads, and
// a thousand vectors to find, we will only allocate the work areas
// 72 times, and only free them 72 times at the end, not a thousand
// times. 

typedef struct
{
   int      *IBLOCK;
   int      *ISPLIT;
   double   *WORK;
   int      *IWORK;
   int      *IFAIL;
} Stein_Array_t;

// This is the workstack.
typedef struct
{
   int   baseIdx;       // Number of EV less than user's low threshold.
   plasma_enum_t range; // PlasmaRangeV or PlasmaRangeI.
   plasma_enum_t eigt;  // PlasmaNoVec or PlasmaVec.
   int   il, iu;        // For PlasmaRangeI. 
   int   eigenvalues;   // total number of eigenvalues to find.
   int   finished;      // # of finished eigenvectors.
   double *diag;        // some stuff the threads need.
   double *offd;        // ...
   int   N;             // size of the matrix.
   EvBracket* ToDo;     // NULL or chain head of EVBrackets yet to be processed.
   Stein_Array_t* stein_arrays; // Workspaces per thread for useDstein.
   int error;           // first error, if non-zero.
   double *pVal;        // where to store eigenvalues. 
   int    *pMul;        // where to store Multiplicity.
   double *pVec;        // where to store eigenvectors.
} WorkStack;

//-----------------------------------------------------------------------------
//SUBROUTINE DORGTSQR( M, N, MB, NB, A, LDA, T, LDT, WORK, LWORK,
//     $                     INFO )
//
//      .. Scalar Arguments ..
//      INTEGER           INFO, LDA, LDT, LWORK, M, N, MB, NB
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION  A( LDA, * ), T( LDT, * ), WORK( * )
//      ..
//
// \par Purpose:
// =============
// DORGTSQR generates an M-by-N real matrix Q_out with orthonormal columns,
// which are the first N columns of a product of real orthogonal
// matrices of order M which are returned by DLATSQR
//
//      Q_out = first_N_columns_of( Q(1)_in * Q(2)_in * ... * Q(k)_in ).
//
// See the documentation for DLATSQR.
//
// Arguments:
// ==========
//
// \param[in] M
//          M is INTEGER
//          The number of rows of the matrix A.  M >= 0.
//
// \param[in] N
//          N is INTEGER
//          The number of columns of the matrix A. M >= N >= 0.
//
// \param[in] MB
//          MB is INTEGER
//          The row block size used by DLATSQR to return
//          arrays A and T. MB > N.
//          (Note that if MB > M, then M is used instead of MB
//          as the row block size).
//
// \param[in] NB
//          NB is INTEGER
//          The column block size used by DLATSQR to return
//          arrays A and T. NB >= 1.
//          (Note that if NB > N, then N is used instead of NB
//          as the column block size).
//
// \param[in,out] A
//          A is DOUBLE PRECISION array, dimension (LDA,N)
//
//          On entry:
//
//             The elements on and above the diagonal are not accessed.
//             The elements below the diagonal represent the unit
//             lower-trapezoidal blocked matrix V computed by DLATSQR
//             that defines the input matrices Q_in(k) (ones on the
//             diagonal are not stored) (same format as the output A
//             below the diagonal in DLATSQR).
//
//          On exit:
//
//             The array A contains an M-by-N orthonormal matrix Q_out,
//             i.e the columns of A are orthogonal unit vectors.
//
// \param[in] LDA
//          LDA is INTEGER
//          The leading dimension of the array A.  LDA >= max(1,M).
//
// \param[in] T
//          T is DOUBLE PRECISION array,
//          dimension (LDT, N * NIRB)
//          where NIRB = Number_of_input_row_blocks
//                     = MAX( 1, CEIL((M-N)/(MB-N)) )
//          Let NICB = Number_of_input_col_blocks
//                   = CEIL(N/NB)
//
//          The upper-triangular block reflectors used to define the
//          input matrices Q_in(k), k=(1:NIRB*NICB). The block
//          reflectors are stored in compact form in NIRB block
//          reflector sequences. Each of NIRB block reflector sequences
//          is stored in a larger NB-by-N column block of T and consists
//          of NICB smaller NB-by-NB upper-triangular column blocks.
//          (same format as the output T in DLATSQR).
//
// \param[in] LDT
//          LDT is INTEGER
//          The leading dimension of the array T.
//          LDT >= max(1,min(NB1,N)).
//
// \param[out] WORK
//          (workspace) DOUBLE PRECISION array, dimension (MAX(2,LWORK))
//          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
//
// \param[in] LWORK
//          The dimension of the array WORK.  LWORK >= (M+NB)*N.
//          If LWORK = -1, then a workspace query is assumed.
//          The routine only calculates the optimal size of the WORK
//          array, returns this value as the first entry of the WORK
//          array, and no error message related to LWORK is issued
//          by XERBLA.
//
// \param[out] INFO
//          INFO is INTEGER
//          = 0:  successful exit
//          < 0:  if INFO = -i, the i-th argument had an illegal value
//
// Authors:
// ========
//
// \author Univ. of Tennessee
// \author Univ. of California Berkeley
// \author Univ. of Colorado Denver
// \author NAG Ltd.
//
// \ingroup doubleOTHERcomputational
//
// \par Contributors:
// ==================
// November 2019, Igor Kozachenko,
//                Computer Science Division,
//                University of California, Berkeley
// =====================================================================
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// DLATSQR computes a blocked Tall-Skinny QR factorization of
// a real M-by-N matrix A for M >= N:
//
//    A = Q * ( R ),
//            ( 0 )
// where:
//    Q is a M-by-M orthogonal matrix, stored on exit in an implicit
//    form in the elements below the diagonal of the array A and in
//    the elements of the array T;
//
//    R is an upper-triangular N-by-N matrix, stored on exit in
//    the elements on and above the diagonal of the array A.
//
//    0 is a (M-N)-by-N zero matrix, and is not stored.
//
// Arguments:
// ==========
//
// \param[in] M
//          M is INTEGER
//          The number of rows of the matrix A.  M >= 0.
//
// \param[in] N
//          N is INTEGER
//          The number of columns of the matrix A. M >= N >= 0.
//
// \param[in] MB
//          MB is INTEGER
//          The row block size to be used in the blocked QR.
//          MB > N.
//
// \param[in] NB
//          NB is INTEGER
//          The column block size to be used in the blocked QR.
//          N >= NB >= 1.
//
// \param[in,out] A
//          A is DOUBLE PRECISION array, dimension (LDA,N)
//          On entry, the M-by-N matrix A.
//          On exit, the elements on and above the diagonal
//          of the array contain the N-by-N upper triangular matrix R;
//          the elements below the diagonal represent Q by the columns
//          of blocked V (see Further Details).
//
// \param[in] LDA
//          LDA is INTEGER
//          The leading dimension of the array A.  LDA >= max(1,M).
//
// \param[out] T
//          T is DOUBLE PRECISION array,
//          dimension (LDT, N * Number_of_row_blocks)
//          where Number_of_row_blocks = CEIL((M-N)/(MB-N))
//          The blocked upper triangular block reflectors stored in compact form
//          as a sequence of upper triangular blocks.
//          See Further Details below.
//
// \param[in] LDT
//          LDT is INTEGER
//          The leading dimension of the array T.  LDT >= NB.
//
// \param[out] WORK
//         (workspace) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
//
// \param[in] LWORK
//          The dimension of the array WORK.  LWORK >= NB*N.
//          If LWORK = -1, then a workspace query is assumed; the routine
//          only calculates the optimal size of the WORK array, returns
//          this value as the first entry of the WORK array, and no error
//          message related to LWORK is issued by XERBLA.
//
// \param[out] INFO
//          INFO is INTEGER
//          = 0:  successful exit
//          < 0:  if INFO = -i, the i-th argument had an illegal value
//
// Authors:
// ========
//
// \author Univ. of Tennessee
// \author Univ. of California Berkeley
// \author Univ. of Colorado Denver
// \author NAG Ltd.
//
// \par Further Details:
// =====================
//
// Tall-Skinny QR (TSQR) performs QR by a sequence of orthogonal transformations,
// representing Q as a product of other orthogonal matrices
//   Q = Q(1) * Q(2) * . . . * Q(k)
// where each Q(i) zeros out subdiagonal entries of a block of MB rows of A:
//   Q(1) zeros out the subdiagonal entries of rows 1:MB of A
//   Q(2) zeros out the bottom MB-N rows of rows [1:N,MB+1:2*MB-N] of A
//   Q(3) zeros out the bottom MB-N rows of rows [1:N,2*MB-N+1:3*MB-2*N] of A
//   . . .
//
// Q(1) is computed by GEQRT, which represents Q(1) by Householder vectors
// stored under the diagonal of rows 1:MB of A, and by upper triangular
// block reflectors, stored in array T(1:LDT,1:N).
// For more information see Further Details in GEQRT.
//
// Q(i) for i>1 is computed by TPQRT, which represents Q(i) by Householder vectors
// stored in rows [(i-1)*(MB-N)+N+1:i*(MB-N)+N] of A, and by upper triangular
// block reflectors, stored in array T(1:LDT,(i-1)*N+1:i*N).
// The last Q(k) may use fewer rows.
// For more information see Further Details in TPQRT.
//
// For more details of the overall algorithm, see the description of
// Sequential TSQR in Section 2.2 of [1].
//
// [1] “Communication-Optimal Parallel and Sequential QR and LU Factorizations,”
//     J. Demmel, L. Grigori, M. Hoemmen, J. Langou,
//     SIAM J. Sci. Comput, vol. 34, no. 1, 2012
//-----------------------------------------------------------------------------
static int useDlatsqr(double *A, int M, int N) {
   int MB, NB, LDA, LDT, RowB, LWORK, INFO;
   double *T, *WORK;

   // MB must be >N. Note we divide (M-N) by (MB-N) for RowB,
   // so we don't want MB-N near zero.
   // What we'd like is MB-N to be a power of 2, so MB=N+2^x.
   int i=512+N;  // test MB-N = 512.
   while ( (i-N) < (2*N) ) {i=2*(i-N)+N;}
   MB = i;
   if (MB>M) MB=M;

   if (N < 128) NB=N;
   else         NB=128;
   
   LDA=M;

   // Note if (MB-N) is small, this produces a massive RowB, and we cannot allocate that much memory.
   RowB = (M-N)/(MB-N);
   if ((MB-N)*RowB < (M-N)) RowB++;
   DSTESS_DEBUG(0) fprintf(stderr, "MB=%i RowB=%i.\n", MB, RowB);

   LDT = NB;

   T = malloc(LDT*N*RowB*sizeof(double));
   // If we cannot allocate: Try to be sensible and try again.
   // The formula divides by (MB-N) which is the culprit; MB-N is three.
   // Force MB=M so RowB = 1.
   if (T == NULL) {
      MB=M;
      RowB = 1;
      T = malloc(LDT*N*RowB*sizeof(double));
      if (T == NULL) {
         DSTESS_DEBUG(0) fprintf(stderr, "%s:%i Out of memory for T, LDT*N*RowB=%i.\n", __func__, __LINE__, LDT*N*RowB);
         return(-1);
      }
   }

   // For dlatsqr, LWORK=NB*N, but for dorgtsqr, we need (M+NB)*N.
   LWORK = (M+NB)*N;

   WORK = malloc(LWORK*sizeof(double));

   DSTESS_DEBUG(0) fprintf(stderr, "M=%i N=%i MB=%i NB=%i A=%p LDA=%i T=%p LDT=%i WORK=%p LWORK=%i\n", 
      M, N, MB, NB, (void*) A, LDA, (void*) T, LDT, (void*) WORK, LWORK);
   dlatsqr_( &M, &N, &MB, &NB, A, &LDA, T, &LDT, WORK, &LWORK, &INFO);
   if (INFO != 0) {
      DSTESS_DEBUG(0) fprintf(stderr, "LAPACK_dlatsqr returned INFO = %i.\n", INFO);
      return(INFO);
   }

   DSTESS_DEBUG(0) fprintf(stderr, "Checkpoint after dlatsqr_(), Line=%i.\n", __LINE__);
   dorgtsqr_( &M, &N, &MB, &NB, A, &LDA, T, &LDT, WORK, &LWORK, &INFO);   
   DSTESS_DEBUG(0) fprintf(stderr, "Checkpoint after dorgtsqr_(), Line=%i.\n", __LINE__);
   if (INFO != 0) {
      DSTESS_DEBUG(0) fprintf(stderr, "LAPACK_dorgtsqr returned INFO = %i.\n", INFO);
      return(INFO);
   }

   free(T);
   free(WORK);
   return(0);
} // end useDlatsqr

//-----------------------------------------------------------------------------
// useDstein: Uses lapack routine dstein() to recover a single eigenvector.
// returns INFO.
//-----------------------------------------------------------------------------
static int useDstein( double *diag, double *offd, double u, double *v, int N, Stein_Array_t *myArrays) {

// parameters to dstein: 
// dstein(int *N, double *D, double *E, int *M, double *W, int *IBLOCK, int *ISPLIT,
//        double *Z, int *LDZ, double *WORK, int *IWORK, int *IFAIL, int *INFO);
//
// int N             The order of the matrix.
// double D[N]       The N diagonal entries.
// double E[N-1]     The N-1 offdiagonal entries.
// int M             The number of eigenvectors to be found.
// double W[M]       The M eigenvalues, smallest to largest.
// int IBLOCK[N]     submatrix indices: 1 for block 1, 2 for block 2, etc.
//                   one indicator for each eigenvalue in W.
// int ISPLIT[N]     indices where matrix splits: first is (1,ISPLIT[0]), 
//                   next is (ISPLIT[0]+1, ISPLIT[1]), etc. 
// double Z[LDZ,M]   2 dimensional array, eigenvector for W[i] is stored in the
//                   i-th column of Z. 
// int LDZ           Leading dimension of Z; we will set to N.
// double WORK[5*N]  Working space.
// int IWORK[N]      Working space.
// int IFAIL[M]      Normally zero, if any eigenvectors fail to converge, their
//                   indices are stored in IFAIL[].
// int INFO          =0 success. <0 i-th argument invalid. >0 # evecs failed to converge.

   int M=1, LDZ=N, INFO;

   int thread = omp_get_thread_num();
   if (myArrays[thread].IBLOCK == NULL) myArrays[thread].IBLOCK = (int*) calloc(N, sizeof(int));
   if (myArrays[thread].ISPLIT == NULL) myArrays[thread].ISPLIT = (int*) calloc(N, sizeof(int));
   if (myArrays[thread].WORK   == NULL) myArrays[thread].WORK   = (double*) calloc(5*N, sizeof(double));
   if (myArrays[thread].IWORK  == NULL) myArrays[thread].IWORK  = (int*) calloc(N, sizeof(int));
   if (myArrays[thread].IFAIL  == NULL) myArrays[thread].IFAIL  = (int*) calloc(N, sizeof(int));
   myArrays[thread].IBLOCK[0]=1;
   myArrays[thread].ISPLIT[0]=N;
   double W = u;

   if (myArrays[thread].IBLOCK == NULL || myArrays[thread].ISPLIT==NULL || 
       myArrays[thread].WORK==NULL || myArrays[thread].IWORK==NULL || 
       myArrays[thread].IFAIL==NULL) {
      DSTESS_DEBUG(1) fprintf(stderr, "%2i:%s:%i dstein failed to allocate workspaces.\n", omp_get_thread_num(), __func__, __LINE__);
      return(PlasmaErrorOutOfMemory);
   }

   dstein_(&N, diag, offd, &M, &W, myArrays[thread].IBLOCK, myArrays[thread].ISPLIT, v, 
           &LDZ, myArrays[thread].WORK, myArrays[thread].IWORK, myArrays[thread].IFAIL, &INFO);
   DSTESS_DEBUG(0) fprintf(stderr, "%2i:%s:%i ev=%.16e dstein returning INFO=%d.\n", 
                   omp_get_thread_num(), __func__, __LINE__, u, INFO);

   return(INFO);
} // end useDstein.

//-----------------------------------------------------------------------------
// This is the scaled Sturm; based on the classic Sturm.
// See https://archive.siam.org/meetings/la03/proceedings/zhangjy3.pdf
// "J. Zhang, 2003, The Scaled Sturm Sequence Computation".
// Both the Sturm (proportional) and the classical Sturm can suffer from
// underflow and overflow for some problematic matrices; automatic rescaling
// avoids that; and using the classical Sturm avoids division (and checking to
// avoid division by zero). Computation is is still O(N), but about 1.5 times
// more flops.
//
// diag[0..n-1] are the diagonals; offd[0..n-2] are the offdiagonals.
// The classic recurrence: (u is the \lambda cutpoint in question).
// p[-1] = 1.;             // zero relative indexing.
// p[0] = diag[0] - u;
// p[i] = (diag[i]-u)*p[i-1] - offd[i-1]*offd[i-1]*p[i-2], i=1, N-1.
// 
// The Classical Sturm recurrence can be shown as a matrix computation; namely
// P[i] = M[i]*P[i-1]. Be careful of the i-1 index:
// M[i] = [(diag[i]-u) , -offd[i-1]*offd[i-1] ] and P[i-1] = [ p[i-1] ]
//        [          1 ,                    0 ]              [ p[i-2] ]
// with P[-1] defined to be [1, 0] transposed.
// notice 'p' is the classical Sturm, 'P' is a vector.
// 
// the matrix computation results in the vector: 
// M[i]*P[i-1] = { (diag[i]-u)*p[i-1] -offd[i-1]*offd[i-1]*p[i-2] , p[i-1] }
// 
// So, in the classical case, P[i][0] is the classic Sturm sequence for p[i];
// the second element is just the classic Sturm for p[i-1].
//
// However, this won't remain that way. For the SCALED Sturm sequence, we 
// will scale P[i] after each calculation, with the scalar 's': 
//
// *********************************
// P[i] = s * M[i]*P[i-1], i=0, N-1. Note we are scaling a vector here.
// *********************************
//
// For code, we represent P[i-1] as two scalars, [Pm1_0 , Pm1_1].
// The matrix calculation is thus:
// M[i]*P[i-1] = { (diag[i]-u)*Pm1_0 -offd[i-1]*offd[i-1]*Pm1_1 , Pm1_0 }
// or in three equations, adding in the scalar:
// save = s * Pm1_0;
// Pm1_0 = s * ( (diag[i]-u)*Pm1_0 -offd[i-1]*offd[i-1]*Pm1_1 );
// Pm1_1 = save;
 
// Pm1_0 is used like the classical Sturm sequence; meaning we must calculate
// sign changes.
// 
// s is computed given the vector X[] = M[i]*P[i-1] above.
// PHI is set to 10^{10}, UPSILON is set to 10^{-10}. Then:
//    w = max(fabs(X[0]), fabs(X[1])). 
//    if w > PHI then s = PHI/w;
//    else if w < UPSILON then s = UPSILON/w;
//    else s=1.0 (or, do not scale X).
// 
// This algorithm is backward stable. execution time is 1.5 times classic Sturm.
// 
// No sign change counts eigenvalues >= u.
// sign changes count eigenvalues <  u.
// This routine returns the number of sign changes, which is the count of
// eigenvalues less than u.
// 
// computation: What we need for each computation:
// M[i], which we compute on the fly from diag[i] and offd[i-1].
// P[i-1], which has two elements, [Pm1_0, Pm1_1]. (Pm1 means P minus 1).
// LAPACK routine DLAEBZ computes a standard Sturm sequences; there is no 
// comparable auto-scaling Sturm sequence.
//-----------------------------------------------------------------------------
static int Sturm_Scaled(double *diag, double *offd, int n, double u) {
   int i, isneg=0;
   double s, w, v0, v1, Pm1_0, Pm1_1, PHI, UPSILON;
   if (n==0) return (0);
   PHI = ((double)(((long long) 1)<<34));
   UPSILON = 1.0/PHI;

   Pm1_1 = 1.0;
   Pm1_0 = (diag[0]-u);
   if (Pm1_0 < 0) isneg = 1;  // our first test.
   for (i=1; i<n; i++) {
      // first part of scaling, just get w.
      v0 = fabs(Pm1_0);
      v1 = fabs(Pm1_1);
      if (v0 > v1) w = v0;
      else         w = v1;

      // Go ahead and calculate P[i]:
      s = Pm1_0;
      Pm1_0 = (diag[i]-u)*Pm1_0 -((offd[i-1]*offd[i-1])*Pm1_1);
      Pm1_1 = s;

      // Now determine whether to scale these new values.
      if (w > PHI) {
         s = PHI/w;
         Pm1_0 *= s;
         Pm1_1 *= s;
      } else if (w < UPSILON) {
         s = UPSILON/w;
         Pm1_0 *= s;
         Pm1_1 *= s;
      } // else skip scaling.

      // Finally, see if the sign changed.
      if ( (Pm1_0 < 0 && Pm1_1 >= 0) ||  
           (Pm1_0 >= 0 && Pm1_1 < 0)
         ) isneg++;  
   }
            
   return(isneg);
} // end Sturm_Scaled.

//-----------------------------------------------------------------------------
// Bounds on min and max eigenvalues; by Gerschgorin disc. May be over/under 
// estimated.
// By Gerschgorin Circle Theorem;
// All Eigval(A) are \in [\lamda_{min}, \lambda_{max}].
// \lambda_{min} = min (i=0; i<n) diag[i]-|offd[i]| - |offd[i-1]|,
// \lambda_{max} = max (i=0; i<n) diag[i]+|offd[i]| + |offd[i-1]|,
// with offd[-1], offd[n] = 0.
// Indexes above are 0 relative.
// Although Gerschgorin is mentioned in ?larr?.f LAPACK files, it is coded
// inline there. This is a simple min/max boundary for the full range of
// eigenvalues. However, there is no guarantee that \lambda_{min} and
// \lambda_{max} are actually eigenvalues. We add a step to find the actual min
// and max, so we can also use this to compute the matrix Norm. 
//-----------------------------------------------------------------------------
static void Bound_MinMax_Eigvalue(double *diag, double *offd, int n, double *Min, double *Max) {
   int i;
   double test, testdi, testdim1, min=DBL_MAX, max=-DBL_MAX;

   for (i=0; i<n; i++) {
      if (i == 0) testdim1=0.;
      else        testdim1=offd[i-1];

      if (i==(n-1)) testdi=0;
      else          testdi=offd[i];

      test=diag[i] - fabs(testdi) - fabs(testdim1);
      if (test < min) {
         min=test;
         DSTESS_DEBUG(0) fprintf(stderr,"Gerschgorin row=%i new min=%.16e.\n", i, min);
      } 

      test=diag[i] + fabs(testdi) + fabs(testdim1);
      if (test > max) {
         max=test;
         DSTESS_DEBUG(0) fprintf(stderr,"Gerschgorin row=%i new max=%.16e.\n", i, max);
      }      
   }
      

   double cp, minLB=min, minUB=max, maxLB=min, maxUB=max;
   // Now, within that range, find the actual minimum.
   while (1) {
      cp = (minLB+minUB)*0.5;
      if (cp == minLB || cp == minUB) break;
      if (Sturm_Scaled(diag, offd, n, cp) == n) minLB = cp;
      else                                      minUB = cp;
   }
    
   // Now find the max within that range. At each midpoint MidP:
   while (1) {
      cp = (maxLB+maxUB)*0.5;
      if (cp == maxLB || cp == maxUB) break;
      if (Sturm_Scaled(diag, offd, n, cp) == n) maxUB=cp;
      else                                      maxLB=cp;
   }

   *Min = minLB;
   *Max = maxLB;
} // end Bound_MinMax_Eigvalue

//-----------------------------------------------------------------------------
// Matrix multiply; A * X = Y.
// A = [diag[0], offd[0], 
//     [offd[0], diag[1], offd[1]
//     [      0, offd[1], diag[2], offd[2],
//     ...
//     [ 0...0                     offd[n-2], diag[n-1] ]
// There is no LAPACK alternative that does just Y=A*X for a symmetric 
// tridiagonal matrix. DGBMV will do Y=alpha*X +beta*Y, but would require
// creation of a calling matrix, copying, etc, with added overhead to process
// alpha, beta, Y, etc.  
//-----------------------------------------------------------------------------
static void MM(double *diag, double *offd, int n, double *X, double *Y) {
   int i;
   Y[0] = diag[0]*X[0] + offd[0]*X[1];
   Y[n-1] = offd[n-2]*X[n-2] + diag[n-1]*X[n-1];

   for (i=1; i<(n-1); i++) {
      Y[i] = offd[i-1]*X[i-1] + diag[i]*X[i] + offd[i]*X[i+1];
   }
} // END MM.


//-----------------------------------------------------------------------------
// eigenpair error: If A*v = u*v, then A*v-u*v should == 0. We compute the
// L_infinity norm of (A*v-u*v).
// We return DBL_MAX if the eigenvector (v) is all zeros.
// If u==0.0, we'll return L_INF of (A*V). 
//-----------------------------------------------------------------------------
static double eigp_error(double *diag, double *offd, int n, double u, double *v) {
   int i, zeros=0;
   double *AV, test_vector[4096]; // static workplace to avoid calloc.
   double norm, dtemp;

   AV = test_vector;              // assume big enough.
   if (n > 4096) AV = (double*) calloc(n, sizeof(double)); // oops, it isn't.

   MM(diag, offd, n, v, AV); // AV = A*v.

   norm = -DBL_MAX;  // Trying to find maximum.
   zeros=0;
   for (i=0; i<n; i++) {
      dtemp = fabs(AV[i] - u*v[i]);    // This should be zero.
      if (dtemp > norm) norm=dtemp;
      if (v[i] == 0.) zeros++;
   }

   if (AV != test_vector) free(AV);
   if (zeros == n) return(DBL_MAX);
   return(norm);
} // end eigp_error.


//-----------------------------------------------------------------------------
// Thread work.
// We are given the WorkStack; we only exit when eigenvalues == finished.  The
// ToDo chain works like a stack, we pop Brackets to work on.  Each Bracket is
// bisected down to one eigenvalue; right-halves are pushed back on the ToDo
// stack as future work to do.
//
// For range=PlasmaRangeI: We still bisect on range values, but the conditions
// change. We begin with the full range Gerschgorin [low, hi+ulp). We compute
// nLT_low, nLT_hi, but we need the indices il, iu. So these nLT_x values are
// the conditions: The smallest (1 relative) index the bracket can contain is
// nLT_low+1. e.g. if nLT_low = 0, then index 1 may be in the bracket.  if
// nLT_low=7, then index 8 may be in the bracket.
//
// The largest (1 relative) index the bracket can contain is nLT_hi: If
// nLT_hi=n, then the nth eigenvalue might be in the bracket. So the bracket
// contains indices [nLT_low+1, nLT_hi]. How does that intersect [il, iu]?
//
// Low end: If iu < nLT_low+1, then [il,iu] is out of Bracket.  Hi end: If il >
// nLT_hi, then [il,iu] is out of Bracket.  Otherwise there is overlap. We find
// the midpoint, and compute nLT_midp.
//
// If nLT_midp were the new high (for the left bracket), then if il > nLT_midp,
// we can discard the left bracket. 
//
// If nLT_midp were the new low (for the right bracket), then if iu <
// nLT_midp+1, we can discard the right bracket. 
//
// The number of eigenvalues in the bracket is still (nLT_hi - nLT_low). if
// that is ever zero, we can discard the bracket.
//
// Stage: Init: establish nLT_low, nLT_hi, numEV in the bracket.
//
// Stage: Bisection. We use Bisection to divide the range until lowerBound and
// upperBound cannot be averaged. (The result is one or the other).  If we
// subdivide and the two halves both have eigenvalues, we create a new bracket
// and (Critical) add it to the ToDo stack, with Bisection as the workstage.
//
// Once a range can't be subdivided (the UpperBound=LowerBound+ULP); we store
// the lowerBound as the eigenvalue and numEV as the multiplicity in the
// caller's array (pVal, pMul), and enter the next Stage, GetVector.
//
// Stage: GetVector. At this point we know the index of the eigenpair within
// the user's arrays. If PlasmaVec, we find the eigenvector using the LAPACK
// routine DSTEIN, storing it in its proper place in pVec.  Then we add numEV
// to the count of finished eigenpairs, and free the bracket.
//
// The most comparable serial LAPACK routine is DSTEVX.
//
// Once all thread work is complete, the code will condense these arrays to
// just the actual number of unique eigenvalues found.
//-----------------------------------------------------------------------------

void thread_work(WorkStack* Stack) {
   double *diag = Stack->diag;
   double *offd = Stack->offd;
   int    N = Stack->N;

   double cp;
   int flag, evLess;
   EvBracket *myB;

   while (1) {
      flag = 0;
      myB = NULL;
      #pragma omp critical (UpdateStack)
      {
         if (Stack->finished == Stack->eigenvalues) flag=1;
         else if (Stack->ToDo != NULL) {
            myB = Stack->ToDo;
            Stack->ToDo = myB->next;
         }
      }

      // Exit, all the work is done.
      if (flag==1) return;

      // If all the work isn't done but I couldn't find any,
      // go back and look again. Another thread must still 
      // be subdividing or working on a vector.
      if (myB == NULL) continue;

      // Okay, myB is popped off the stack, we must resolve it.
      switch (myB->stage) {
         case Init:
            myB->nLT_low = Sturm_Scaled(diag, offd, N, myB->lowerBound);
            myB->nLT_hi =  Sturm_Scaled(diag, offd, N, myB->upperBound);
            // compute number of eigenvalues in this range. 
            myB->numEV = (myB->nLT_hi - myB->nLT_low);
            DSTESS_DEBUG(0) fprintf(stderr, "%2i:%s:%i On entry, #EV in [%.7f, %.7f]==%d, nLT_low=%d, nLT_hi=%d.\n", omp_get_thread_num(), __func__, __LINE__, myB->lowerBound, myB->upperBound, myB->numEV, myB->nLT_low, myB->nLT_hi);

            // If no eigenvalues in this bracket, we discard it and break from the switch() to continue the while().
            // This happens when ranges are part of the first arbitrary range division.
            if (myB->numEV == 0) {
               free(myB);  
               myB=NULL;
               break;
            }
         
            if (Stack->range == PlasmaRangeI) {
               if (myB->nLT_hi  < Stack->il ||     // e.g if il=500, and nLT_hi=499, this bracket is under range of interest.
                   myB->nLT_low > Stack->iu) {     // e.g if iu=1000, and lLT_low=1001, this bracket is above range of interest.
                   DSTESS_DEBUG(0) fprintf(stderr, "Line:%i, discard myB, nLT_hi=%i, nLT_low=%i, Stack.il=%i, Stack.iu=%i, myB=%p.\n", 
                  __LINE__, myB->nLT_hi, myB->nLT_low, Stack->il, Stack->iu, myB);
                  free(myB);
                  myB=NULL;
                  break;
               }
            }

            myB->stage = Bisection;
            // fall into Bisection.
            
         case Bisection:
            flag = 0;
            while (1) {
               cp = (myB->lowerBound+myB->upperBound)*0.5;
               DSTESS_DEBUG(0) fprintf(stderr, "%2i:%s:%i lowerBound=%.16f, upperBound=%.16f, cp=%.16f, nLT_low=%i, nLT_hi=%i.\n", 
                     omp_get_thread_num(), __func__, __LINE__, myB->lowerBound, myB->upperBound, cp, myB->nLT_low, myB->nLT_hi);
               if (cp == myB->lowerBound || cp == myB->upperBound) {
                  // Our bracket has been narrowed to machine epsilon for this magnitude (=ulp).
                  // We are done; the bracket is always [low,high). 'high' is not included, so
                  // we have myB->numEV eigenvalues at low, whether it == 1 or is > 1.
                  // We find the eigenvector.
                  // (We can test multiplicity with GluedWilk).
                  DSTESS_DEBUG(0) fprintf(stderr, "%2i:%s:%i cutpoint found eigenvalue %.16e nLT_low=%d idx=%d numEV=%i.\n", omp_get_thread_num(), __func__, __LINE__, myB->lowerBound, myB->nLT_low, myB->nLT_low - Stack->baseIdx, myB->numEV);
                  break;
               } else {
                  // we have a cutpoint. 
                  evLess = Sturm_Scaled(diag, offd, N, cp);
                  DSTESS_DEBUG(0) fprintf(stderr, "%2i:%s:%i cp=%.16f, evLess<cp=%i, nLT_low=%i, nLT_hi=%i.\n", 
                     omp_get_thread_num(), __func__, __LINE__, cp, evLess, myB->nLT_low, myB->nLT_hi);
                  if (evLess < 0) {
                     // We could not compute the Sturm sequence for it.
                     flag = -1; // indicate an error.
                     DSTESS_DEBUG(0) fprintf(stderr, "Sturm Sequence compute fails for this matrix.\n");
                     break; // exit while true.
                  }

                  // Discard empty halves in both PlasmaRangeV and PlasmaRangeI.
                  // If #EV < cutpoint is the same as the #EV < high, it means
                  // no EV are in [cutpoint, hi]. We can discard that range.
                  if (evLess == myB->nLT_hi) {
                     myB->upperBound = cp;
                     DSTESS_DEBUG(0) fprintf(stderr, "%2i:%s:%i cp=%.16f Discard high range, now [%.16f, %.16f].\n", omp_get_thread_num(), __func__, __LINE__, cp, myB->lowerBound, myB->upperBound);
                     continue;
                  }

                  // If #EV < cutpoint is the same as #EV < low, it means no
                  // EV are in [low, cutpoint]. We can discard that range. 
                  if (evLess == myB->nLT_low) {
                     myB->lowerBound = cp;
                     DSTESS_DEBUG(0) fprintf(stderr, "%2i:%s:%i cp=%.16f Discard Low range, now [%.16f, %.16f].\n", omp_get_thread_num(), __func__, __LINE__, cp, myB->lowerBound, myB->upperBound);
                     continue;
                  }

                  // Note: If we were PlasmaRangeV, the initial bounds given by the user are the ranges,
                  // so we have nothing further to do. In PlasmaRangeI; the initial bounds are Gerschgorin,
                  // limits and not enough: We must further narrow to the desired indices.
                  if (Stack->range == PlasmaRangeI) {
                        DSTESS_DEBUG(0) fprintf(stderr, "%2i:%s:%i PlasmaRangeI cp=%.16f, evLess=%i, Stack->il=%i, Stack->iu=%i.\n", 
                        omp_get_thread_num(), __func__, __LINE__, cp, evLess, Stack->il, Stack->iu);
                     // For PlasmaRangeI:
                     // Recall that il, iu are 1-relative; while evLess is zero-relative; i.e.
                     // if [il,iu]=[1,2], evless must be 0, or 1. 
                     // when evLess<cp == il-1, or just <il, cp is a good boundary and 
                     // we can discard the lower half.
                     //
                     // To judge the upper half, the cutpoint must be < iu, so if it is >= iu, cannot
                     // contain eigenvalue[iu-1].
                     // if evLess >= iu, we can discard upper half.
                     if (evLess < Stack->il) {
                        // The lower half [lowerBound, cp) is not needed, it has no indices >= il.
                        myB->lowerBound = cp;
                        myB->nLT_low    = evLess;
                        myB->numEV = (myB->nLT_hi-myB->nLT_low);
                        DSTESS_DEBUG(0) fprintf(stderr, "%2i:%s:%i cp=%.16f PlasmaRangeI: Discard Low range, now [%.16f, %.16f] #Eval=%i.\n", omp_get_thread_num(), __func__, __LINE__, cp, myB->lowerBound, myB->upperBound, myB->numEV);
                        continue;
                     }

                     if (evLess >= Stack->iu) {
                        // The upper half [cp, upperBound) is not needed, it has no indices > iu; 
                        myB->upperBound = cp;
                        myB->nLT_hi     = evLess;
                        myB->numEV = (myB->nLT_hi-myB->nLT_low);
                        DSTESS_DEBUG(0) fprintf(stderr, "%2i:%s:%i cp=%.16f PlasmaRangeI: Discard High range, now [%.16f, %.16f] #Eval=%i.\n", omp_get_thread_num(), __func__, __LINE__, cp, myB->lowerBound, myB->upperBound, myB->numEV);
                        continue;
                     }
                  } // end if index search.

                  // Here, the cutpoint has some valid EV on the left and some on the right.
                  DSTESS_DEBUG(0) fprintf(stderr, "%2i:%s:%i splitting Bracket [%.16f,%.16f,%.16f], nLT_low=%i,nLT_cp=%i,nLT_hi=%i\n", 
                         omp_get_thread_num(), __func__, __LINE__, myB->lowerBound, cp, myB->upperBound, myB->nLT_low, evLess, myB->nLT_hi);
                  EvBracket* newBracket = (EvBracket*) calloc(1, sizeof(EvBracket)); 
                  memcpy(newBracket, myB, sizeof(EvBracket));
                  // the right side: Low is cp; Hi stays the same; stage is still Bisection.
                  newBracket->lowerBound = cp;
                  newBracket->nLT_low = evLess;
                  newBracket->numEV = (myB->nLT_hi - evLess);
                  #pragma omp critical (UpdateStack)
                  {
                     // make new Bracket head of the ToDo work,
                     newBracket->next = Stack->ToDo;
                     Stack->ToDo = newBracket;
                  }

                  // Update the Bracket I kept.                  
                  myB->upperBound = cp;
                  myB->nLT_hi = evLess;
                  myB->numEV =( evLess - myB->nLT_low); 
                  continue; 
               }
            } // end while(true) for Bisection. 

            // When we are done Bisecting, we have an Eigenvalue; proceed to GetVector.
            DSTESS_DEBUG(0) fprintf(stderr, "%2i:%s:%i Exit while(true), found eigenvalue %.16e nLT_low=%d idx=%d numEV=%i.\n", omp_get_thread_num(), __func__, __LINE__, myB->lowerBound, myB->nLT_low, myB->nLT_low - Stack->baseIdx, myB->numEV);
            myB->stage=GetVector;

         case GetVector:
            DSTESS_DEBUG(0) fprintf(stderr, "%2i:%s:%i Getvector ev=%.16e idx=%d\n", omp_get_thread_num(), __func__, __LINE__, myB->lowerBound, myB->nLT_low-Stack->baseIdx);
            // Okay, count this eigenpair done, add to the Done list.
            // NOTE: myB->nLT_low is the global zero-relative index
            //       of this set of mpcity eigenvalues.
            //       No other brackets can change our entry.
            int myIdx;
            if (Stack->range == PlasmaRangeI) {
               myIdx = myB->nLT_low - (Stack->il-1);
            } else { // range == PlasmaRangeV
               myIdx = myB->nLT_low - Stack->baseIdx;
            }

            if (Stack->eigt == PlasmaVec) {

               // get the eigenvector.
               int ret=useDstein(diag, offd, myB->lowerBound, &(Stack->pVec[myIdx*N]), N, Stack->stein_arrays);
               if (ret != 0) {
                  #pragma omp critical (UpdateStack)
                  {
                     if (Stack->error != 0) Stack->error = ret;
                  }
               }
            }

            // Add eigenvalue and multiplicity.
            Stack->pVal[myIdx]=myB->lowerBound;
            Stack->pMul[myIdx]=myB->numEV;

            DSTESS_DEBUG(0) fprintf(stderr, "%2i:%s:%i Success adding eigenvector #%d myIdx=%d of %d, value %.16f, mpcity=%d\n", omp_get_thread_num(), __func__, __LINE__, myB->nLT_low, myIdx, Stack->eigenvalues, myB->lowerBound, myB->numEV);
            #pragma omp atomic 
               Stack->finished += myB->numEV;
            
            // Done with this bracket.
            free(myB);
            break;
      } // End switch on stage.
   } // end Master Loop.

   DSTESS_DEBUG(0) fprintf(stderr, "%2i:%s:%i Exiting thread work.\n", omp_get_thread_num(), __func__, __LINE__);
} // end thread_work.


//-----------------------------------------------------------------------------
// This is the main routine; plasma_dstess. 
//-----------------------------------------------------------------------------
int plasma_dstess(plasma_enum_t eigt, plasma_enum_t range, // args 1,2
                  int n, int k,                            // args 3,4
                  double *diag, double *offd,              // args 5,6
                  double vl, double vu,                    // args 7,8
                  int il, int iu,                          // args 9,10,
                  int *pFound,                             // arg 11
                  double *pVal,                            // arg 12
                  int    *pMul,                            // arg 13
                  double *pVec)                            // arg 14
{
    int i, max_threads;
    Stein_Array_t *stein_arrays = NULL;
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    // Check input arguments 
    if (eigt != PlasmaVec && eigt != PlasmaNoVec && eigt != PlasmaCount) {
        plasma_error("illegal value of eigt");
        return -1;
    }
    if (range != PlasmaRangeV &&
        range != PlasmaRangeI ) {
        plasma_error("illegal value of range");
        return -2;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -3;
    }

    // arg 4: Any value of 'k' is legal on entry, we check it later.

    if (diag == NULL) {
        plasma_error("illegal pointer diag");
        return -5;
    }
    if (offd == NULL) {
        plasma_error("illegal pointer offd");
        return -6;
    }
    
    // Check args 7, 8.
    if (range == PlasmaRangeV && vu <= vl ) {
        plasma_error("illegal value of vl and vu");
        return -7;
    }

    // check args 9, 10.
    if (range == PlasmaRangeI) {
        if (il < 1 || il > imax(1,n)) {
             plasma_error("illegal value of il");
             return -9;
        } else if (iu < imin(n,il) || iu > n) {
            plasma_error("illegal value of iu");
            return -10;
        }
    }

    if (pFound == NULL) return -11;

    // Quick return 
    if (n == 0) {
        pFound[0]=0;
        return PlasmaSuccess;
    }

    DSTESS_DEBUG(0) omp_set_num_threads(1); // For single-thread testing.
    max_threads = omp_get_max_threads();

    if (eigt == PlasmaVec) { 
        stein_arrays = (Stein_Array_t*) calloc(max_threads, sizeof(Stein_Array_t));
        if (stein_arrays == NULL) {
            DSTESS_DEBUG(0) fprintf(stderr, "%s:%i check.\n", __func__, __LINE__);
            return PlasmaErrorOutOfMemory;
        }
    }
        
    // Initialize sequence.
    plasma_sequence_t sequence;
    plasma_sequence_init(&sequence);

    // Initialize request.
    plasma_request_t request;
    plasma_request_init(&request);

    double globMinEval, globMaxEval; 
    double tv_start;
    DSTESS_DEBUG(0) tv_start = omp_get_wtime();

    WorkStack workStack;
    memset(&workStack, 0, sizeof(WorkStack)); 
    workStack.N = n;
    workStack.diag = diag;
    workStack.offd = offd;
    workStack.eigt = eigt;
    workStack.range = range;
    workStack.il = il;
    workStack.iu = iu;
    workStack.stein_arrays = stein_arrays;

    // Find actual min and max eigenvalues.
    Bound_MinMax_Eigvalue(workStack.diag, workStack.offd, workStack.N, &globMinEval, &globMaxEval);

    int evLessThanVL=0, evLessThanVU=n, nEigVals=0;
    if (range == PlasmaRangeV) {
        // We don't call Sturm if we already know the answer. 
        if (vl >= globMinEval) evLessThanVL=Sturm_Scaled(diag, offd, n, vl);
        if (vu <= globMaxEval) evLessThanVU=Sturm_Scaled(diag, offd, n, vu);
        // Compute the number of eigenvalues in [vl, vu).
        nEigVals = (evLessThanVU - evLessThanVL);
         workStack.baseIdx = evLessThanVL;
    } else {
        // PlasmaRangeI: iu, il already vetted by code above.
        nEigVals = iu+1-il; // The range is inclusive.
        // We still bisect by values to discover eigenvalues, though. 
        vl = globMinEval;
        vu = nexttoward(globMaxEval, DBL_MAX); // be sure to include globMaxVal.
        workStack.baseIdx = 0; // There are zero eigenvalues less than vl.
    }

    // if we just need to find the count of eigenvalues in a value range,
    if (eigt == PlasmaCount) {
        pFound[0] = nEigVals;
        return(PlasmaSuccess);
    }

    // Now if user's K (arg 4) isn't enough room, we have a problem.
    if (k < nEigVals) {
        return -4;             // problem with user's K value.
    }   

    // We are going into discovery. Make sure we have arrays.
    if (pVal == NULL) return -12;   // pointers cannot be null.
    if (pMul == NULL) return -13;   // ...
    if (eigt == PlasmaVec && pVec == NULL) return -14;   // If to be used, cannot be NULL.

    // handle value range.
    // Set up workStack controls.
    workStack.eigenvalues = nEigVals;
    workStack.finished = 0;
    workStack.pVal = pVal;
    workStack.pMul = pMul;
    workStack.pVec = pVec;
    
    // Create a bracket per processor to kick off the work stack.
    // here, low and high are the range values.
    // We need low+max_threads*step = hi-step. ==>
    // (hi-low) = step*(max_threads+1) ==>
    // step = (hi-low)/(max_threads+1).
    double step = (vu - vl)/(max_threads+1);
    double prevUpper = vl;
    for (i=0; i<max_threads; i++) {
        EvBracket *thisBracket = (EvBracket*) calloc(1, sizeof(EvBracket));
        thisBracket->stage = Init;
        thisBracket->lowerBound = prevUpper;
        if (i == max_threads-1) thisBracket->upperBound = vu;
        else                    thisBracket->upperBound = thisBracket->lowerBound + step;
        prevUpper = thisBracket->upperBound; // don't rely on arithmetic for final value.
        thisBracket->nLT_low = -1;
        thisBracket->nLT_hi  = -1;
        thisBracket->numEV   = -1;
    
        // Now add to the workstack. OMP not active yet.             
        thisBracket->next = workStack.ToDo;
        workStack.ToDo = thisBracket;
    }

    // We can launch the threads.
    #pragma omp parallel proc_bind(close) // requires gcc >=4.9.
    {
       thread_work(&workStack);
    }
 
    double tv_epair, epair_us;
    DSTESS_DEBUG(0) {
        tv_epair = omp_get_wtime();
        epair_us = (tv_epair-tv_start)*1.e6; 
        fprintf(stderr, "%s:%i Checkpoint. epair_us=%.3f.\n", __func__, __LINE__, epair_us);
    }

    // Now, all the eigenvalues should have unit eigenvectors in the array  workStack.Done.
    // We don't need to sort that, but we do want to compress it; in case of multiplicity.
    // We compute the final number of eigenvectors in vectorsFound, and mpcity is recorded.
    int vectorsFound = 0;
    for (i=0; i<workStack.eigenvalues; i++) {
        if (pMul[i] > 0) {
            vectorsFound++;
            DSTESS_DEBUG(0) fprintf(stderr, "Done[%d] mpcity=%d value=%.18e.\n", i, pMul[i], pVal[i]);
        }
    }

    DSTESS_DEBUG(0) fprintf(stderr, "%s:%i Checkpoint. vectorsFound=%d.\n", __func__, __LINE__, vectorsFound);
    // record for user.
    pFound[0] = vectorsFound;

    // compress the array in case vectorsFound < nEigVals (due to multiplicities).
    // Note that pMul[] is initialized to zeros, if still zero, a multiplicity entry.
    if (vectorsFound < workStack.eigenvalues) {
        int j=0;   
        for (i=0; i<workStack.eigenvalues; i++) {
            if (pMul[i] > 0) {
                pMul[j] = pMul[i];
                pVal[j] = pVal[i];
                if (pMul[j] != 1) DSTESS_DEBUG(0) fprintf(stderr, "eigenvalue[%i]=%.16e, mpcity=%i.\n", j, pVal[j], pMul[j]);
                if (workStack.eigt == PlasmaVec) {
                    if (j != i) {
                        memcpy(&pVec[j*workStack.N], &pVec[i*workStack.N], workStack.N*sizeof(double));
                    }
                }

                j++;
            }
        }
    } // end if compression is needed.

    double orth_us;
    double start_orth;
    DSTESS_DEBUG(0) start_orth = omp_get_wtime();

    int ret = useDlatsqr(pVec, workStack.N, vectorsFound);
    if (ret != 0) return(ret);
    DSTESS_DEBUG(0) orth_us = (omp_get_wtime() - start_orth)*1.e6;

    int swaps=0;
    if (1 && eigt == PlasmaVec) {  // Whether or not we want swapping applied.
        int N = workStack.N; 
        double *Y = calloc(N, sizeof(double));
        double *test = calloc(4, sizeof(double));
        for (i=0; i<vectorsFound-1; i++) {
            if (fabs(pVal[i+1]-pVal[i]) > 1.E-11) continue;
            // We've tried to parallelize the following four tests
            // as four omp tasks. It works, but takes an average of
            // 8% longer (~3.6 ms) than just serial execution. 
            // omp schedule and taskwait verhead, I presume.
            test[0]= eigp_error(workStack.diag, workStack.offd, N,
                    pVal[i], &pVec[i*N]);
            test[1] = eigp_error(workStack.diag, workStack.offd, N,
                    pVal[i+1], &pVec[(i+1)*N]);
            
            test[2] = eigp_error(workStack.diag, workStack.offd, N,
                    pVal[i], &pVec[(i+1)*N]);
            test[3] = eigp_error(workStack.diag, workStack.offd, N,
                    pVal[i+1], &pVec[i*N]);
            
            if ( (test[2] < test[0])         // val1 with vec2 beats val1 with vec1
              && (test[3] < test[1]) ) {     // val2 with vec1 beats val2 with vec2
                DSTESS_DEBUG(0) fprintf(stderr, "%s:%i Swapping vectors for %d and %d; eigenvalue diff=%.16e.\n", __func__, __LINE__, i, i+1, pVal[i+1]-pVal[i] );
                memcpy(Y, &pVec[i*N], N*sizeof(double));
                memcpy(&pVec[i*N], &pVec[(i+1)*N], N*sizeof(double));
                memcpy(&pVec[(i+1)*N], Y, N*sizeof(double));
                swaps++;
            }
        } // end swapping.

        if (test) free(test);
        if (Y) free(Y);
    } // end if we want to swap at all.

    // Free all the blocks that got used.
    for (i=0; i<max_threads; i++) {
       if (stein_arrays[i].IBLOCK) free(stein_arrays[i].IBLOCK);
       if (stein_arrays[i].ISPLIT) free(stein_arrays[i].ISPLIT);
       if (stein_arrays[i].WORK  ) free(stein_arrays[i].WORK  );
       if (stein_arrays[i].IWORK ) free(stein_arrays[i].IWORK );
       if (stein_arrays[i].IFAIL ) free(stein_arrays[i].IFAIL );
    }

    if (stein_arrays) free(stein_arrays);

    // Return status.
    DSTESS_DEBUG(0) fprintf(stderr, "plasma_dstess exit-return %d.\n", sequence.status);
    return sequence.status;
} // END plasma_dstess

