/**
 *
 * @file
 *
 *  PLASMA header.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of Manchester, Univ. of California Berkeley and
 *  Univ. of Colorado Denver.
 *
 * @precisions normal z -> s d c
 *
 **/
#ifndef PLASMA_ZLAEBZ2_H
#define PLASMA_ZLAEBZ2_H
/******************************************************************************* 
 * These structures support the ZLAEBZ2 code and ZSTEVX2 code, for eigenvalue 
 * and eigenvector discovery.
*******************************************************************************/

/******************************************************************************* 
 * zstein needs work areas to function. Instead of allocating and deallocating
 * these work areas for every vector, we provide a set of work areas per thread.
 * They are allocated as needed; so we don't allocate more often than we need,
 * and only allocate at most once per thread and not once per eigenvector.
*******************************************************************************/

typedef struct
{
    int     *IBLOCK;
    int     *ISPLIT;
    plasma_complex64_t  *WORK;
    int     *IWORK;
    int     *IFAIL;
} zlaebz2_Stein_Array_t;

/******************************************************************************* 
 * Control is all the global variables needed. 
*******************************************************************************/

typedef struct
{
    int     N;
    plasma_complex64_t  *diag;  /* pointers the threads need.                   */
    plasma_complex64_t  *offd;
    plasma_enum_t range;        /* PlasmaRangeV or PlasmaRangeI.                */
    plasma_enum_t jobtype;      /* PlasmaNoVec, PlasmaVec, PlasmaCount          */
    int     il;                 /* For PlasmaRangeI, least index desired.       */
    int     iu;                 /* For PlasmaRangeI, max index desired.         */
    zlaebz2_Stein_Array_t* stein_arrays;  /* Workspaces per thread for useStein.*/
    int     baseIdx;            /* Number of EV less than user's low threshold. */
    int     error;              /* first error, if non-zero.                    */
    plasma_complex64_t  *pVal;  /* where to store eigenvalues.                  */
    plasma_complex64_t  *pVec;  /* where to store eigenvectors.                 */
    int                 *pMul;  /* where to store Multiplicity.                 */
} zlaebz2_Control_t;

#endif /* PLASMA_ZLAEBZ2_H */
