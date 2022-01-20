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
 * This is a bracket for a range; any children are subdivisions of the range.
 * We give each thread a bracket. The thread bisects the bracket until it has
 * one eigenvalue, throwing off children in the chain as it does. Empty
 * divisions are discarded; they won't be in the chain; e.g. if I divide the
 * range in half and one half contains no eigenvalues, then we just update
 * the existing bracket appropriately and divide again.
*******************************************************************************/

typedef struct
{
    int    stage;       /* stage of operations on this bracket. */
    
    plasma_complex64_t lowerBound;
    plasma_complex64_t upperBound;
    int    nLT_low;     /* # < lowerBound. -1 if it needs to be found.  */
    int    nLT_hi;      /* # < upperBound. -1 if it needs to be found.  */
    int    numEV;       /* number of Eigenvalues in bracket.            */
    void   *next;       /* A bracket subdivides to more brackets.       */
} zlaebz2_EV_Bracket_t;

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
 * The work stack is the global repository or work to do. It is either a range 
 * containing multiple eigenvalues that must be subdivided (into multiple units 
 * of work) or a final eigenvalue that needs an eigenvector, and recording into
 * the final return arrays. When the work stack is empty, discovery is done.
*******************************************************************************/

typedef struct
{
    int     baseIdx;            /* Number of EV less than user's low threshold. */
    plasma_enum_t range;        /* PlasmaRangeV or PlasmaRangeI.                */
    plasma_enum_t jobtype;      /* PlasmaNoVec, PlasmaVec, PlasmaCount          */
    int     il, iu;             /* For PlasmaRangeI.                            */
    int     eigenvalues;        /* total number of eigenvalues to find.         */
    int     finished;           /* # of finished eigenvectors.                  */
    plasma_complex64_t  *diag;  /* pointers the threads need.                   */
    plasma_complex64_t  *offd;
    int     N;
    zlaebz2_EV_Bracket_t* ToDo; /* NULL or chain head of EVBrackets yet to do.  */
    zlaebz2_Stein_Array_t* stein_arrays;  /* Workspaces per thread for useStein.*/
    int     error;              /* first error, if non-zero.                    */
    plasma_complex64_t  *pVal;  /* where to store eigenvalues.                  */
    int     *pMul;              /* where to store Multiplicity.                 */
    plasma_complex64_t  *pVec;  /* where to store eigenvectors.                 */
} zlaebz2_WorkStack_t;

#endif /* PLASMA_ZLAEBZ2_H */
