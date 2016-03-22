/**
 *
 * @file pzoocm2ccrb.c
 *
 *  PLASMA computational routine.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver.
 *
 * @version 3.0.0
 * @author Jakub Kurzak
 * @date 2016-01-01
 * @precisions normal z -> s d c
 *
 **/

#define AF77(m, n) &(Af77[ ((int64_t)A.nb*(int64_t)lda*(int64_t)(n)) + (int64_t)(A.mb*(m)) ])
#define ABDL(m, n) BLKADDR(A, PLASMA_Complex64_t, m, n)

/******************************************************************************/
void plasma_pzoocm2ccrb(PLASMA_Complex64_t *Af77, int lda, PLASMA_desc A,
                        PLASMA_sequence *sequence, PLASMA_request *request)
{
    PLASMA_Complex64_t *f77;
    PLASMA_Complex64_t *bdl;

    int x1, y1;
    int x2, y2;
    int n, m, ldt;

    if (sequence->status != PLASMA_SUCCESS)
        return;

    for (m = 0; m < A.mt; m++) {

        ldt = BLKLDD(A, m);
        for (n = 0; n < A.nt; n++) {

            x1 = n == 0 ? A.j%A.nb : 0;
            y1 = m == 0 ? A.i%A.mb : 0;
            x2 = n == A.nt-1 ? (A.j+A.n-1)%A.nb+1 : A.nb;
            y2 = m == A.mt-1 ? (A.i+A.m-1)%A.mb+1 : A.mb;

            f77 = AF77(m, n);
            bdl = ABDL(m, n);

            CORE_OMP_zlacpy(PlasmaUpperLower,
                            y2-y1, x2-x1, A.mb,
                            &(f77[x1*lda+y1]), lda,
                            &(bdl[x1*lda+y1]), ldt);
        }
    }
}
