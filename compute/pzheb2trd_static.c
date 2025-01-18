/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> s d c
 *
 **/

#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_types.h"
#include "plasma_workspace.h"
#include "plasma_core_blas.h"
#include "bulge.h"

#include <omp.h>
#include <sched.h>
#include <string.h>

#undef REAL
#define COMPLEX

/***************************************************************************//**
 *  Static scheduler
 **/

#define shift 3

#define ss_cond_set(m, n, val) \
    { \
        plasma->ss_progress[(m)+plasma->ss_ld*(n)] = (val); \
    }

#define ss_cond_wait(m, n, val) \
    { \
        while (plasma->ss_progress[(m)+plasma->ss_ld*(n)] != (val)) \
            sched_yield(); \
    }


//  Parallel bulge chasing column-wise, static scheduling
void plasma_pzheb2trd_static(
    plasma_enum_t uplo, int n, int nb, int Vblksiz,
    plasma_complex64_t *A, int lda,
    plasma_complex64_t *V, plasma_complex64_t *tau,
    double *D, double *E, int wantz,
    plasma_workspace_t work,
    plasma_sequence_t *sequence, plasma_request_t *request)
{

    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return;
    }

    // Check sequence status.
    if (sequence->status != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorSequence);
        return;
    }

    if (uplo != PlasmaLower) {
        plasma_request_fail(sequence, request, PlasmaErrorNotSupported);
        return;
    }

    // Quick return
    if (n == 0) {
        return;
    }

    // General case:
    //
    // As I store V in the V vector there are overlap between
    // tasks so shift is now 4 where group need to be always
    // multiple of 2 (or shift = 5 if not multiple of 2),
    // because as example if grs = 1 task 2 from
    // sweep 2 can run with task 6 sweep 1., but task 2 sweep 2
    // will overwrite the V of tasks 5 sweep 1 which are used by
    // task 6, so keep in mind that group need to be multiple of 2,
    // and thus tasks 2 sweep 2 will never run with task 6 sweep 1.
    // OR if we allocate V as V(n,2) and we switch between the storing of
    // sweep's like odd in V(n,1) and even in V(n,2) then no overlap and so
    // shift is 3.
    // when storing V in matrix style, shift could be back to 3.

    // Some tunning for the bulge chasing code;
    // see technical report for details.
    int nbtiles = plasma_ceildiv(n,nb);
    int colblktile = 1;
    int grsiz = 1;
    int maxrequiredcores = imax( nbtiles/colblktile, 1 );
    int colpercore = colblktile*nb;
    int thgrsiz = n;


    // Initialize static scheduler progress table.
    int cores_num;
    #pragma omp parallel
    {
        cores_num  = omp_get_num_threads();
    }
    int size = 2*nbtiles + shift + cores_num + 10;
    plasma->ss_progress = (volatile int *)malloc(size*sizeof(int));
    for (int index = 0; index < size; ++index)
        plasma->ss_progress[index] = 0;
    plasma->ss_ld = (size);

    // main bulge chasing code
    int ii = shift/grsiz;
    int stepercol = ii*grsiz == shift ? ii : ii + 1;
    ii = (n - 1)/thgrsiz;
    int thgrnb = ii*thgrsiz == (n - 1) ? ii : ii + 1;
    int allcoresnb = imin( cores_num, maxrequiredcores );

    #pragma omp parallel
    {
        int coreid, sweepid, myid, stt, st, ed, stind, edind;
        int blklastind, colpt,  thgrid, thed;
        int i,j,m,k;

        int my_core_id = omp_get_thread_num();
        plasma_complex64_t *my_work = work.spaces[my_core_id];

        for (thgrid = 1; thgrid <= thgrnb; ++thgrid) {
            stt  = (thgrid - 1)*thgrsiz + 1;
            thed = imin( stt + thgrsiz - 1, n - 1 );
            for (i = stt; i <= n - 1; ++i) {
                ed = imin(i,thed);
                if (stt > ed)
                    break;
                for (m = 1; m <= stepercol; ++m) {
                    st = stt;
                    for (sweepid = st; sweepid <= ed; ++sweepid) {
                        for (k = 1; k <= grsiz; ++k) {
                            myid = (i - sweepid)*(stepercol*grsiz) + (m - 1)*grsiz + k;
                            if (myid % 2 == 0) {
                                colpt = (myid/2)*nb + 1 + sweepid - 1;
                                stind = colpt - nb + 1;
                                edind = imin(colpt,n);
                                blklastind = colpt;
                            }
                            else {
                                colpt = ((myid + 1)/2)*nb + 1 + sweepid - 1;
                                stind = colpt - nb + 1;
                                edind = imin(colpt,n);
                                if ((stind >= edind - 1) && (edind == n))
                                    blklastind = n;
                                else
                                    blklastind = 0;
                            }
                            coreid = (stind / colpercore) % allcoresnb;

                            if (my_core_id == coreid) {
                                if (myid == 1) {
                                    ss_cond_wait(myid + shift - 1, 0, sweepid - 1);
                                    plasma_core_zhbtype1cb(
                                        n, nb, A, lda, V, tau,
                                        stind - 1, edind - 1, sweepid - 1,
                                        Vblksiz, wantz, my_work);
                                    ss_cond_set(myid, 0, sweepid);

                                    if (blklastind >= (n - 1)) {
                                        for (j = 1; j <= shift; ++j)
                                            ss_cond_set(myid + j, 0, sweepid);
                                    }
                                }
                                else {
                                    ss_cond_wait(myid - 1,       0, sweepid);
                                    ss_cond_wait(myid + shift - 1, 0, sweepid - 1);
                                    if (myid%2 == 0) {
                                        plasma_core_zhbtype2cb(
                                            n, nb, A, lda, V, tau,
                                            stind - 1, edind - 1, sweepid - 1,
                                            Vblksiz, wantz, my_work);
                                    }
                                    else {
                                        plasma_core_zhbtype3cb(
                                            n, nb, A, lda, V, tau,
                                            stind - 1, edind - 1, sweepid - 1,
                                            Vblksiz, wantz, my_work);
                                    }

                                    ss_cond_set(myid, 0, sweepid);
                                    if (blklastind >= (n - 1)) {
                                        for (j = 1; j <= shift + allcoresnb; ++j)
                                            ss_cond_set(myid + j, 0, sweepid);
                                    }
                                } // end if myid == 1
                            } // end if my_core_id == coreid

                            if (blklastind >= (n - 1)) {
                                ++stt;
                                break;
                            }
                        } // end for k = 1:grsiz
                    } // end for sweepid = st:ed
                } // end for m = 1:stepercol
            } // end for i = 1:n - 1
         } // end for thgrid = 1:thgrnb
    }
    // finalize static sched
    free((void*)plasma->ss_progress);

    //================================================
    // Make the resulting diagonal and super-diagonal elements real,
    // storing them in D and E, respectively.
    //================================================*
    // In complex case, the off diagonal element are
    // not necessary real. We have to make off-diagonal
    // elements real and copy them to E.
    // When using Householder elimination,
    // the ZLARFG give us a real as output, so all the
    // diagonal/off-diagonal element except the last one are already
    // real and thus we need only to take the abs of the last one.
    // @todo Where is abs?
    //
    // Sequential code here so only core 0 will work.
    if (uplo == PlasmaLower) {
        for (int i = 0; i < n - 1; ++i) {
            D[i] = creal(A[i*lda]);
            E[i] = creal(A[i*lda + 1]);
        }
        D[n - 1] = creal(A[(n - 1)*lda]);
    }
    else { // PlasmaUpper not yet tested
        for (int i = 0; i < n - 1; ++i) {
            D[i] = creal(A[i*lda + nb]);
            E[i] = creal(A[i*lda + nb - 1]);
        }
        D[n - 1] = creal(A[(n - 1)*lda + nb]);
    }
}
