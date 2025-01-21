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
#include <stdbool.h>

#undef REAL
#define COMPLEX

//------------------------------------------------------------------------------
/// Static scheduling condition wait.
///
/// Atomically set progress[ i ] = val, indicating sweep val, task i can start.
///
static inline void ss_cond_set_( plasma_context_t* plasma, int i, int val )
{
    #pragma omp atomic write
    plasma->ss_progress[ i ] = val;
}

#define ss_cond_set( i, val ) \
        ss_cond_set_( plasma, (i), (val) )

//------------------------------------------------------------------------------
/// Static scheduling read value.
///
/// Atomically read progress[ i ].
///
static inline int ss_cond_read_( plasma_context_t* plasma, int i )
{
    int val;
    #pragma omp atomic read
    val = plasma->ss_progress[ i ];
    return val;
}

//------------------------------------------------------------------------------
/// Static scheduling condition wait.
///
/// Atomically busy-wait until progress[ i ] == val,
/// indicating sweep val, task i can start.
///
/// There was no performance difference between this safe version using
/// atomics and the unsafe version using volatile variables.
/// This uses sched_yield(). It works without sched_yield(), but bulge
/// chasing is slower.
///
static inline void ss_cond_wait_( plasma_context_t* plasma, int i, int val )
{
    while (ss_cond_read_( plasma, i ) != val) {
        sched_yield();
    }
}

#define ss_cond_wait( i, val ) \
        ss_cond_wait_( plasma, (i), (val) )


//------------------------------------------------------------------------------
/// Print progress table.
///
static void print_progress(
    char *buf, int len, plasma_context_t *plasma, int progress_size,
    int tid, int sweep, int task, int wait, int wait2 )
{
    int p;
    int ind = snprintf(
        buf, len, "tid %d, sweep %2d, task %2d, wait %2d, %2d, progress = [",
        tid, sweep, task, wait, wait2 );
    for (int k = 0; k < progress_size; ++k) {
        #pragma omp atomic read
        p = plasma->ss_progress[ k ];
        ind += snprintf( &buf[ ind ], len - ind, " %2d", p );
    }
    printf( "%s ]\n", buf );
}

//------------------------------------------------------------------------------
/// Parallel Hermitian band bulge chasing, column-wise, static scheduling.
///
void plasma_pzhbtrd_static(
    plasma_enum_t uplo, int n, int nb, int Vblksiz,
    plasma_complex64_t *A, int lda,
    plasma_complex64_t *V, plasma_complex64_t *tau,
    double *D, double *E, int wantz,
    plasma_workspace_t work,
    plasma_sequence_t *sequence, plasma_request_t *request)
{
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error( "PLASMA not initialized" );
        return;
    }

    // Check sequence status.
    if (sequence->status != PlasmaSuccess) {
        plasma_request_fail( sequence, request, PlasmaErrorSequence );
        return;
    }

    if (uplo != PlasmaLower) {
        plasma_request_fail( sequence, request, PlasmaErrorNotSupported );
        return;
    }

    // Quick return
    if (n == 0) {
        return;
    }

    int num_tiles = plasma_ceildiv( n, nb );

    int num_threads;
    #pragma omp parallel
    #pragma omp single
    {
        num_threads = omp_get_num_threads();
    }
    num_threads = imin( num_threads, num_tiles );

    // Initialize static scheduler progress table.
    // There are at most 2*num_tiles tasks in a sweep, and each task
    // checks task + shift - 1.
    // (It seems 2*nt + shift - 2 works and is tight.)
    const int shift = 3;
    int progress_size = 2*num_tiles + shift - 1;
    plasma->ss_progress = (volatile int*) malloc( progress_size*sizeof( int ) );
    for (int t = 0; t < progress_size; ++t) {
        ss_cond_set( t, 0 );
    }

    // main bulge chasing code
    #pragma omp parallel num_threads( num_threads ) \
            default( none ) \
            shared( plasma ) \
            firstprivate( A, lda, n, nb, num_threads, shift, tau, uplo, \
                          V, Vblksiz, wantz, work, progress_size )
    {
        int tid = omp_get_thread_num();
        plasma_complex64_t *my_work = work.spaces[ tid ];

        // Each sweep brings one column, A(:,sweep), to tridiagonal and
        // chases the resulting bulge to the end of the matrix. There
        // are n-1 sweeps. Sweeps are in parallel, and
        // tasks in consecutive sweeps must be 3 apart (shift = 3). That is,
        // task t in sweep s can start when task t+2 in sweep s-1 is finished.
        //
        // Sweeps are divided into sets of 3 tasks. i = 0, ..., n-2
        // iterates over the sets, with
        // set = (i - sweep) and task = (i - sweep)*3 + k. Thus:
        //
        // i = 0 is set 0 of sweep 0  (i - sweep == 0)
        //
        // i = 1 is set 1 of sweep 0  (i - sweep == 1)
        //      and set 0 of sweep 1  (i - sweep == 0),
        //
        // i = 2 is set 2 of sweep 0  (i - sweep == 2)
        //      and set 1 of sweep 1  (i - sweep == 1)
        //      and set 0 of sweep 2  (i - sweep == 0),
        //
        // and so on. k = 0, 1, 2 iterates over the 3 tasks in each set.
        // For index i, there are up to i+1 sweeps, but as sweeps reach
        // the bottom of the matrix, sweep_begin is incremented,
        // reducing the number of sweeps for subsequent i and k iterations.

        int sweep_begin = 0;
        for (int i = 0; i < n - 1; ++i) {
            for (int k = 0; k < shift; ++k) {
                for (int sweep = sweep_begin; sweep <= i; ++sweep) {
                    // task is number within the sweep   (0-based).
                    // j_first is first column to update (0-based).
                    // j_last  is last  column to update (0-based), inclusive.
                    // Task type 1 brings column (j_first - 1) to tridiagonal,
                    // then updates columns [j_first, .., j_last].
                    int task = (i - sweep)*shift + k;
                    int type = (task == 0 ? 1 : 3 - task % 2);
                    int j_first = (task/2)*nb + sweep + 1;
                    int j_last  = imin( j_first + nb - 1, n - 1 );
                    int task_tid = (j_first / nb) % num_threads;

                    bool sweep_done = (j_last >= n - 1)
                                   || (type == 2 && j_last >= n - 2);

                    if (tid == task_tid) {
                        if (task != 0) {
                            // Wait on previous task in this sweep.
                            ss_cond_wait( task - 1, sweep + 1 );
                        }

                        // Wait on (task + 2) in previous sweep, e.g.,
                        // 1st task waits for 3rd task of prev sweep to finish.
                        ss_cond_wait( task + shift - 1, sweep );

                        if (type == 1) {
                            plasma_core_zhbtype1cb(
                                n, nb, A, lda, V, tau,
                                j_first, j_last, sweep,
                                Vblksiz, wantz, my_work);
                        }
                        else if (type == 2) {
                            plasma_core_zhbtype2cb(
                                n, nb, A, lda, V, tau,
                                j_first, j_last, sweep,
                                Vblksiz, wantz, my_work);
                        }
                        else {
                            plasma_core_zhbtype3cb(
                                n, nb, A, lda, V, tau,
                                j_first, j_last, sweep,
                                Vblksiz, wantz, my_work);
                        }

                        // Signal that this task is done, ready for next sweep.
                        ss_cond_set( task, sweep + 1 );

                        // At the end of the matrix, signal that the
                        // next 2 tasks are ready for the next sweep.
                        // Marking these non-existent tasks simplifies
                        // the next sweep's dependencies.
                        if (sweep_done) {
                            for (int t = 1; t < shift; ++t) {
                                ss_cond_set( task + t, sweep + 1 );
                            }
                        }
                    } // if tid == task_tid

                    // sweep_begin reached the end of the matrix.
                    if (sweep_done) {
                        assert( sweep == sweep_begin );
                        ++sweep_begin;
                    }
                } // for sweep
            } // for k
        } // for i
    } // omp parallel

    free( (void*) plasma->ss_progress );

    //----------
    // Store resulting diag and sub-diag D and E.
    // Note that D and E are always real.
    // For lower, top row (i = 0) of band matrix A is diagonal D,
    // row i = 1 is sub-diagonal E.
    // For upper (untested), bottom row (i = nb) is diagonal D,
    // row i = nb-1 is super-diagonal E.
    // Sequential code here so only core 0 will work.
    if (uplo == PlasmaLower) {
        for (int j = 0; j < n - 1; ++j) {
            D[ j ] = creal( A[ j*lda ] );
            E[ j ] = creal( A[ j*lda + 1 ] );
        }
        D[ n-1 ] = creal( A[ (n-1)*lda ] );
    }
    else {
        for (int j = 0; j < n - 1; ++j) {
            D[ j ] = creal( A[ j*lda + nb ] );
            E[ j ] = creal( A[ j*lda + nb - 1 ] );
        }
        D[ n-1 ] = creal( A[ (n-1)*lda + nb ] );
    }
}
