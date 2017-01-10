/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/

#include "plasma_barrier.h"

/******************************************************************************/
void plasma_barrier_init(plasma_barrier_t *barrier, int size)
{
    barrier->size = size;
    barrier->count = 0;
    barrier->passed = 0;
}

/******************************************************************************/
void plasma_barrier_wait(plasma_barrier_t *barrier)
{
    int passed_old = barrier->passed;

    __sync_fetch_and_add(&barrier->count, 1);
    if (__sync_bool_compare_and_swap(&barrier->count, barrier->size, 0))
        barrier->passed++;
    else
        while (barrier->passed == passed_old);
}
