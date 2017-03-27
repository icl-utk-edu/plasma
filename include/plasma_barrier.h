/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#ifndef ICL_PLASMA_BARRIER_H
#define ICL_PLASMA_BARRIER_H

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************/
typedef struct {
    int count;
    volatile int passed;
} plasma_barrier_t;

/******************************************************************************/
void plasma_barrier_init(plasma_barrier_t *barrier);
void plasma_barrier_wait(plasma_barrier_t *barrier, int size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_PLASMA_BARRIER_H
