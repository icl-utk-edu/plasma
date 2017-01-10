/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#include "plasma_workspace.h"
#include "plasma_internal.h"

#include <omp.h>

/******************************************************************************/
int plasma_workspace_create(plasma_workspace_t *work, size_t lwork,
                            plasma_enum_t dtyp)
{
    // Allocate array of pointers.
    #pragma omp parallel
    #pragma omp master
    {
        work->nthread = omp_get_num_threads();
    }
    work->lwork = lwork;
    work->dtyp  = dtyp;
    if ((work->spaces = (void**)calloc(work->nthread, sizeof(void*))) == NULL) {
        free(work->spaces);
        plasma_error("malloc() failed");
        return PlasmaErrorOutOfMemory;
    }

    // Each thread allocates its workspace.
    size_t size = (size_t)lwork * plasma_element_size(work->dtyp);
    int info = PlasmaSuccess;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if ((work->spaces[tid] = (void*)malloc(size)) == NULL) {
            info = PlasmaErrorOutOfMemory;
        }
    }
    if (info != PlasmaSuccess) {
        plasma_workspace_destroy(work);
    }
    return info;
}

/******************************************************************************/
int plasma_workspace_destroy(plasma_workspace_t *work)
{
    if (work->spaces != NULL) {
        for (int i = 0; i < work->nthread; ++i) {
            free(work->spaces[i]);
            work->spaces[i] = NULL;
        }
        free(work->spaces);
        work->spaces  = NULL;
        work->nthread = 0;
        work->lwork   = 0;
    }
    return PlasmaSuccess;
}
