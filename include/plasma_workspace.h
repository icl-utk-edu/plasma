/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#ifndef ICL_PLASMA_WORKSPACE_H
#define ICL_PLASMA_WORKSPACE_H

#include "plasma_types.h"

#include <stdlib.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void **spaces;    ///< array of nthread pointers to workspaces
    size_t lwork;     ///< length in elements of workspace on each core
    int nthread;      ///< number of threads
    PLASMA_enum dtyp; ///< precision of the workspace
} PLASMA_workspace;

/******************************************************************************/
int plasma_workspace_alloc(PLASMA_workspace *work, size_t lwork, PLASMA_enum dtyp);

int plasma_workspace_free(PLASMA_workspace *work);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_PLASMA_DESCRIPTOR_H
