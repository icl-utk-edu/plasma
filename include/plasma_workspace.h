/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#ifndef PLASMA_WORKSPACE_H
#define PLASMA_WORKSPACE_H

#include "plasma_types.h"

#include <stdlib.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void **spaces;      ///< array of nthread pointers to workspaces
    size_t lworkspace;  ///< length in elements of workspace on each core
    int nthread;        ///< number of threads
    plasma_enum_t dtyp; ///< precision of the workspace
} plasma_workspace_t;

/******************************************************************************/
int plasma_workspace_create(plasma_workspace_t *workspace, size_t lworkspace,
                           plasma_enum_t dtyp);

int plasma_workspace_destroy(plasma_workspace_t *workspace);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // PLASMA_WORKSPACE_H
