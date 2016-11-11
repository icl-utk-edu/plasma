/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 **/

#ifndef ICL_PLASMA_RH_TREE_H
#define ICL_PLASMA_RH_TREE_H

enum {
    PlasmaGEKernel,
    PlasmaTTKernel,
    PlasmaTSKernel,
};


void plasma_rh_tree_operation_insert(int *operations, int iops,
                                     plasma_enum_t kernel,
                                     int col, int row, int rowpiv);

void plasma_rh_tree_operation_get(int *operations, int iops,
                                  plasma_enum_t *kernel,
                                  int *col, int *row, int *rowpiv);

void plasma_rh_tree_operations(int mt, int nt,
                               int **operations, int *noperations);

#endif // ICL_PLASMA_RH_TREE_H
