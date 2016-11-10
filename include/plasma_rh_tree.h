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


inline void plasma_rh_tree_operation_insert(int *operations, int iops,
                                            plasma_enum_t kernel,
                                            int col, int row, int rowpiv)
{
    operations[iops*4]   = kernel;
    operations[iops*4+1] = col;
    operations[iops*4+2] = row;
    operations[iops*4+3] = rowpiv;
}

inline void plasma_rh_tree_operation_get(int *operations, int iops,
                                         plasma_enum_t *kernel, 
                                         int *col, int *row, int *rowpiv)
{
    *kernel = operations[iops*4];
    *col    = operations[iops*4+1];
    *row    = operations[iops*4+2];
    *rowpiv = operations[iops*4+3];
}

void plasma_rh_tree_operations(int mt, int nt, int **operations, int *noperations);

#endif // ICL_PLASMA_RH_TREE_H
