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

inline void plasma_rh_tree_operation_insert(int *operations, int iops,
                                            int col, int row, int rowpiv)
{
    operations[iops*3  ] = col;
    operations[iops*3+1] = row;
    operations[iops*3+2] = rowpiv;
}

inline void plasma_rh_tree_operation_get(int *operations, int iops,
                                         int *col, int *row, int *rowpiv)
{
    *col    = operations[iops*3];
    *row    = operations[iops*3+1];
    *rowpiv = operations[iops*3+2];
}

void plasma_rh_tree_operations(int mt, int nt, int **operations, int *noperations);

#endif // ICL_PLASMA_RH_TREE_H
