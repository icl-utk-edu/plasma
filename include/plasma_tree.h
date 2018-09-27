/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 **/

#ifndef PLASMA_TREE_H
#define PLASMA_TREE_H

enum {
    PlasmaGeKernel = 1,
    PlasmaTtKernel = 2,
    PlasmaTsKernel = 3
};

enum {
    PlasmaTreeFlatTs      = 1,
    PlasmaTreeFlatTt      = 2,
    PlasmaTreeBinary      = 3,
    PlasmaTreeAuto        = 4,
    PlasmaTreeGreedy      = 5,
    PlasmaTreeBlockGreedy = 6,
};

/***************************************************************************//**
 *  Routine for registering a kernel into the list of operations for tile
 *  QR and LQ factorization.
 * @see plasma_omp_zgeqrf
 **/
static inline int plasma_tree_insert_operation(int *operations,
                                               int loperations,
                                               int ind_op,
                                               plasma_enum_t kernel,
                                               int col, int row, int rowpiv)
{
    assert(ind_op < loperations);

    operations[ind_op*4]   = kernel;
    operations[ind_op*4+1] = col;
    operations[ind_op*4+2] = row;
    operations[ind_op*4+3] = rowpiv;

    ind_op++;

    return ind_op;
}

/***************************************************************************//**
 *  Routine for getting a kernel from the list of operations for tile
 *  QR and LQ factorization.
 * @see plasma_omp_zgeqrf
 **/
static inline void plasma_tree_get_operation(int *operations,
                                             int ind_op,
                                             plasma_enum_t *kernel,
                                             int *col, int *row, int *rowpiv)
{
    *kernel = operations[ind_op*4];
    *col    = operations[ind_op*4+1];
    *row    = operations[ind_op*4+2];
    *rowpiv = operations[ind_op*4+3];
}

void plasma_tree_operations(int mt, int nt,
                            int **operations, int *num_operations,
                            plasma_sequence_t *sequence,
                            plasma_request_t *request);

#endif // PLASMA_TREE_H
