/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 **/

#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_tree.h"

#include <omp.h>

void plasma_tree_flat_ts(int mt, int nt,
                         int **operations, int *num_operations,
                         plasma_sequence_t *sequence,
                         plasma_request_t *request);

void plasma_tree_flat_tt(int mt, int nt,
                         int **operations, int *num_operations,
                         plasma_sequence_t *sequence,
                         plasma_request_t *request);

void plasma_tree_binary(int mt, int nt,
                        int **operations, int *num_operations,
                        plasma_sequence_t *sequence,
                        plasma_request_t *request);

void plasma_tree_auto(int mt, int nt,
                      int **operations, int *num_operations,
                      int concurrency,
                      plasma_sequence_t *sequence,
                      plasma_request_t *request);

void plasma_tree_greedy(int mt, int nt,
                        int **operations, int *num_operations,
                        plasma_sequence_t *sequence,
                        plasma_request_t *request);

void plasma_tree_block_greedy(int mt, int nt,
                              int **operations, int *num_operations,
                              int concurrency,
                              plasma_sequence_t *sequence,
                              plasma_request_t *request);

static inline int get_super_tiles(int n, int bs) {
    return (n+(bs-1)) / bs;
}

static int plasma_tree_insert_flat_tree(int *operations, int loperations,
                                        int iops,
                                        int j, int k, int bs)
{
    iops = plasma_tree_insert_operation(operations,
                                        loperations,
                                        iops,
                                        PlasmaGeKernel,
                                        j, k, -1);
    for (int m = k+1; m < k+bs; m++) {
        iops = plasma_tree_insert_operation(operations,
                                            loperations,
                                            iops,
                                            PlasmaTsKernel,
                                            j, m, k);
    }
    return iops;
}

/***************************************************************************//**
 *  Routine for precomputing a given order of operations for tile
 *  QR and LQ factorization.
 * @see plasma_omp_zgeqrf
 **/
void plasma_tree_operations(int mt, int nt,
                            int **operations, int *num_operations,
                            plasma_sequence_t *sequence,
                            plasma_request_t *request)
{
    // Different algorithms can be implemented and switched here.
    const int tree_type = PlasmaTreeBlockGreedy;

    // Number of cores is useful for some algorithms.
    int ncores = omp_get_num_threads();

    switch (tree_type) {
        case PlasmaTreeFlatTs:
            // Flat tree as in the standard geqrf routine.
            // Combines only GE and TS kernels. Included mainly for debugging.
            plasma_tree_flat_ts(mt, nt, operations, num_operations,
                                sequence, request);
            break;
        case PlasmaTreeFlatTt:
            // Flat tree as in the standard geqrf routine, but this time
            // combines only GE and TT kernels.
            plasma_tree_flat_tt(mt, nt, operations, num_operations,
                                sequence, request);
            break;
        case PlasmaTreeBinary:
            // PLASMA-Tree from PLASMA 2.8.0.
            // Binary tree of flat trees of constant size.
            plasma_tree_binary(mt, nt, operations, num_operations,
                               sequence, request);
            break;
        case PlasmaTreeGreedy:
            // Pure Greedy algorithm combining only GE and TT kernels.
            plasma_tree_greedy(mt, nt, operations, num_operations,
                               sequence, request);
            break;
        case PlasmaTreeAuto:
            // Binary tree of flat trees, with changing size of the flat trees in each
            // column.
            plasma_tree_auto(mt, nt, operations, num_operations, ncores,
                             sequence, request);
            break;
        case PlasmaTreeBlockGreedy:
            // Greedy tree of flat trees.
            plasma_tree_block_greedy(mt, nt, operations, num_operations, ncores,
                                     sequence, request);
            break;
        default:
            plasma_error("Wrong value of tree_type.");
            plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
    }
}

/***************************************************************************//**
 *  Parallel tile QR factorization using the flat tree. This is the simplest
 *  tiled-QR algorithm based on TS (Triangle on top of Square) kernels.
 *  Implemented directly in the pzgeqrf and pzgelqf routines, it is included
 *  here mostly for debugging purposes.
 * @see plasma_omp_zgeqrf
 **/
void plasma_tree_flat_ts(int mt, int nt,
                         int **operations, int *num_operations,
                         plasma_sequence_t *sequence,
                         plasma_request_t *request)
{
    // How many columns to involve?
    int minnt = imin(mt, nt);

    // Only diagonal tiles are triangularized.
    size_t num_triangularized_tiles  = minnt;
    // Tiles on diagonal and above are not anihilated.
    size_t num_anihilated_tiles      = mt*minnt - (minnt+1)*minnt/2;

    // Number of operations can be directly computed.
    size_t loperations = num_triangularized_tiles + num_anihilated_tiles;

    // Allocate array of operations.
    *operations = (int *) malloc(loperations*4*sizeof(int));
    if (*operations == NULL) {
        plasma_error("Allocation of the array of operations failed.");
        plasma_request_fail(sequence, request, PlasmaErrorOutOfMemory);
    }

    // Counter of number of inserted operations.
    int iops = 0;
    for (int k = 0; k < minnt; k++) {
        iops = plasma_tree_insert_operation(*operations,
                                            loperations,
                                            iops,
                                            PlasmaGeKernel,
                                            k, k, -1);

        for (int m = k+1; m < mt; m++) {
            iops = plasma_tree_insert_operation(*operations,
                                                loperations,
                                                iops,
                                                PlasmaTsKernel,
                                                k, m, k);
        }
    }

    // Check that the expected number of operations was reached.
    if (iops != loperations) {
        plasma_error("Wrong number of operations in the tree.");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
    }

    // Copy over the number of operations.
    *num_operations = iops;
}

/***************************************************************************//**
 *  Parallel tile QR factorization using the flat tree. This is the simplest
 *  tiled-QR algorithm based on TT (Triangle on top of Triangle) kernels.
 * @see plasma_omp_zgeqrf
 **/
void plasma_tree_flat_tt(int mt, int nt,
                         int **operations, int *num_operations,
                         plasma_sequence_t *sequence,
                         plasma_request_t *request)
{
    // How many columns to involve?
    int minnt = imin(mt, nt);

    // Tiles above diagonal are not triangularized.
    size_t num_triangularized_tiles  = mt*minnt - (minnt-1)*minnt/2;
    // Tiles on diagonal and above are not anihilated.
    size_t num_anihilated_tiles      = mt*minnt - (minnt+1)*minnt/2;

    // Number of operations can be determined exactly.
    size_t loperations = num_triangularized_tiles + num_anihilated_tiles;

    // Allocate array of operations.
    *operations = (int *) malloc(loperations*4*sizeof(int));
    if (*operations == NULL) {
        plasma_error("Allocation of the array of operations failed.");
        plasma_request_fail(sequence, request, PlasmaErrorOutOfMemory);
    }

    // Counter of number of inserted operations.
    int iops = 0;
    for (int k = 0; k < minnt; k++) {
        // all tiles on diagonal and below are triangularized
        for (int m = k; m < mt; m++) {
            iops = plasma_tree_insert_operation(*operations,
                                                loperations,
                                                iops,
                                                PlasmaGeKernel,
                                                k, m, -1);
        }
        // all tiles below diagonal are eliminated by TT kernels
        for (int m = k+1; m < mt; m++) {
            iops = plasma_tree_insert_operation(*operations,
                                                loperations,
                                                iops,
                                                PlasmaTtKernel,
                                                k, m, k);
        }
    }

    // Check that the expected number of operations was reached.
    if (iops != loperations) {
        plasma_error("Wrong number of operations in the tree.");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
    }

    // Copy over the number of operations.
    *num_operations = iops;
}

/***************************************************************************//**
 *  Parallel tile communication avoiding QR factorization from
 *  PLASMA version 2.8.0.
 *  Also known as PLASMA-TREE, it combines TS kernels within
 *  blocks of tiles of height BS and TT kernels on top of these blocks in
 *  a binary-tree fashion.
 * @see plasma_omp_zgeqrf
 **/
void plasma_tree_binary(int mt, int nt,
                        int **operations, int *num_operations,
                        plasma_sequence_t *sequence,
                        plasma_request_t *request)
{
    static const int BS = 4;

    // How many columns to involve?
    int minnt = imin(mt, nt);

    // Tiles above diagonal are not triangularized.
    size_t num_triangularized_tiles  = ((mt/BS)+1)*minnt;
    // Tiles on diagonal and above are not anihilated.
    size_t num_anihilated_tiles      = mt*minnt - (minnt+1)*minnt/2;

    // An upper bound on the number of operations.
    size_t loperations = num_triangularized_tiles + num_anihilated_tiles;

    // Allocate array of operations.
    *operations = (int *) malloc(loperations*4*sizeof(int));
    if (*operations == NULL) {
        plasma_error("Allocation of the array of operations failed.");
        plasma_request_fail(sequence, request, PlasmaErrorOutOfMemory);
    }

    // Counter of number of inserted operations.
    int iops = 0;
    for (int k = 0; k < minnt; k++) {
        for (int M = k; M < mt; M += BS) {
            iops = plasma_tree_insert_operation(*operations,
                                                loperations,
                                                iops,
                                                PlasmaGeKernel,
                                                k, M, -1);
            for (int m = M+1; m < imin(M+BS, mt); m++) {
                iops = plasma_tree_insert_operation(*operations,
                                                    loperations,
                                                    iops,
                                                    PlasmaTsKernel,
                                                    k, m, M);
            }
        }
        for (int rd = BS; rd < mt-k; rd *= 2) {
            for (int M = k; M+rd < mt; M += 2*rd) {
                iops = plasma_tree_insert_operation(*operations,
                                                    loperations,
                                                    iops,
                                                    PlasmaTtKernel,
                                                    k, M+rd, M);
            }
        }
    }

    // Check that we have reached the expected number of operations.
    if (iops > loperations) {
        plasma_error("Too many operations in the tree.");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
    }

    // Copy over the number of operations.
    *num_operations = iops;
}

/***************************************************************************//**
 *  Parallel tile QR factorization inspired by the AUTO algorithm from
 *  M. Faverge, J. Langou, Y. Robert, and J. Dongarra.
 *  Bidiagonalization with Parallel Tiled Algorithms. (2016). arXiv:1611.06892
 *  http://arxiv.org/abs/1611.06892
 * @see plasma_omp_zgeqrf
 **/
void plasma_tree_auto(int mt, int nt,
                      int **operations, int *num_operations,
                      int concurrency,
                      plasma_sequence_t *sequence,
                      plasma_request_t *request)
{
    // Multiple of the target concurrency to set sizes of the flat tree in
    // each column.
    static const int gamma = 2;

    // Check input.
    if (concurrency < 1) {
        plasma_error("Illegal value of concurrency.");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
    }

    // How many columns to involve?
    int minnt = imin(mt, nt);

    // Tiles above diagonal are not triangularized.
    size_t num_triangularized_tiles  = mt*minnt - (minnt-1)*minnt/2;
    // Tiles on diagonal and above are not anihilated.
    size_t num_anihilated_tiles      = mt*minnt - (minnt+1)*minnt/2;

    // Number of operations can be only estimated.
    size_t loperations = num_triangularized_tiles + num_anihilated_tiles;

    // Allocate array of operations.
    *operations = (int *) malloc(loperations*4*sizeof(int));
    if (*operations == NULL) {
        plasma_error("Allocation of the array of operations failed.");
        plasma_request_fail(sequence, request, PlasmaErrorOutOfMemory);
    }

    int iops  = 0;
    for (int j = 0; j < minnt; j++) { // loop over columns from the beginning

        // Constant block size.
        //int bs = 4;
        // Determine the size of the flat tree for this column.
        // intentional integer division
        int bs = imax(1,(mt-j-1)*(minnt-j-1) / (gamma*concurrency));

        // Triangularize all supertiles in this column.

        // number of supertiles to triangularize - i.e. insert flat tree
        int nT = get_super_tiles(imax(0, mt - j), bs);
        for (int ks = 0; ks < nT; ks++) {
            int k = j + (nT-ks-1)*bs;

            iops = plasma_tree_insert_flat_tree(*operations,
                                                loperations,
                                                iops,
                                                j, k,
                                                imin(bs,mt-k));
        }

        // Eliminate every tile triangularized in the previous step.
        int nZ_target = get_super_tiles(imax(0, mt - j - bs), bs);
        int nZ = 0;
        while (nZ < nZ_target) {
            int batch = (nT - nZ) / 2; // intentional integer division
            int nZnew = nZ + batch;

            for (int ks = nZ; ks < nZnew; ks++) {
                // row index of a tile to be zeroed
                int pmkk    = j + (nZ_target-ks)*bs;
                // row index of the anihilator tile
                int pivpmkk = pmkk - batch*bs;

                iops = plasma_tree_insert_operation(*operations,
                                                    loperations,
                                                    iops,
                                                    PlasmaTtKernel,
                                                    j, pmkk, pivpmkk);
            }
            // Update the number of eliminated tiles.
            nZ = nZnew;
        }
    }

    // Check that we have reached the expected number of operations.
    if (iops > loperations) {
        plasma_error("Wrong number of operations in the tree.");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
    }

    // Copy over the number of operations.
    *num_operations = iops;
}

/***************************************************************************//**
 *  Parallel tile QR factorization based on the GREEDY algorithm from
 *  H. Bouwmeester, M. Jacquelin, J. Langou, Y. Robert
 *  Tiled QR factorization algorithms. INRIA Report no. 7601, 2011.
 * @see plasma_omp_zgeqrf
 **/
void plasma_tree_greedy(int mt, int nt,
                        int **operations, int *num_operations,
                        plasma_sequence_t *sequence,
                        plasma_request_t *request)
{
    // How many columns to involve?
    int minnt = imin(mt, nt);

    // Tiles above diagonal are not triangularized.
    size_t num_triangularized_tiles  = mt*minnt - (minnt-1)*minnt/2;
    // Tiles on diagonal and above are not anihilated.
    size_t num_anihilated_tiles      = mt*minnt - (minnt+1)*minnt/2;

    // Number of operations can be determined exactly.
    size_t loperations = num_triangularized_tiles + num_anihilated_tiles;

    // Allocate array of operations.
    *operations = (int *) malloc(loperations*4*sizeof(int));
    if (*operations == NULL) {
        plasma_error("Allocation of the array of operations failed.");
        plasma_request_fail(sequence, request, PlasmaErrorOutOfMemory);
    }

    // Prepare memory for column counters.
    int *NZ = (int*) malloc(minnt*sizeof(int));
    if (NZ == NULL) {
        plasma_error("Allocation of the array NZ failed.");
        plasma_request_fail(sequence, request, PlasmaErrorOutOfMemory);
    }
    int *NT = (int*) malloc(minnt*sizeof(int));
    if (NT == NULL) {
        plasma_error("Allocation of the array NT failed.");
        plasma_request_fail(sequence, request, PlasmaErrorOutOfMemory);
    }

    // Initialize column counters.
    for (int j = 0; j < minnt; j++) {
        // NZ[j] is the number of tiles which have been eliminated in column j
        NZ[j] = 0;
        // NT[j] is the number of tiles which have been triangularized
        // in column j
        NT[j] = 0;
    }

    int nZnew = 0;
    int nTnew = 0;
    int iops  = 0;
    // Until the last column is finished...
    while ((NT[minnt-1] < mt - minnt + 1) ||
           (NZ[minnt-1] < mt - minnt)    ) {
        for (int j = minnt-1; j >= 0; j--) {
            if (j == 0) {
                // Triangularize the first column if not yet done.
                nTnew = NT[j] + (mt-NT[j]);
                if (mt - NT[j] > 0) {
                    for (int k = mt - 1; k >= 0; k--) {
                        iops = plasma_tree_insert_operation(*operations,
                                                            loperations,
                                                            iops,
                                                            PlasmaGeKernel,
                                                            j, k, -1);
                    }
                }
            }
            else {
                // Triangularize every tile having zero in the previous column.
                nTnew = NZ[j-1];
                for (int k = NT[j]; k < nTnew; k++) {
                    int kk = mt-k-1;

                    iops = plasma_tree_insert_operation(*operations,
                                                        loperations,
                                                        iops,
                                                        PlasmaGeKernel,
                                                        j, kk, -1);
                }
            }

            // Eliminate every tile triangularized in the previous step.
            int batch = (NT[j] - NZ[j]) / 2; // intentional integer division
            nZnew = NZ[j] + batch;
            for (int kk = NZ[j]; kk < nZnew; kk++) {
                int pmkk    = mt-kk-1;  // row index of a tile to be zeroed
                int pivpmkk = pmkk-batch; // row index of the anihilator tile

                iops = plasma_tree_insert_operation(*operations,
                                                    loperations,
                                                    iops,
                                                    PlasmaTtKernel,
                                                    j, pmkk, pivpmkk);
            }
            // Update the number of triangularized and eliminated tiles at the
            // next step.
            NT[j] = nTnew;
            NZ[j] = nZnew;
        }
    }

    // Check that we have reached the expected number of operations.
    if (iops != loperations) {
        plasma_error("Wrong number of operations in the tree.");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
    }

    // Copy over the number of operations.
    *num_operations = iops;

    // Deallocate column counters.
    free(NZ);
    free(NT);
}

/***************************************************************************//**
 *  Parallel tile QR factorization based on the GREEDY algorithm from
 *  H. Bouwmeester, M. Jacquelin, J. Langou, Y. Robert
 *  Tiled QR factorization algorithms. INRIA Report no. 7601, 2011.
 *  Extended to blocks of flat-trees combined in the greedy fashion.
 * @see plasma_omp_zgeqrf
 **/
void plasma_tree_block_greedy(int mt, int nt,
                              int **operations, int *num_operations,
                              int concurrency,
                              plasma_sequence_t *sequence,
                              plasma_request_t *request)
{
    // Check input.
    if (concurrency < 1) {
        plasma_error("Illegal value of concurrency.");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
    }

    // How many columns to involve?
    int minnt = imin(mt, nt);

    // Costant block size.
    //int bs = 4;

    // Block size adapting to the number of columns.
    // Multiple of the target concurrency to set sizes of the flat trees.
    static const int gamma = 4;
    int bs = imin(mt, imax(1, mt * (minnt*minnt/2 + minnt/2)
                                 / (gamma * concurrency)));
    //printf("bs = %d \n", bs);

    // Tiles above diagonal are not triangularized.
    size_t num_triangularized_tiles  = mt*minnt
                                     - minnt*(minnt-1)/2;
    // Tiles on diagonal and above are not anihilated.
    size_t num_anihilated_tiles      = mt*minnt - (minnt+1)*minnt/2;

    // Number of operations can be determined exactly.
    size_t loperations = num_triangularized_tiles + num_anihilated_tiles;

    // Allocate array of operations.
    *operations = (int *) malloc(loperations*4*sizeof(int));
    if (*operations == NULL) {
        plasma_error("Allocation of the array of operations failed.");
        plasma_request_fail(sequence, request, PlasmaErrorOutOfMemory);
    }

    // Prepare memory for column counters.
    int *NZ = (int*) malloc(minnt*sizeof(int));
    if (NZ == NULL) {
        plasma_error("Allocation of the array NZ failed.");
        plasma_request_fail(sequence, request, PlasmaErrorOutOfMemory);
    }
    int *NT = (int*) malloc(minnt*sizeof(int));
    if (NT == NULL) {
        plasma_error("Allocation of the array NT failed.");
        plasma_request_fail(sequence, request, PlasmaErrorOutOfMemory);
    }

    // Initialize column counters.
    for (int j = 0; j < minnt; j++) {
        // NZ[j] is the number of tiles which have been eliminated in column j
        NZ[j] = 0;
        // NT[j] is the number of tiles which have been triangularized
        // in column j
        NT[j] = 0;
    }

    int nZnew = 0;
    int nTnew = 0;
    int iops  = 0;
    // Until the last column is finished...
    while ((NT[minnt-1] < get_super_tiles(mt - minnt + 1, bs) ||
           (NZ[minnt-1] < get_super_tiles(mt - minnt + 1, bs) - 1))) {
        int updated = 0;
        for (int j = minnt-1; j >= 0; j--) {
            if (j == 0) {
                // Triangularize the first column if not yet done.
                nTnew = NT[j] + (get_super_tiles(mt, bs) - NT[j]);
                if (get_super_tiles(mt, bs) - NT[j] > 0) {
                    for (int k = get_super_tiles(mt, bs) - 1; k >= 0; k--) {
                        int itile = j + k*bs;
                        iops = plasma_tree_insert_flat_tree(*operations,
                                                            loperations,
                                                            iops,
                                                            j, itile,
                                                            imin(bs, mt-itile));
                    }
                }
            }
            else {
                // Triangularize every tile having zero in the previous column.
                if ((mt - (j-1)) % bs != 1 && bs > 1 && NT[j-1] > NZ[j-1]) {
                    nTnew = NZ[j-1] + 1;
                }
                else {
                    nTnew = NZ[j-1];
                }

                for (int k = NT[j]; k < nTnew; k++) {
                    int kk = get_super_tiles(mt-j, bs) - k - 1;
                    int itile = j + kk*bs;
                    iops = plasma_tree_insert_flat_tree(*operations,
                                                        loperations,
                                                        iops,
                                                        j, itile,
                                                        imin(bs, mt-itile));
                }
            }

            // Eliminate every tile triangularized in the previous step.
            int batch = (NT[j] - NZ[j]) / 2; // intentional integer division
            nZnew = NZ[j] + batch;
            for (int kk = NZ[j]; kk < nZnew; kk++) {
                // row index of a tile to be zeroed
                int pmkk    = get_super_tiles(mt-j, bs)-kk-1;
                // row index of the anihilator tile
                int pivpmkk = pmkk-batch;

                int itilepmkk    = j + pmkk*bs;    // row index of a tile to be zeroed
                int itilepivpmkk = j + pivpmkk*bs; // row index of the anihilator tile

                iops = plasma_tree_insert_operation(*operations,
                                                    loperations,
                                                    iops,
                                                    PlasmaTtKernel,
                                                    j, itilepmkk, itilepivpmkk);
            }
            // Update the number of triangularized and eliminated tiles at the
            // next step.
            if (nTnew != NT[j] || nZnew != NZ[j]) {
                updated = 1;
            }
            NT[j] = nTnew;
            NZ[j] = nZnew;
        }

        if (!updated) {
            printf("plasma_tree_block_greedy: Error, no column updated! \n");
            break;
        }
    }

    // Check that we have reached the expected number of operations.
    if (iops > loperations) {
        plasma_error("Too many operations in the tree.");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
    }

    // Copy over the number of operations.
    *num_operations = iops;

    // Deallocate column counters.
    free(NZ);
    free(NT);
}
