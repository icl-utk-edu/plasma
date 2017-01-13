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
#include "plasma_rh_tree.h"

#include <omp.h>

void plasma_rh_tree_flat_ts(int mt, int nt,
                            int **operations, int *num_operations);

void plasma_rh_tree_flat_tt(int mt, int nt,
                            int **operations, int *num_operations);

void plasma_rh_tree_plasmatree(int mt, int nt,
                               int **operations, int *num_operations);

void plasma_rh_tree_greedy(int mt, int nt,
                           int **operations, int *num_operations);

void plasma_rh_tree_auto_forest(int mt, int nt,
                                int **operations, int *num_operations,
                                int concurrency);

/***************************************************************************//**
 *  Routine for precomputing a given order of operations for tile
 *  QR and LQ factorization.
 * @see plasma_omp_zgeqrf
 **/
void plasma_rh_tree_operations(int mt, int nt,
                               int **operations, int *num_operations)
{
    // Different algorithms can be implemented and switched here:

    // Flat tree as in the standard geqrf routine.
    // Combines only GE and TS kernels. Included mainly for debugging.
    //plasma_rh_tree_flat_ts(mt, nt, operations, num_operations);

    // Flat tree as in the standard geqrf routine, but this time
    // combines only GE and TT kernels.
    //plasma_rh_tree_flat_tt(mt, nt, operations, num_operations);

    // PLASMA-Tree from PLASMA 2.8.0
    plasma_rh_tree_plasmatree(mt, nt, operations, num_operations);

    // Pure Greedy algorithm combining only GE and TT kernels.
    //plasma_rh_tree_greedy(mt, nt, operations, num_operations);

    // Binary forest of flat trees.
    //int ncores = omp_get_num_threads();
    //plasma_rh_tree_auto_forest(mt, nt, operations, num_operations, ncores);
}

/***************************************************************************//**
 *  Parallel tile QR factorization using the flat tree. This is the simplest
 *  tiled-QR algorithm based on TS (Triangle on top of Square) kernels.
 *  Implemented directly in the pzgeqrf and pzgelqf routines, it is included
 *  here mostly for debugging purposes.
 * @see plasma_omp_zgeqrf
 **/
void plasma_rh_tree_flat_ts(int mt, int nt,
                            int **operations, int *num_operations)
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
    assert(*operations != NULL);

    // Counter of number of inserted operations.
    int iops = 0;
    for (int k = 0; k < minnt; k++) {
        iops = plasma_rh_tree_insert_operation(*operations,
                                               loperations,
                                               iops,
                                               PlasmaGeKernel,
                                               k, k, -1);

        for (int m = k+1; m < mt; m++) {
            iops = plasma_rh_tree_insert_operation(*operations,
                                                   loperations,
                                                   iops,
                                                   PlasmaTsKernel,
                                                   k, m, k);
        }
    }

    // Check that the expected number of operations was reached.
    assert(iops == loperations);

    // Copy over the number of operations.
    *num_operations = iops;
}

/***************************************************************************//**
 *  Parallel tile QR factorization using the flat tree. This is the simplest
 *  tiled-QR algorithm based on TT (Triangle on top of Triangle) kernels.
 * @see plasma_omp_zgeqrf
 **/
void plasma_rh_tree_flat_tt(int mt, int nt,
                            int **operations, int *num_operations)
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
    assert(*operations != NULL);

    // Counter of number of inserted operations.
    int iops = 0;
    for (int k = 0; k < minnt; k++) {
        // all tiles on diagonal and below are triangularized
        for (int m = k; m < mt; m++) {
            iops = plasma_rh_tree_insert_operation(*operations,
                                                   loperations,
                                                   iops,
                                                   PlasmaGeKernel,
                                                   k, m, -1);
        }
        // all tiles below diagonal are eliminated by TT kernels
        for (int m = k+1; m < mt; m++) {
            iops = plasma_rh_tree_insert_operation(*operations,
                                                   loperations,
                                                   iops,
                                                   PlasmaTtKernel,
                                                   k, m, k);
        }
    }

    // Check that the expected number of operations was reached.
    assert(iops == loperations);

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
void plasma_rh_tree_plasmatree(int mt, int nt,
                               int **operations, int *num_operations)
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
    assert(*operations != NULL);

    // Counter of number of inserted operations.
    int iops = 0;
    for (int k = 0; k < minnt; k++) {
        for (int M = k; M < mt; M += BS) {
            iops = plasma_rh_tree_insert_operation(*operations,
                                                   loperations,
                                                   iops,
                                                   PlasmaGeKernel,
                                                   k, M, -1);
            for (int m = M+1; m < imin(M+BS, mt); m++) {
                iops = plasma_rh_tree_insert_operation(*operations,
                                                       loperations,
                                                       iops,
                                                       PlasmaTsKernel,
                                                       k, m, M);
            }
        }
        for (int rd = BS; rd < mt-k; rd *= 2) {
            for (int M = k; M+rd < mt; M += 2*rd) {
                iops = plasma_rh_tree_insert_operation(*operations,
                                                       loperations,
                                                       iops,
                                                       PlasmaTtKernel,
                                                       k, M+rd, M);
            }
        }
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
void plasma_rh_tree_greedy(int mt, int nt,
                           int **operations, int *num_operations)
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
    assert(*operations != NULL);

    // Prepare memory for column counters.
    int *NZ = (int*) malloc(minnt*sizeof(int));
    assert(NZ != NULL);
    int *NT = (int*) malloc(minnt*sizeof(int));
    assert(NT != NULL);

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
                        iops = plasma_rh_tree_insert_operation(*operations,
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

                    iops = plasma_rh_tree_insert_operation(*operations,
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

                iops = plasma_rh_tree_insert_operation(*operations,
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
    assert(iops == loperations);

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
 *  It is combined with a flat tree on top of blocks of tiles.
 * @see plasma_omp_zgeqrf
 **/
static inline int get_super_tiles(int n, int bs) {
    return (n+(bs-1)) / bs;
}

static int plasma_rh_tree_insert_flat_tree(int *operations, int loperations,
                                           int iops,
                                           int j, int k, int bs)
{
    iops = plasma_rh_tree_insert_operation(operations,
                                           loperations,
                                           iops,
                                           PlasmaGeKernel,
                                           j, k, -1);
    for (int m = k+1; m < k+bs; m++) {
        iops = plasma_rh_tree_insert_operation(operations,
                                               loperations,
                                               iops,
                                               PlasmaTsKernel,
                                               j, m, k);
    }
    return iops;
}

void plasma_rh_tree_auto_forest(int mt, int nt,
                                int **operations, int *num_operations,
                                int concurrency)
{
    // Multiple of the target concurrency to set sizes of the flat tree in
    // each column.
    static const int gamma = 2;

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
    assert(*operations != NULL);

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

            iops = plasma_rh_tree_insert_flat_tree(*operations,
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

                iops = plasma_rh_tree_insert_operation(*operations,
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
    assert(iops <= loperations);

    // Copy over the number of operations.
    *num_operations = iops;
}
