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

void plasma_rh_tree_greedy(int mt, int nt, int **operations, int *noperations);

/***************************************************************************//**
 *  Routine for precomputing a given order of operations for tile 
 *  QR and LQ factorization.
 * @see plasma_omp_zgeqrf
 **/
void plasma_rh_tree_operations(int mt, int nt,
                               int **operations, int *noperations)
{
    // Different algorithms can be implemented and plugged-in here.
    plasma_rh_tree_greedy(mt, nt, operations, noperations);
}


/***************************************************************************//**
 *  Parallel tile QR factorization based on the GREEDY algorithm from 
 *  H. Bouwmeester, M. Jacquelin, J. Langou, Y. Robert
 *  Tiled QR factorization algorithms. INRIA Report no. 7601, 2011.
 * @see plasma_omp_zgeqrf
 **/
void plasma_rh_tree_greedy(int mt, int nt, int **operations, int *noperations)
{
    static const int debug = 0;

    // how many columns to involve
    int minnt = imin(mt, nt);

    // tiles above diagonal are not triangularized
    int num_triangularized_tiles  = mt*minnt - (minnt-1)*minnt/2; 
    // tiles on diagonal and above are not anihilated
    int num_anihilated_tiles      = mt*minnt - (minnt+1)*minnt/2; 

    // Allocate array of operations
    int nops = num_triangularized_tiles + num_anihilated_tiles;

    *operations = (int *) malloc(nops*3*sizeof(int));

    // Prepare memory for column counters.
    int *NZ = (int*) malloc(minnt*sizeof(int));
    int *NT = (int*) malloc(minnt*sizeof(int));

    // Initialize column counters.
    for (int j = 0; j < minnt; j++) {
        // NZ[j] is the number of tiles which have been eliminated in column j
        NZ[j] = 0;
        // NT[j] is the number of tiles which have been triangularized in column j
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

                        // GEQRT(k,j)
                        if (debug) printf("GEQRT (%d,%d) ", k, j);
                        plasma_rh_tree_operation_insert(*operations, iops,
                                                        j, k, -1);
                        iops++;
                        if (debug) printf("\n ");
                    }
                }
            }
            else {
                // Triangularize every tile having zero in the previous column.
                nTnew = NZ[j-1];
                for (int k = NT[j]; k < nTnew; k++) {
                    int kk = mt-k-1;

                    // GEQRT(kk,j)
                    if (debug) printf("GEQRT (%d,%d) ", kk, j);
                    plasma_rh_tree_operation_insert(*operations, iops,
                                                    j, kk, -1);
                    iops++;

                    if (debug) printf("\n ");
                }
            }

            // Eliminate every tile triangularized in the previous step.
            int batch = (NT[j] - NZ[j]) / 2; // intentional integer division
            nZnew = NZ[j] + batch;
            for (int kk = NZ[j]; kk < nZnew; kk++) {

                int pmkk    = mt-kk-1;  // row index of a tile to be zeroed
                int pivpmkk = pmkk-batch; // row index of the anihilator tile

                // TTQRT(mt- kk - 1, pivpmkk, j)
                if (debug) printf("TTQRT (%d,%d,%d) ", pmkk, pivpmkk, j);
                plasma_rh_tree_operation_insert(*operations, iops,
                                                j, pmkk, pivpmkk);
                iops++;

                if (debug) printf("\n ");
            }
            // Update the number of triangularized and eliminated tiles at the
            // next step.
            NT[j] = nTnew;
            NZ[j] = nZnew;
        }
    }

    if (iops != nops) {
        printf("I have not reached the expected number of operations.");
    }

    // return number of operations
    *noperations = nops;

    // Deallocate column counters.
    free(NZ);
    free(NT);
}

