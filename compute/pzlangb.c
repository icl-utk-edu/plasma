/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> s d c
 *
 **/

#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_types.h"
#include "plasma_workspace.h"
#include "core_blas.h"
#include "core_lapack.h"

#define A(m, n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)

/***************************************************************************//**
 *  Parallel tile calculation of max, one, infinity or Frobenius matrix norm
 *  for a general band matrix.
 ******************************************************************************/
void plasma_pzlangb(plasma_enum_t norm,
                    plasma_desc_t A, double *work, double *value,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;
    int klut = A.klt + A.kut - 1; // # tile rows in band packed storage
    
    double stub;
    switch (norm) {
    
    //================
    // PlasmaMaxNorm
    //================
    case PlasmaMaxNorm:
        for (int n = 0; n < A.nt; n++ ) {
            int nvan = plasma_tile_nview(A, n);
            for (int m = imax(0, n-A.kut+1); m < imin(A.mt, n+A.klt-1); m++ ) {
                int ldam = plasma_tile_mmain_band(A, m, n);
                int mvam = plasma_tile_mview(A, m);
                /* printf("[plasma_pzlangb]: dispatching work to work[%d]\n", */
                /*        (A.kut-1+m-n)+n*klut); */
                core_omp_zlange(PlasmaMaxNorm,
                                mvam, nvan,
                                A(m, n), ldam, 
                                &stub, &work[(A.kut-1+m-n)+n*klut],
                                sequence, request);
            }
        }
		// zero out the unused elements in work.
		for (int j = 0; j < A.nt; j++) {
		    for (int i = 0; i < A.kut - 1 - j; i++ ) 
		        work[i+j*klut] = 0;
		    for (int i = klut-1; i >= A.kut+A.nt-j-1; i-- )
		        work[i+j*klut] = 0;
		}        
        #pragma omp taskwait
        printf("[plasma_pzlangb]: klt=%d, kut=%d, klut=%d\n", A.klt,A.kut,klut);
        printf("[plasma_pzlangb]: aggregating...\n");
        printf("[plasma_work]: work...");
        for (int i=0; i<klut*A.nt; i++) {
        	printf("%.3f\t", work[i]);
        }
        printf("\n");
        core_omp_dlange(PlasmaMaxNorm,
                        klut, A.nt,
                        work, klut,
                        &stub, value,
                        sequence, request);
        break;
    case PlasmaOneNorm:
        for (int n = 0; n < A.nt; n++ ) {
            int nvan = plasma_tile_nview(A, n);
            for (int m = imax(0, n-A.kut+A.klt); m < imin(A.mt, n+A.klt-1); m++ ) {
                int ldam = plasma_tile_mmain_band(A, m, n);
                int mvam = plasma_tile_mview(A, m);
                core_omp_zlange_aux(PlasmaOneNorm,
                                    mvam, nvan,
                                    A(m,n), ldam,
                                    &work[n*A.nb+(m-imax(0, n-A.kut+A.klt))*A.n],
                                    sequence, request);
            }
        }
        #pragma omp taskwait
        printf("[plasma_work]: work...\n");
        printf("\n");
        double *workspace = work + klut*A.n;
        /*core_omp_dlange(PlasmaInfNorm,
                        A.n, klut,
                        work, A.n,
                        workspace, value,
                        sequence, request);*/
        char *c = "i";
        klut -= klut - 1;
        *value = dlange_(c, &A.n, &klut, work, &A.n, workspace);
        printf("%s:%d [%s] value = %.3f\n", __FILE__, __LINE__, __FUNCTION__, *value);
        break;
    default:
        assert(0);
    }
}
