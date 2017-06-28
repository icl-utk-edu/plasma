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
    //int klut = A.klt + A.kut - 1; // # tile rows in band packed storage
#if 0
    // let's take a look at the A structure.
    printf("[plasma_pzlangb]: inspecting the A structure\n");
    printf("mb\tnb\tgm\tgn\tgmt\tgnt\ti\tj\tm\tn\tmt\tnt\tkl\tku\tklt\tkut\n");
    printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t\nt",
	   A.mb, A.nb, A.gm, A.gn, A.gmt, A.gnt, A.i, A.j, A.m, A.n, A.mt, A.nt,
	   A.kl, A.ku, A.klt, A.kut);
#endif
    double stub;
    int wcnt = 0;
    int ldwork, nwork, klt, kut;
    char *c;
    double *workspace, *scale, *sumsq;
    switch (norm) {
    
    //================
    // PlasmaMaxNorm
    //================
    case PlasmaMaxNorm:

        for (int n = 0; n < A.nt; n++ ) {
            int nvan = plasma_tile_nview(A, n);
	    int m_start = (imax(0, n*A.nb-A.ku)) / A.nb;
	    int m_end = (imin(A.m-1, (n+1)*A.nb+A.kl-1)) / A.nb;
            for (int m = m_start; m <= m_end; m++ ) {
                int ldam = plasma_tile_mmain_band(A, m, n);
                int mvam = plasma_tile_mview(A, m);
                core_omp_zlange(PlasmaMaxNorm,
                                mvam, nvan,
                                A(m, n), ldam, 
                                &stub, &work[wcnt],
                                sequence, request);
		wcnt++;
            }
        }

        #pragma omp taskwait
        core_omp_dlange(PlasmaMaxNorm,
                        1, wcnt,
                        work, 1,
                        &stub, value,
                        sequence, request);
        break;
    case PlasmaOneNorm:
        for (int n = 0; n < A.nt; n++ ) {
            int nvan = plasma_tile_nview(A, n);
	    int m_start = (imax(0, n*A.nb-A.ku)) / A.nb;
	    int m_end = (imin(A.m-1, (n+1)*A.nb+A.kl-1)) / A.nb;
	    int kut  = (A.ku+A.nb-1)/A.nb; // # of tiles in upper band (not including diagonal)
	    int klt  = (A.kl+A.nb-1)/A.nb;    // # of tiles in lower band (not including diagonal)
	    ldwork = kut+klt+1;
            for (int m = m_start; m <= m_end; m++ ) {
                int ldam = plasma_tile_mmain_band(A, m, n);
                int mvam = plasma_tile_mview(A, m);
                core_omp_zlange_aux(PlasmaOneNorm,
				    mvam, nvan,
				    A(m,n), ldam,
				    &work[(m-m_start)*A.n+n*A.nb],
				    sequence, request);
            }
        }
        #pragma omp taskwait
#if 0
	printf("Inspecting work...\n");
	for (int i=0; i<A.n; i++) {
	    printf ("R%d\t", i);
	    for (int j=0; j<ldwork; j++) {
		if (work[i+j*A.n]!=0) printf("%.2f\t", work[i+j*A.n]);
		else printf("*\t");
	    }
	    printf("\n");
	}
#endif
        c = "i";
	workspace =
	    (plasma_complex64_t*)malloc(A.n*sizeof(plasma_complex64_t*));
        *value = dlange_(c, &A.n, &ldwork, work, &A.n, workspace);
	free(workspace);
        /* printf("%s:%d [%s] value = %.3f\n", __FILE__, __LINE__, __FUNCTION__, *value); */
        break;
    case PlasmaInfNorm:
        for (int n = 0; n < A.nt; n++ ) {
            int nvan = plasma_tile_nview(A, n);
	    int m_start = (imax(0, n*A.nb-A.ku)) / A.nb;
	    int m_end = (imin(A.m-1, (n+1)*A.nb+A.kl-1)) / A.nb;
	    ldwork = A.mb*A.mt; 
            for (int m = m_start; m <= m_end; m++ ) {
                int ldam = plasma_tile_mmain_band(A, m, n);
                int mvam = plasma_tile_mview(A, m);
                core_omp_zlange_aux(PlasmaInfNorm,
				    mvam, nvan,
				    A(m,n), ldam,
				    &work[m*A.mb+n*ldwork],
				    sequence, request);
            }
        }
        #pragma omp taskwait
#if 0
	printf("Inspecting work...\n");
	for (int i=0; i<ldwork; i++) {
	    printf ("R%d\t", i);
	    for (int j=0; j<A.nt; j++) {
		if (work[i+j*ldwork]!=0) printf("%.2f\t", work[i+j*ldwork]);
		else printf("*\t");
	    }
	    printf("\n");
	}
#endif
        c = "i";
	workspace = (double*)malloc(ldwork*sizeof(double));
	nwork = A.nt;
        *value = dlange_(c, &ldwork, &nwork, work, &ldwork, workspace);
	free(workspace);
        /* printf("%s:%d [%s] value = %.3f\n", __FILE__, __LINE__, __FUNCTION__, *value); */
        break;
    case PlasmaFrobeniusNorm:
	kut  = (A.ku+A.nb-1)/A.nb; // # of tiles in upper band (not including diagonal)
	klt  = (A.kl+A.nb-1)/A.nb;    // # of tiles in lower band (not including diagonal)
	ldwork = kut+klt+1;
	scale = work;
	sumsq = &work[ldwork*A.nt];
        for (int n = 0; n < A.nt; n++ ) {
            int nvan = plasma_tile_nview(A, n);
	    int m_start = (imax(0, n*A.nb-A.ku)) / A.nb;
	    int m_end = (imin(A.m-1, (n+1)*A.nb+A.kl-1)) / A.nb;

            for (int m = m_start; m <= m_end; m++ ) {
                int ldam = plasma_tile_mmain_band(A, m, n);
                int mvam = plasma_tile_mview(A, m);
                core_omp_zgessq(mvam, nvan,
				A(m,n), ldam,
				&scale[n*ldwork+m-m_start],
				&sumsq[n*ldwork+m-m_start],
				sequence, request);
            }
        }
        #pragma omp taskwait
#if 0
	printf("Inspecting work...\n");
	for (int i=0; i<ldwork; i++) {
	    printf ("R%d\t", i);
	    for (int j=0; j<A.nt; j++) {
		if (work[i+j*ldwork]!=0) printf("%.2f,%.2f\t", scale[i+j*ldwork], sumsq[i+j*ldwork]);
		else printf("*\t");
	    }
	    printf("\n");
	}
#endif
	core_omp_dgessq_aux(ldwork*A.nt, scale, sumsq,
			    value, sequence, request);
        break;
    default:
        assert(0);
    }
}
