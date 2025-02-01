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
#include "plasma.h"

#include <stdio.h>

#define COMPLEX
#undef REAL

#define A( i, j ) A[ (i) + (j)*lda ]

//------------------------------------------------------------------------------
void plasma_zprint_matrix(
    const char* label, int m, int n, plasma_complex64_t* A, int lda )
{
    const plasma_complex64_t zero = 0;
    printf( "%s = [\n", label );
    for (int i = 0; i < m; ++i) {
        printf( "  " );
        for (int j = 0; j < n; ++j) {
            plasma_complex64_t Aij = A( i, j );
            #ifdef COMPLEX
                if (Aij == zero) {
                    printf( "     0.0                " );
                }
                else {
                    printf( "  %9.4f + %9.4fi", creal( Aij ), cimag( Aij ) );
                }
            #else
                if (Aij == zero) {
                    printf( "     0.0   " );
                }
                else {
                    printf( "  %9.4f", Aij );
                }
            #endif
        }
        printf( "\n" );
    }
    printf( "];\n" );
}

//------------------------------------------------------------------------------
void plasma_zprint_vector(
    const char* label, int n, plasma_complex64_t* x, int incx )
{
    plasma_complex64_t zero = 0;
    printf( "%s = [\n  ", label );
    for (int i = 0; i < n; ++i) {
        plasma_complex64_t xi = x[ i*incx ];
        #ifdef COMPLEX
            if (xi == zero) {
                printf( "     0.0                " );
            }
            else {
                printf( "  %9.4f + %9.4fi", creal( xi ), cimag( xi ) );
            }
        #else
            if (xi == zero) {
                printf( "     0.0   " );
            }
            else {
                printf( "  %9.4f", xi );
            }
        #endif
    }
    printf( "\n];\n" );
}
