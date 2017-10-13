/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#include "plasma_types.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"

/******************************************************************************/
int plasma_desc_general_create(plasma_enum_t precision, int mb, int nb,
                               int lm, int ln, int i, int j, int m, int n,
                               plasma_desc_t *A)
{
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }
    // Initialize the descriptor.
    int retval = plasma_desc_general_init(precision, NULL, mb, nb,
                                          lm, ln, i, j, m, n, A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_init() failed");
        return retval;
    }
    // Check the descriptor.
    retval = plasma_desc_check(*A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_check() failed");
        return PlasmaErrorIllegalValue;
    }
    // Allocate the matrix.
    size_t size = (size_t)A->gm*A->gn*
                  plasma_element_size(A->precision);
    A->matrix = malloc(size);
    if (A->matrix == NULL) {
        plasma_error("malloc() failed");
        return PlasmaErrorOutOfMemory;
    }
    return PlasmaSuccess;
}

/******************************************************************************/
int plasma_desc_general_band_create(plasma_enum_t precision, plasma_enum_t uplo,
                                    int mb, int nb, int lm, int ln,
                                    int i, int j, int m, int n, int kl, int ku,
                                    plasma_desc_t *A)
{
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }
    // Initialize the descriptor.
    int retval = plasma_desc_general_band_init(precision, uplo, NULL, mb, nb,
                                               lm, ln, i, j, m, n, kl, ku, A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_band_init() failed");
        return retval;
    }
    // Check the descriptor.
    retval = plasma_desc_check(*A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_check() failed");
        return PlasmaErrorIllegalValue;
    }
    // Allocate the matrix.
    size_t size = (size_t)A->gm*A->gn*
                  plasma_element_size(A->precision);
    A->matrix = malloc(size);
    if (A->matrix == NULL) {
        plasma_error("malloc() failed");
        return PlasmaErrorOutOfMemory;
    }
    return PlasmaSuccess;
}

/******************************************************************************/
int plasma_desc_triangular_create(plasma_enum_t precision, plasma_enum_t uplo, int mb, int nb,
                                  int lm, int ln, int i, int j, int m, int n,
                                  plasma_desc_t *A)
{
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }
    // Initialize the descriptor.
    int retval = plasma_desc_triangular_init(precision, uplo, NULL, mb, nb,
                                             lm, ln, i, j, m, n, A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_init() failed");
        return retval;
    }
    // Check the descriptor.
    retval = plasma_desc_check(*A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_check() failed");
        return PlasmaErrorIllegalValue;
    }
    // Allocate the matrix.
    int lm1 = lm/mb;
    int ln1 = ln/nb;
    int mnt = (ln1*(1+lm1))/2;
    size_t size = (size_t)(mnt*mb*nb + (lm * (ln%nb)))*
                  plasma_element_size(A->precision);
    A->matrix = malloc(size);
    if (A->matrix == NULL) {
        plasma_error("malloc() failed");
        return PlasmaErrorOutOfMemory;
    }
    return PlasmaSuccess;
}

/******************************************************************************/
int plasma_desc_destroy(plasma_desc_t *A)
{
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }
    free(A->matrix);
    return PlasmaSuccess;
}

/******************************************************************************/
int plasma_desc_general_init(plasma_enum_t precision, void *matrix,
                             int mb, int nb, int lm, int ln, int i, int j,
                             int m, int n, plasma_desc_t *A)
{
    // type and precision
    A->type = PlasmaGeneral;
    A->precision = precision;

    // pointer and offsets
    A->matrix = matrix;
    A->A21 = (size_t)(lm - lm%mb) * (ln - ln%nb);
    A->A12 = (size_t)(     lm%mb) * (ln - ln%nb) + A->A21;
    A->A22 = (size_t)(lm - lm%mb) * (     ln%nb) + A->A12;

    // tile parameters
    A->mb = mb;
    A->nb = nb;

    // main matrix parameters
    A->gm = lm;
    A->gn = ln;

    A->gmt = (lm%mb == 0) ? (lm/mb) : (lm/mb+1);
    A->gnt = (ln%nb == 0) ? (ln/nb) : (ln/nb+1);

    // submatrix parameters
    A->i = i;
    A->j = j;
    A->m = m;
    A->n = n;

    A->mt = (m == 0) ? 0 : (i+m-1)/mb - i/mb + 1;
    A->nt = (n == 0) ? 0 : (j+n-1)/nb - j/nb + 1;

    // band parameters
    A->kl = m-1;
    A->ku = n-1;
    A->klt = A->mt;
    A->kut = A->nt;

    return PlasmaSuccess;
}

/******************************************************************************/
int plasma_desc_general_band_init(plasma_enum_t precision, plasma_enum_t uplo,
                                  void *matrix, int mb, int nb, int lm, int ln,
                                  int i, int j, int m, int n, int kl, int ku,
                                  plasma_desc_t *A)
{
    // Init parameters for a general matrix.
    int retval = plasma_desc_general_init(precision, matrix, mb, nb,
                                          lm, ln, i, j, m, n, A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_init() failed");
        return retval;
    }
    // Change matrix type to band.
    A->type = PlasmaGeneralBand;
    A->uplo = uplo;

    // Initialize band matrix parameters.
    // bandwidth
    A->kl = kl;
    A->ku = ku;

    // number of tiles within band, 1+ for diagonal
    if (uplo == PlasmaGeneral) {
        A->klt = 1+(i+kl + mb-1)/mb - i/mb;
        A->kut = 1+(i+ku+kl + nb-1)/nb - i/nb;
    }
    else if (uplo == PlasmaUpper) {
        A->klt = 1;
        A->kut = 1+(i+ku + nb-1)/nb - i/nb;
    }
    else {
        A->klt = 1+(i+kl + mb-1)/mb - i/mb;
        A->kut = 1;
    }
    return PlasmaSuccess;
}

/******************************************************************************/
int plasma_desc_triangular_init(plasma_enum_t precision, plasma_enum_t uplo, void *matrix,
                                int mb, int nb, int lm, int ln, int i, int j,
                                int m, int n, plasma_desc_t *A)
{
    // only for square matrix..
    if (lm != ln) {
        plasma_error("invalid lm or ln");
    }
    // type and precision
    A->type = uplo;
    A->precision = precision;

    // pointer and offsets
    int lm1 = lm/mb;
    int ln1 = ln/nb;
    int mnt = (ln1*(1+lm1))/2;
    A->matrix = matrix;
    A->A21 = (size_t)(mb * nb) * mnt; // only for PlasmaLower
    A->A12 = (size_t)(mb * nb) * mnt; // only for PlasmaUpper
    A->A22 = (size_t)(lm - lm%mb) * (ln%nb) + A->A12;

    // tile parameters
    A->mb = mb;
    A->nb = nb;

    // main matrix parameters
    A->gm = lm;
    A->gn = ln;

    A->gmt = (lm%mb == 0) ? (lm/mb) : (lm/mb+1);
    A->gnt = (ln%nb == 0) ? (ln/nb) : (ln/nb+1);

    // submatrix parameters
    A->i = i;
    A->j = j;
    A->m = m;
    A->n = n;

    A->mt = (m == 0) ? 0 : (i+m-1)/mb - i/mb + 1;
    A->nt = (n == 0) ? 0 : (j+n-1)/nb - j/nb + 1;

    // band parameters
    A->kl = m-1;
    A->ku = n-1;
    A->klt = A->mt;
    A->kut = A->nt;

    return PlasmaSuccess;
}

/******************************************************************************/
int plasma_desc_check(plasma_desc_t A)
{
    if (A.type == PlasmaGeneral ||
        A.type == PlasmaUpper ||
        A.type == PlasmaLower) {
        return plasma_desc_general_check(A);
    }
    else if (A.type == PlasmaGeneralBand) {
        return plasma_desc_general_band_check(A);
    }
    else {
        plasma_error("invalid matrix type");
        return PlasmaErrorIllegalValue;
    }
}

/******************************************************************************/
int plasma_desc_general_check(plasma_desc_t A)
{
    if (A.precision != PlasmaRealFloat &&
        A.precision != PlasmaRealDouble &&
        A.precision != PlasmaComplexFloat &&
        A.precision != PlasmaComplexDouble  ) {
        plasma_error("invalid matrix type");
        return PlasmaErrorIllegalValue;
    }
    if (A.mb <= 0 || A.nb <= 0) {
        plasma_error("negative tile dimension");
        return PlasmaErrorIllegalValue;
    }
    if ((A.m < 0) || (A.n < 0)) {
        plasma_error("negative matrix dimension");
        return PlasmaErrorIllegalValue;
    }
    if ((A.gm < A.m) || (A.gn < A.n)) {
        plasma_error("invalid leading dimensions");
        return PlasmaErrorIllegalValue;
    }
    if ((A.i > 0 && A.i >= A.gm) ||
        (A.j > 0 && A.j >= A.gn)) {
        plasma_error("beginning of the matrix out of bounds");
        return PlasmaErrorIllegalValue;
    }
    if (A.i+A.m > A.gm || A.j+A.n > A.gn) {
        plasma_error("submatrix out of bounds");
        return PlasmaErrorIllegalValue;
    }
    if ((A.i % A.mb != 0) || (A.j % A.nb != 0)) {
        plasma_error("submatrix not aligned to a tile");
        return PlasmaErrorIllegalValue;
    }
    return PlasmaSuccess;
}

/******************************************************************************/
int plasma_desc_general_band_check(plasma_desc_t A)
{
    if (A.precision != PlasmaRealFloat &&
        A.precision != PlasmaRealDouble &&
        A.precision != PlasmaComplexFloat &&
        A.precision != PlasmaComplexDouble  ) {
        plasma_error("invalid matrix type");
        return PlasmaErrorIllegalValue;
    }
    if (A.mb <= 0 || A.nb <= 0) {
        plasma_error("negative tile dimension");
        return PlasmaErrorIllegalValue;
    }
    if ((A.m < 0) || (A.n < 0)) {
        plasma_error("negative matrix dimension");
        return PlasmaErrorIllegalValue;
    }
    if (A.gn < A.n) {
        plasma_error("invalid leading column dimensions");
        return PlasmaErrorIllegalValue;
    }
    if ((A.uplo == PlasmaGeneral &&
         A.gm < A.mb*((2*A.kl+A.ku+A.mb)/A.mb)) ||
        (A.uplo == PlasmaUpper &&
         A.gm < A.mb*((A.ku + A.mb)/A.mb)) ||
        (A.uplo == PlasmaUpper &&
         A.gm < A.mb*((A.kl + A.mb)/A.mb))) {
        plasma_error("invalid leading row dimensions");
        return PlasmaErrorIllegalValue;
    }
    if ((A.i > 0 && A.i >= A.gm) ||
        (A.j > 0 && A.j >= A.gn)) {
        plasma_error("beginning of the matrix out of bounds");
        return PlasmaErrorIllegalValue;
    }
    if (A.j+A.n > A.gn) {
        plasma_error("submatrix out of bounds");
        return PlasmaErrorIllegalValue;
    }
    if ((A.i % A.mb != 0) || (A.j % A.nb != 0)) {
        plasma_error("submatrix not aligned to a tile");
        return PlasmaErrorIllegalValue;
    }

    if (A.kl+1 > A.m || A.ku+1 > A.n) {
        plasma_error("band width larger than matrix dimension");
        return PlasmaErrorIllegalValue;
    }
    return PlasmaSuccess;
}

/******************************************************************************/
plasma_desc_t plasma_desc_view(plasma_desc_t A, int i, int j, int m, int n)
{
    if ((A.i+i+m) > A.gm)
        plasma_fatal_error("rows out of bound");

    if ((A.j+j+n) > A.gn)
        plasma_fatal_error("columns out of bound");

    plasma_desc_t B = A;
    int mb = A.mb;
    int nb = A.nb;

    // submatrix parameters
    B.i = A.i + i;
    B.j = A.j + j;
    B.m = m;
    B.n = n;

    // submatrix derived parameters
    B.mt = (m == 0) ? 0 : (B.i+m-1)/mb - B.i/mb + 1;
    B.nt = (n == 0) ? 0 : (B.j+n-1)/nb - B.j/nb + 1;

    return B;
}

/******************************************************************************/
int plasma_descT_create(plasma_desc_t A, int ib, plasma_enum_t householder_mode,
                        plasma_desc_t *T)
{
    // T uses tiles ib x nb, typically, ib < nb, and these tiles are
    // rectangular. This dimension is the same for QR and LQ factorizations.
    int mb = ib;
    int nb = A.nb;

    // Number of tile rows and columns in T is the same as for T.
    int mt = A.mt;
    int nt = A.nt;
    // nt is doubled for tree-reduction QR and LQ
    if (householder_mode == PlasmaTreeHouseholder) {
        nt = 2*nt;
    }

    // Dimension of the matrix as whole multiples of the tiles.
    int m = mt*mb;
    int n = nt*nb;

    // Create the descriptor using the standard function.
    int retval = plasma_desc_general_create(A.precision, mb, nb, m, n,
                                            0, 0, m, n, T);
    return retval;
}
