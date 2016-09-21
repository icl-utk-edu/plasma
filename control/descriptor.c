/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#include <string.h>

#include "plasma_types.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"


/******************************************************************************/
int plasma_desc_general_create(plasma_enum_t precision, int mb, int nb,
                               int lm, int ln, int i, int j, int m, int n,
                               plasma_desc_t *desc)
{
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }
    // Initialize and check the descriptor.
    int retval = plasma_desc_init(precision, mb, nb, lm, ln, i, j, m, n, desc);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_init() failed");
        return retval;
    }
    retval = plasma_desc_check(desc);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_check() failed");
        return PlasmaErrorIllegalValue;
    }
    // Allocate the matrix.
    size_t size = (size_t)desc->lm*desc->ln*plasma_element_size(desc->precision);
    desc->matrix = malloc(size);
    if (desc->matrix == NULL) {
        plasma_error("malloc() failed");
        return PlasmaErrorOutOfMemory;
    }
    return PlasmaSuccess;
}

/******************************************************************************/
int plasma_desc_general_band_create(plasma_enum_t precision, plasma_enum_t uplo,
                                    int mb, int nb, int lm, int ln,
                                    int i, int j, int m, int n, int kl, int ku,
                                    plasma_desc_t *desc)
{
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }
    // Initialize and check the descriptor.
    int retval = plasma_desc_band_init(precision, uplo, mb, nb,
                                       lm, ln, i, j, m, n, kl, ku, desc);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_band_init() failed");
        return retval;
    }
    retval = plasma_desc_band_check(desc);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_band_check() failed");
        return PlasmaErrorIllegalValue;
    }
    // Allocate the matrix.
    size_t size = (size_t)desc->lm*desc->ln*plasma_element_size(desc->precision);
    desc->matrix = malloc(size);
    if (desc->matrix == NULL) {
        plasma_error("malloc() failed");
        return PlasmaErrorOutOfMemory;
    }
    return PlasmaSuccess;
}

/******************************************************************************/
int plasma_desc_create_for_function(const char *function_name, int m, int n,
                                    plasma_desc_t *desc)
{
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    // Compute the parameters of the descriptor.
    int nb = plasma->nb;
    int ib = plasma->ib;
    int mt = (m%nb == 0) ? (m/nb) : (m/nb+1);
    int nt = (n%nb == 0) ? (n/nb) : (n/nb+1);
    // nt should be doubled if tree-reduction QR is performed,
    // not implemented now

    // Set function-dependent variables.
    int mb;
    plasma_enum_t precision;
    if      (strcmp(function_name, "zgeqrf") == 0 ||
             strcmp(function_name, "zgelqf") == 0 ||
             strcmp(function_name, "zgels")  == 0) {

        mb = ib;
        precision = PlasmaComplexDouble;
    }
    else if (strcmp(function_name, "cgeqrf") == 0 ||
             strcmp(function_name, "cgelqf") == 0 ||
             strcmp(function_name, "cgels")  == 0) {

        mb = ib;
        precision = PlasmaComplexFloat;
    }
    else if (strcmp(function_name, "dgeqrf") == 0 ||
             strcmp(function_name, "dgelqf") == 0 ||
             strcmp(function_name, "dgels")  == 0) {

        mb = ib;
        precision = PlasmaRealDouble;
    }
    else if (strcmp(function_name, "sgeqrf") == 0 ||
             strcmp(function_name, "sgelqf") == 0 ||
             strcmp(function_name, "sgels")  == 0) {

        mb = ib;
        precision = PlasmaRealFloat;
    }
    else {
        plasma_error(strcat("Unknown routine ",function_name));
        return PlasmaErrorIllegalValue;
    }

    // Create the descriptor using the standard function.
    int retval = plasma_desc_general_create(precision, mb, nb, mt*mb, nt*nb,
                                            0, 0, mt*mb, nt*nb, desc);
    return retval;
}

/******************************************************************************/
int plasma_desc_destroy(plasma_desc_t *desc)
{
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }
    free(desc->matrix);
    return PlasmaSuccess;
}

/******************************************************************************/
int plasma_desc_init(plasma_enum_t precision, int mb, int nb, int lm, int ln,
                     int i, int j, int m, int n, plasma_desc_t *desc)
{
    size_t A21 = (size_t)(lm - lm%mb) * (size_t)(ln - ln%nb);
    size_t A12 = (size_t)(     lm%mb) * (size_t)(ln - ln%nb);
    size_t A22 = (size_t)(lm - lm%mb) * (size_t)(     ln%nb);

    // matrix address
    desc->matrix = NULL;
    desc->A21 = A21;
    desc->A12 = A12 + desc->A21;
    desc->A22 = A22 + desc->A12;

    // matrix properties
    desc->type = PlasmaGeneral;
    desc->precision = precision;
    desc->mb = mb;
    desc->nb = nb;

    // large matrix parameters
    desc->lm = lm;
    desc->ln = ln;

    // large matrix derived parameters
    desc->lmt = (lm%mb == 0) ? (lm/mb) : (lm/mb+1);
    desc->lnt = (ln%nb == 0) ? (ln/nb) : (ln/nb+1);

    // submatrix parameters
    desc->i = i;
    desc->j = j;
    desc->m = m;
    desc->n = n;

    // submatrix derived parameters
    desc->mt = (m == 0) ? 0 : (i+m-1)/mb - i/mb + 1;
    desc->nt = (n == 0) ? 0 : (j+n-1)/nb - j/nb + 1;

    return PlasmaSuccess;
}

/******************************************************************************/
int plasma_desc_band_init(plasma_enum_t precision, plasma_enum_t uplo,
                          int mb, int nb, int lm, int ln, int i, int j,
                          int m, int n, int kl, int ku, plasma_desc_t *desc)
{
    // Init parameters for a general matrix.
    int retval = plasma_desc_init(precision, mb, nb, lm, ln, i, j, m, n, desc);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_init() failed");
        return retval;
    }
    // Change matrix type to band.
    desc->type = PlasmaGeneralBand;
    desc->uplo = uplo;

    // init params for band matrix
    // * band width
    desc->kl = kl;
    desc->ku = ku;

    // * number of tiles within band, 1+ for diagonal
    if (uplo == PlasmaGeneral) {
        desc->klt = 1+(i+kl + mb-1)/mb - i/mb;
        desc->kut = 1+(i+ku+kl + nb-1)/nb - i/nb;
    }
    else if (uplo == PlasmaUpper) {
        desc->klt = 1;
        desc->kut = 1+(i+ku + nb-1)/nb - i/nb;
    }
    else {
        desc->klt = 1+(i+kl + mb-1)/mb - i/mb;
        desc->kut = 1;
    }
    return PlasmaSuccess;
}

/******************************************************************************/
int plasma_desc_check(plasma_desc_t *desc)
{
    if (desc == NULL) {
        plasma_error("NULL descriptor");
        return PlasmaErrorNullParameter;
    }
    else if (desc->type == PlasmaGeneral) {
        return plasma_desc_full_check(desc);
    }
    else if (desc->type == PlasmaGeneralBand) {
        return plasma_desc_band_check(desc);
    }
    else {
        plasma_error("invalid matrix type");
        return PlasmaErrorIllegalValue;
    }
}

/******************************************************************************/
int plasma_desc_full_check(plasma_desc_t *desc)
{
    if (desc == NULL) {
        plasma_error("NULL descriptor");
        return PlasmaErrorIllegalValue;
    }
    if (desc->precision != PlasmaRealFloat &&
        desc->precision != PlasmaRealDouble &&
        desc->precision != PlasmaComplexFloat &&
        desc->precision != PlasmaComplexDouble  ) {
        plasma_error("invalid matrix type");
        return PlasmaErrorIllegalValue;
    }
    if (desc->mb <= 0 || desc->nb <= 0) {
        plasma_error("negative tile dimension");
        return PlasmaErrorIllegalValue;
    }
    if ((desc->m < 0) || (desc->n < 0)) {
        plasma_error("negative matrix dimension");
        return PlasmaErrorIllegalValue;
    }
    if ((desc->lm < desc->m) || (desc->ln < desc->n)) {
        plasma_error("invalid leading dimensions");
        return PlasmaErrorIllegalValue;
    }
    if ((desc->i > 0 && desc->i >= desc->lm) ||
        (desc->j > 0 && desc->j >= desc->ln)) {
        plasma_error("beginning of the matrix out of bounds");
        return PlasmaErrorIllegalValue;
    }
    if (desc->i+desc->m > desc->lm || desc->j+desc->n > desc->ln) {
        plasma_error("submatrix out of bounds");
        return PlasmaErrorIllegalValue;
    }
    if ((desc->i % desc->mb != 0) || (desc->j % desc->nb != 0)) {
        plasma_error("submatrix not aligned to a tile");
        return PlasmaErrorIllegalValue;
    }
    return PlasmaSuccess;
}

/******************************************************************************/
int plasma_desc_band_check(plasma_desc_t *desc)
{
    if (desc == NULL) {
        plasma_error("NULL descriptor");
        return PlasmaErrorIllegalValue;
    }
    if (desc->precision != PlasmaRealFloat &&
        desc->precision != PlasmaRealDouble &&
        desc->precision != PlasmaComplexFloat &&
        desc->precision != PlasmaComplexDouble  ) {
        plasma_error("invalid matrix type");
        return PlasmaErrorIllegalValue;
    }
    if (desc->mb <= 0 || desc->nb <= 0) {
        plasma_error("negative tile dimension");
        return PlasmaErrorIllegalValue;
    }
    if ((desc->m < 0) || (desc->n < 0)) {
        plasma_error("negative matrix dimension");
        return PlasmaErrorIllegalValue;
    }
    if (desc->ln < desc->n) {
        plasma_error("invalid leading column dimensions");
        return PlasmaErrorIllegalValue;
    }
    if ((desc->uplo == PlasmaGeneral &&
         desc->lm < desc->mb*((2*desc->kl+desc->ku+desc->mb)/desc->mb)) ||
        (desc->uplo == PlasmaUpper &&
         desc->lm < desc->mb*((desc->ku + desc->mb)/desc->mb)) ||
        (desc->uplo == PlasmaUpper &&
         desc->lm < desc->mb*((desc->kl + desc->mb)/desc->mb))) {
        plasma_error("invalid leading row dimensions");
        return PlasmaErrorIllegalValue;
    }
    if ((desc->i > 0 && desc->i >= desc->lm) ||
        (desc->j > 0 && desc->j >= desc->ln)) {
        plasma_error("beginning of the matrix out of bounds");
        return PlasmaErrorIllegalValue;
    }
    if (desc->j+desc->n > desc->ln) {
        plasma_error("submatrix out of bounds");
        return PlasmaErrorIllegalValue;
    }
    if ((desc->i % desc->mb != 0) || (desc->j % desc->nb != 0)) {
        plasma_error("submatrix not aligned to a tile");
        return PlasmaErrorIllegalValue;
    }

    if (desc->kl+1 > desc->m || desc->ku+1 > desc->n) {
        plasma_error("band width larger than matrix dimension");
        return PlasmaErrorIllegalValue;
    }
    return PlasmaSuccess;
}

/******************************************************************************/
plasma_desc_t plasma_desc_view(plasma_desc_t descA, int i, int j, int m, int n)
{
    if ((descA.i+i+m) > descA.lm)
        plasma_error("rows out of bounds");

    if ((descA.j+j+n) > descA.ln)
        plasma_error("columns out of bounds");

    plasma_desc_t descB = descA;
    int mb = descA.mb;
    int nb = descA.nb;

    // submatrix parameters
    descB.i = descA.i + i;
    descB.j = descA.j + j;
    descB.m = m;
    descB.n = n;

    // submatrix derived parameters
    descB.mt = (m == 0) ? 0 : (descB.i+m-1)/mb - descB.i/mb + 1;
    descB.nt = (n == 0) ? 0 : (descB.j+n-1)/nb - descB.j/nb + 1;

    return descB;
}
