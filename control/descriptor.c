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
int PLASMA_Desc_Create(plasma_desc_t **desc, void *mat, plasma_enum_t dtyp,
                       int mb, int nb, int bsiz, int lm, int ln, int i,
                       int j, int m, int n)
{
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }
    // Allocate the descriptor.
    *desc = (plasma_desc_t*)malloc(sizeof(plasma_desc_t));
    if (*desc == NULL) {
        plasma_error("malloc() failed");
        return PlasmaErrorOutOfMemory;
    }
    // Initialize the descriptor.
    **desc = plasma_desc_init(dtyp, mb, nb, bsiz, lm, ln, i, j, m, n);
    (**desc).mat = mat;
    int status = plasma_desc_check(*desc);
    if (status != PlasmaSuccess) {
        plasma_error("invalid descriptor");
        return status;
    }
    return PlasmaSuccess;
}

/******************************************************************************/
int PLASMA_Desc_Destroy(plasma_desc_t **desc)
{
    plasma_context_t *plasma;

    plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }
    if (*desc == NULL) {
        plasma_error("NULL descriptor");
        return PlasmaErrorNullParameter;
    }
    free(*desc);
    *desc = NULL;
    return PlasmaSuccess;
}

/******************************************************************************/
plasma_desc_t plasma_desc_init(plasma_enum_t dtyp, int mb, int nb, int bsiz,
                             int lm, int ln, int i, int j, int m, int n)
{
    plasma_desc_t desc;

    size_t A21 = (size_t)(lm - lm%mb) * (size_t)(ln - ln%nb);
    size_t A12 = (size_t)(     lm%mb) * (size_t)(ln - ln%nb);
    size_t A22 = (size_t)(lm - lm%mb) * (size_t)(     ln%nb);

    // matrix address
    desc.mat = NULL;
    desc.A21 = A21;
    desc.A12 = A12 + desc.A21;
    desc.A22 = A22 + desc.A12;

    // matrix properties
    desc.dtyp = dtyp;
    desc.mb = mb;
    desc.nb = nb;
    desc.bsiz = bsiz;

    // large matrix parameters
    desc.lm = lm;
    desc.ln = ln;

    // large matrix derived parameters
    desc.lm1 = (lm/mb);
    desc.ln1 = (ln/nb);
    desc.lmt = (lm%mb == 0) ? (lm/mb) : (lm/mb+1);
    desc.lnt = (ln%nb == 0) ? (ln/nb) : (ln/nb+1);

    // submatrix parameters
    desc.i = i;
    desc.j = j;
    desc.m = m;
    desc.n = n;

    // submatrix derived parameters
    desc.mt = (m == 0) ? 0 : (i+m-1)/mb - i/mb + 1;
    desc.nt = (n == 0) ? 0 : (j+n-1)/nb - j/nb + 1;

    return desc;
}

/******************************************************************************/
plasma_desc_t plasma_desc_band_init(plasma_enum_t dtyp, plasma_enum_t uplo,
                                  int mb, int nb, int bsiz,
                                  int lm, int ln, int i, int j, int m, int n,
                                  int kl, int ku)
{
    plasma_desc_t desc;
    // init params for a general matrix
    desc = plasma_desc_init(dtyp, mb, nb, bsiz, lm, ln, i, j, m, n);

    // init params for band matrix
    // * band width
    desc.kl = kl;
    desc.ku = ku;

    // * number of tiles within band, 1+ for diagonal
    if (uplo == PlasmaFull) {
        desc.klt = 1+(i+kl + mb-1)/mb - i/mb;
        desc.kut = 1+(i+ku+kl + nb-1)/nb - i/nb;
    }
    else if (uplo == PlasmaUpper) {
        desc.klt = 1;
        desc.kut = 1+(i+ku + nb-1)/nb - i/nb;
    }
    else {
        desc.klt = 1+(i+kl + mb-1)/mb - i/mb;
        desc.kut = 1;
    }

    return desc;
}

/******************************************************************************/
plasma_desc_t plasma_desc_submatrix(plasma_desc_t descA, int i, int j, int m, int n)
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

/******************************************************************************/
int plasma_desc_check(plasma_desc_t *desc)
{
    if (desc == NULL) {
        plasma_error("NULL descriptor");
        return PlasmaErrorIllegalValue;
    }
    if (desc->mat == NULL) {
        plasma_error("NULL matrix pointer");
        return PlasmaErrorNullParameter;
    }
    if (desc->dtyp != PlasmaRealFloat &&
        desc->dtyp != PlasmaRealDouble &&
        desc->dtyp != PlasmaComplexFloat &&
        desc->dtyp != PlasmaComplexDouble  ) {
        plasma_error("invalid matrix type");
        return PlasmaErrorIllegalValue;
    }
    if (desc->mb <= 0 || desc->nb <= 0) {
        plasma_error("negative tile dimension");
        return PlasmaErrorIllegalValue;
    }
    if (desc->bsiz < desc->mb*desc->nb) {
        plasma_error("invalid tile memory size");
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
int plasma_desc_band_check(plasma_enum_t uplo, plasma_desc_t *desc)
{
    if (desc == NULL) {
        plasma_error("NULL descriptor");
        return PlasmaErrorIllegalValue;
    }
    if (desc->mat == NULL) {
        plasma_error("NULL matrix pointer");
        return PlasmaErrorNullParameter;
    }
    if (desc->dtyp != PlasmaRealFloat &&
        desc->dtyp != PlasmaRealDouble &&
        desc->dtyp != PlasmaComplexFloat &&
        desc->dtyp != PlasmaComplexDouble  ) {
        plasma_error("invalid matrix type");
        return PlasmaErrorIllegalValue;
    }
    if (desc->mb <= 0 || desc->nb <= 0) {
        plasma_error("negative tile dimension");
        return PlasmaErrorIllegalValue;
    }
    if (desc->bsiz < desc->mb*desc->nb) {
        plasma_error("invalid tile memory size");
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
    if ((uplo == PlasmaFull &&
         desc->lm < desc->mb*((2*desc->kl+desc->ku+desc->mb)/desc->mb)) ||
        (uplo == PlasmaUpper &&
         desc->lm < desc->mb*((desc->ku + desc->mb)/desc->mb)) ||
        (uplo == PlasmaUpper &&
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
int plasma_desc_mat_alloc(plasma_desc_t *desc)
{
    size_t size = (size_t)desc->lm * desc->ln * plasma_element_size(desc->dtyp);

    if ((desc->mat = malloc(size)) == NULL) {
        plasma_error("malloc() failed");
        return PlasmaErrorOutOfMemory;
    }
    return PlasmaSuccess;
}

/******************************************************************************/
int plasma_desc_mat_free(plasma_desc_t *desc)
{
    if (desc->mat != NULL) {
        free(desc->mat);
        desc->mat = NULL;
    }
    return PlasmaSuccess;
}
