/**
 *
 * @file context.h
 *
 *  PLASMA control routines.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver.
 *
 * @version 3.0.0
 * @author Jakub Kurzak
 * @date 2016-01-01
 *
 **/

#include "context.h"
#include "descriptor.h"
#include "internal.h"

/******************************************************************************/
int PLASMA_Desc_Create(PLASMA_desc **desc, void *mat, PLASMA_enum dtyp,
                       int mb, int nb, int bsiz, int lm, int ln, int i,
                       int j, int m, int n)
{
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA_Desc_Create", "PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }
    /* Allocate memory and initialize the descriptor */
    *desc = (PLASMA_desc*)malloc(sizeof(PLASMA_desc));
    if (*desc == NULL) {
        plasma_error("PLASMA_Desc_Create", "malloc() failed");
        return PLASMA_ERR_OUT_OF_RESOURCES;
    }
    **desc = plasma_desc_init(dtyp, mb, nb, bsiz, lm, ln, i, j, m, n);
    (**desc).mat = mat;
    int status = plasma_desc_check(*desc);
    if (status != PLASMA_SUCCESS) {
        plasma_error("PLASMA_Desc_Create", "invalid descriptor");
        return status;
    }
    return PLASMA_SUCCESS;
}

/******************************************************************************/
int PLASMA_Desc_Destroy(PLASMA_desc **desc)
{
    plasma_context_t *plasma;

    plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA_Desc_Destroy", "PLASMA not initialized");
        return PLASMA_ERR_NOT_INITIALIZED;
    }
    if (*desc == NULL) {
        plasma_error("PLASMA_Desc_Destroy",
                     "attempting to destroy a NULL descriptor");
        return PLASMA_ERR_UNALLOCATED;
    }
    free(*desc);
    *desc = NULL;
    return PLASMA_SUCCESS;
}

/******************************************************************************/
PLASMA_desc plasma_desc_init(PLASMA_enum dtyp, int mb, int nb, int bsiz,
                             int lm, int ln, int i, int j, int m, int n)
{
    PLASMA_desc desc;

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
    desc.lmt = (lm%mb==0) ? (lm/mb) : (lm/mb+1);
    desc.lnt = (ln%nb==0) ? (ln/nb) : (ln/nb+1);

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
PLASMA_desc plasma_desc_submatrix(PLASMA_desc descA, int i, int j, int m, int n)
{
    if ((descA.i+i+m) > descA.lm)
        plasma_error("plasma_desc_submatrix", "rows out of bounds");

    if ((descA.j+j+n) > descA.ln)
        plasma_error("plasma_desc_submatrix", "columns out of bounds");

    PLASMA_desc descB = descA;
    int mb = descA.mb;
    int nb = descA.nb;

    // Submatrix parameters
    descB.i = descA.i + i;
    descB.j = descA.j + j;
    descB.m = m;
    descB.n = n;

    // Submatrix derived parameters
    descB.mt = (m == 0) ? 0 : (descB.i+m-1)/mb - descB.i/mb + 1;
    descB.nt = (n == 0) ? 0 : (descB.j+n-1)/nb - descB.j/nb + 1;

    return descB;
}

/******************************************************************************/
int plasma_desc_check(PLASMA_desc *desc)
{
    if (desc == NULL) {
        plasma_error("plasma_desc_check", "NULL descriptor");
        return PLASMA_ERR_NOT_INITIALIZED;
    }
    if (desc->mat == NULL) {
        plasma_error("plasma_desc_check", "NULL matrix pointer");
        return PLASMA_ERR_UNALLOCATED;
    }
    if (desc->dtyp != PlasmaRealFloat &&
        desc->dtyp != PlasmaRealDouble &&
        desc->dtyp != PlasmaComplexFloat &&
        desc->dtyp != PlasmaComplexDouble  ) {
        plasma_error("plasma_desc_check", "invalid matrix type");
        return PLASMA_ERR_ILLEGAL_VALUE;
    }
    if (desc->mb <= 0 || desc->nb <= 0) {
        plasma_error("plasma_desc_check", "negative tile dimension");
        return PLASMA_ERR_ILLEGAL_VALUE;
    }
    if (desc->bsiz < desc->mb*desc->nb) {
        plasma_error("plasma_desc_check",
                     "tile memory size smaller than the product of dimensions");
        return PLASMA_ERR_ILLEGAL_VALUE;
    }
    if ((desc->m < 0) || (desc->n < 0)) {
        plasma_error("plasma_desc_check", "negative matrix dimension");
        return PLASMA_ERR_ILLEGAL_VALUE;
    }
    if ((desc->lm < desc->m) || (desc->ln < desc->n)) {
        plasma_error("plasma_desc_check",
                     "matrix dimensions larger than leading dimensions");
        return PLASMA_ERR_ILLEGAL_VALUE;
    }
    if ((desc->i > 0 && desc->i >= desc->lm) ||
        (desc->j > 0 && desc->j >= desc->ln)) {
        plasma_error("plasma_desc_check",
                     "beginning of the matrix out of scope");
        return PLASMA_ERR_ILLEGAL_VALUE;
    }
    if (desc->i+desc->m > desc->lm || desc->j+desc->n > desc->ln) {
        plasma_error("plasma_desc_check", "submatrix out of scope");
        return PLASMA_ERR_ILLEGAL_VALUE;
    }
    if ((desc->i % desc->mb != 0) || (desc->j % desc->nb != 0)) {
        plasma_error("plasma_desc_check",
                     "submatrix has to start in the corner of a tile");
        return PLASMA_ERR_ILLEGAL_VALUE;
    }
    return PLASMA_SUCCESS;
}

/******************************************************************************/
int plasma_desc_mat_alloc(PLASMA_desc *desc)
{
    size_t size = (size_t)desc->lm * (size_t)desc->ln *
                  (size_t)plasma_element_size(desc->dtyp);

    if ((desc->mat = malloc(size)) == NULL) {
        plasma_error("plasma_desc_mat_alloc", "malloc() failed");
        return PLASMA_ERR_OUT_OF_RESOURCES;
    }
    return PLASMA_SUCCESS;
}

/******************************************************************************/
int plasma_desc_mat_free(PLASMA_desc *desc)
{
    if (desc->mat != NULL) {
        free(desc->mat);
        desc->mat = NULL;
    }
    return PLASMA_SUCCESS;
}
