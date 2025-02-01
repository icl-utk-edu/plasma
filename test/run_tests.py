#!/usr/bin/env python3
#
# Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
#
# Example usage:
# help
#     ./run_tests.py -h
#
# run everything with default sizes
# output is redirected; summary information is printed on stderr
#     ./run_tests.py > output.txt
#
# run LU (gesv, getrf, getri, ...), Cholesky (posv, potrf, potri, ...)
# with single, double and default sizes
#     ./run_tests.py --lu --chol --type s,d
#
# run getrf, potrf with small, medium sizes
#     ./run_tests.py -s -m getrf potrf

from __future__ import print_function

import sys
import os
import re
import argparse
import subprocess
import xml.etree.ElementTree as ET
import io
import time

#-------------------------------------------------------------------------------
# command line arguments
parser = argparse.ArgumentParser()

group_test = parser.add_argument_group( 'test' )
group_test.add_argument( '-t', '--test', action='store',
    help='test command to run, e.g., --test "mpirun -np 4 ./tester"; default "%(default)s"',
    default='./plasmatest' )
group_test.add_argument( '--xml', help='generate report.xml for jenkins' )
group_test.add_argument( '--dry-run', action='store_true', help='print commands, but do not execute them' )
group_test.add_argument( '--start',   action='store', help='routine to start with, helpful for restarting', default='' )
group_test.add_argument( '-x', '--exclude', action='append', help='routines to exclude; repeatable', default=[] )
group_test.add_argument( '--timeout', action='store', help='timeout in seconds for each routine', type=float )

group_size = parser.add_argument_group( 'matrix dimensions (default is medium)' )
group_size.add_argument( '--quick',  action='store_true', help='run quick "sanity check" of few, small tests' )
group_size.add_argument( '--xsmall', action='store_true', help='run x-small tests' )
group_size.add_argument( '--small',  action='store_true', help='run small tests' )
group_size.add_argument( '--medium', action='store_true', help='run medium tests' )
group_size.add_argument( '--large',  action='store_true', help='run large tests' )
group_size.add_argument( '--square', action='store_true', help='run square (m = n = k) tests', default=False )
group_size.add_argument( '--tall',   action='store_true', help='run tall (m > n) tests', default=False )
group_size.add_argument( '--wide',   action='store_true', help='run wide (m < n) tests', default=False )
group_size.add_argument( '--mnk',    action='store_true', help='run tests with m, n, k all different', default=False )
group_size.add_argument( '--dim',    action='store',      help='explicitly specify size', default='' )

group_cat = parser.add_argument_group( 'category (default is all)' )
categories = [
    #group_cat.add_argument( '--blas1',         action='store_true', help='run Level 1 BLAS tests' ),
    #group_cat.add_argument( '--blas2',         action='store_true', help='run Level 2 BLAS tests' ),
    group_cat.add_argument( '--blas3',         action='store_true', help='run Level 3 BLAS tests' ),
    group_cat.add_argument( '--lu',            action='store_true', help='run LU tests' ),
    group_cat.add_argument( '--chol',          action='store_true', help='run Cholesky tests' ),
    group_cat.add_argument( '--sysv',          action='store_true', help='run symmetric indefinite (Aasen) tests' ),
    group_cat.add_argument( '--hesv',          action='store_true', help='run Hermitian indefinite (Aasen) tests' ),
    group_cat.add_argument( '--least-squares', action='store_true', help='run least squares tests' ),
    group_cat.add_argument( '--qr',            action='store_true', help='run QR tests' ),
    group_cat.add_argument( '--lq',            action='store_true', help='run LQ tests' ),
    group_cat.add_argument( '--ql',            action='store_true', help='run QL tests' ),
    group_cat.add_argument( '--rq',            action='store_true', help='run RQ tests' ),
    group_cat.add_argument( '--heev',          action='store_true', help='run Hermitian/symmetric eigenvalue tests' ),
    group_cat.add_argument( '--hegv',          action='store_true', help='run generalized Hermitian/symmetric eigenvalue tests' ),
    group_cat.add_argument( '--geev',          action='store_true', help='run non-symmetric eigenvalue tests' ),
    group_cat.add_argument( '--svd',           action='store_true', help='run SVD tests' ),
    group_cat.add_argument( '--aux',           action='store_true', help='run auxiliary routine tests' ),
    group_cat.add_argument( '--norms',         action='store_true', help='run norm tests' ),
]
# map category objects to category names: ['lu', 'chol', ...]
categories = list( map( lambda x: x.dest, categories ) )

group_opt = parser.add_argument_group( 'options' )
# BLAS and LAPACK
# Empty defaults (check, ref, etc.) use the default in test.cc.
group_opt.add_argument( '--type',   action='store', help='default=%(default)s', default='s,d,c,z' )
#group_opt.add_argument( '--layout', action='store', help='default=%(default)s', default='c,r' )
group_opt.add_argument( '--transA', action='store', help='default=%(default)s', default='n,t,c' )
group_opt.add_argument( '--transB', action='store', help='default=%(default)s', default='n,t,c' )
group_opt.add_argument( '--trans',  action='store', help='default=%(default)s', default='n,t,c' )
group_opt.add_argument( '--geuplo', action='store', help='default=%(default)s', default='g,l,u' )
group_opt.add_argument( '--uplo',   action='store', help='default=%(default)s', default='l,u' )
group_opt.add_argument( '--diag',   action='store', help='default=%(default)s', default='n,u' )
group_opt.add_argument( '--side',   action='store', help='default=%(default)s', default='l,r' )
group_opt.add_argument( '--alpha',  action='store', help='default=%(default)s', default='' )
group_opt.add_argument( '--beta',   action='store', help='default=%(default)s', default='' )
group_opt.add_argument( '--incx',   action='store', help='default=%(default)s', default='1,2,-1,-2' )
group_opt.add_argument( '--incy',   action='store', help='default=%(default)s', default='1,2,-1,-2' )
#group_opt.add_argument( '--align',  action='store', help='default=%(default)s', default='32' )
group_opt.add_argument( '--check',  action='store', help='default=y', default='' )  # default in test.cc
group_opt.add_argument( '--ref',    action='store', help='default=y', default='' )  # default in test.cc
group_opt.add_argument( '--tol',    action='store', help='default=%(default)s', default='' )
group_opt.add_argument( '--verbose', action='store', help='default=0', default='' )  # default in test.cc
group_opt.add_argument( '--repeat', action='store', help='times to repeat each test', default='' )

# LAPACK only
#group_opt.add_argument( '--itype',  action='store', help='default=%(default)s', default='1,2,3' )
#group_opt.add_argument( '--factored', action='store', help='default=%(default)s', default='f,n,e' )
#group_opt.add_argument( '--equed',  action='store', help='default=%(default)s', default='n,r,c,b' )
#group_opt.add_argument( '--pivot',  action='store', help='default=%(default)s', default='v,t,b' )
#group_opt.add_argument( '--direction', action='store', help='default=%(default)s', default='f,b' )
#group_opt.add_argument( '--storev', action='store', help='default=%(default)s', default='c,r' )
group_opt.add_argument( '--norm',   action='store', help='default=%(default)s', default='m,o,i,f' )  # PLASMA supports only "o" spelling, not "1" spelling
#group_opt.add_argument( '--ijob',   action='store', help='default=%(default)s', default='0:5:1' )
#group_opt.add_argument( '--jobz',   action='store', help='default=%(default)s', default='n,v' )
#group_opt.add_argument( '--jobvl',  action='store', help='default=%(default)s', default='n,v' )
#group_opt.add_argument( '--jobvr',  action='store', help='default=%(default)s', default='n,v' )
#group_opt.add_argument( '--jobvs',  action='store', help='default=%(default)s', default='n,v' )
group_opt.add_argument( '--job',    action='store', help='default=%(default)s', default='n,s' )  # PLASMA supports only n,s
#group_opt.add_argument( '--jobu',   action='store', help='default=%(default)s', default='n,s,o,a' )
#group_opt.add_argument( '--jobvt',  action='store', help='default=%(default)s', default='n,s,o,a' )
#group_opt.add_argument( '--balanc', action='store', help='default=%(default)s', default='n,p,s,b' )
#group_opt.add_argument( '--sort',   action='store', help='default=%(default)s', default='n,s' )
#group_opt.add_argument( '--select', action='store', help='default=%(default)s', default='n,s' )
#group_opt.add_argument( '--sense',  action='store', help='default=%(default)s', default='n,e,v,b' )
#group_opt.add_argument( '--vect',   action='store', help='default=%(default)s', default='n,v' )
#group_opt.add_argument( '--l',      action='store', help='default=%(default)s', default='0,100' )
#group_opt.add_argument( '--ka',     action='store', help='default=%(default)s', default='20,40' )
#group_opt.add_argument( '--kb',     action='store', help='default=%(default)s', default='20,40' )
#group_opt.add_argument( '--kd',     action='store', help='default=%(default)s', default='20,40' )
group_opt.add_argument( '--kl',     action='store', help='default=%(default)s', default='20,40' )
group_opt.add_argument( '--ku',     action='store', help='default=%(default)s', default='20,40' )
#group_opt.add_argument( '--vl',     action='store', help='default=%(default)s', default='-inf,0' )
#group_opt.add_argument( '--vu',     action='store', help='default=%(default)s', default='inf' )
#group_opt.add_argument( '--il',     action='store', help='default=%(default)s', default='10' )
#group_opt.add_argument( '--iu',     action='store', help='default=%(default)s', default='-1,100' )
group_opt.add_argument( '--nb',     action='store', help='default=%(default)s', default='64' )
#group_opt.add_argument( '--matrixtype', action='store', help='default=%(default)s', default='g,l,u' )

# PLASMA specific

parser.add_argument( 'tests', nargs=argparse.REMAINDER )
opts = parser.parse_args()

for t in opts.tests:
    if (t.startswith('--')):
        print( 'Error: option', t, 'must come before any routine names' )
        print( 'usage:', sys.argv[0], '[options]', '[routines]' )
        print( '      ', sys.argv[0], '--help' )
        exit(1)

# by default, run medium sizes
if (not (opts.quick or opts.xsmall or opts.small or opts.medium or opts.large)):
    opts.medium = True

# by default, run all shapes
if (not (opts.square or opts.tall or opts.wide or opts.mnk)):
    opts.square = True
    opts.tall   = True
    opts.wide   = True
    opts.mnk    = True

# By default, or if specific test routines given, enable all categories
# to get whichever has the routines.
if (opts.tests or not any( map( lambda c: opts.__dict__[ c ], categories ))):
    for c in categories:
        opts.__dict__[ c ] = True

start_routine = opts.start

#-------------------------------------------------------------------------------
# parameters
# begin with space to ease concatenation

# if given, use explicit dim
dim = ' --dim=' + opts.dim if (opts.dim) else ''
n        = dim
tall     = dim
wide     = dim
mn       = dim
mnk      = dim
nk_tall  = dim
nk_wide  = dim
nk       = dim

if (not opts.dim):
    if (opts.quick):
        n        = ' --dim=100'
        tall     = ' --dim=100x50'  # 2:1
        wide     = ' --dim=50x100'  # 1:2
        mnk      = ' --dim=25x50x75'
        nk_tall  = ' --dim=1x100x50'  # 2:1
        nk_wide  = ' --dim=1x50x100'  # 1:2
        opts.incx  = '1,-1'
        #opts.incy  = '1,-1'
        opts.l     = '0,20,50'
        opts.nb    = '16'

    if (opts.xsmall):
        n       += ' --dim=10'
        tall    += ' --dim=20x10'
        wide    += ' --dim=10x20'
        mnk     += ' --dim=10x15x20 --dim=15x10x20' \
                +  ' --dim=10x20x15 --dim=15x20x10' \
                +  ' --dim=20x10x15 --dim=20x15x10'
        nk_tall += ' --dim=1x20x10'
        nk_wide += ' --dim=1x10x20'
        # tpqrt, tplqt needs small l, nb <= min( m, n )
        #if (opts.l == parser.get_default('l')):
        #    opts.l = '0,5,100'
        if (opts.nb == parser.get_default('nb')):
            opts.nb = '8,64'
        #if (opts.ka == parser.get_default('ka')):
        #    opts.ka = '5'
        #if (opts.kb == parser.get_default('kb')):
        #    opts.kb = '5'
        #if (opts.kd == parser.get_default('kd')):
        #    opts.kd = '5'

    if (opts.small):
        n       += ' --dim=25:100:25'
        tall    += ' --dim=50:200:50x25:100:25'  # 2:1
        wide    += ' --dim=25:100:25x50:200:50'  # 1:2
        mnk     += ' --dim=25x50x75 --dim=50x25x75' \
                +  ' --dim=25x75x50 --dim=50x75x25' \
                +  ' --dim=75x25x50 --dim=75x50x25'
        nk_tall += ' --dim=1x50:200:50x25:100:25'
        nk_wide += ' --dim=1x25:100:25x50:200:50'

    if (opts.medium):
        n       += ' --dim=100:500:100'
        tall    += ' --dim=200:1000:200x100:500:100'  # 2:1
        wide    += ' --dim=100:500:100x200:1000:200'  # 1:2
        mnk     += ' --dim=100x300x600 --dim=300x100x600' \
                +  ' --dim=100x600x300 --dim=300x600x100' \
                +  ' --dim=600x100x300 --dim=600x300x100'
        nk_tall += ' --dim=1x200:1000:200x100:500:100'
        nk_wide += ' --dim=1x100:500:100x200:1000:200'

    if (opts.large):
        n       += ' --dim=1000:5000:1000'
        tall    += ' --dim=2000:10000:2000x1000:5000:1000'  # 2:1
        wide    += ' --dim=1000:5000:1000x2000:10000:2000'  # 1:2
        mnk     += ' --dim=1000x3000x6000 --dim=3000x1000x6000' \
                +  ' --dim=1000x6000x3000 --dim=3000x6000x1000' \
                +  ' --dim=6000x1000x3000 --dim=6000x3000x1000'
        nk_tall += ' --dim=1x2000:10000:2000x1000:5000:1000'
        nk_wide += ' --dim=1x1000:5000:1000x2000:10000:2000'

    mn  = ''
    nk  = ''
    if (opts.square):
        mn = n
        nk = n
    if (opts.tall):
        mn += tall
        nk += nk_tall
    if (opts.wide):
        mn += wide
        nk += nk_wide
    if (opts.mnk):
        mnk = mn + mnk
    else:
        mnk = mn
# end

# BLAS and LAPACK
dtype  = opts.type.split( ',' )
#layout = ' --layout=' + opts.layout if (opts.layout) else ''
transA = ' --transA=' + opts.transA if (opts.transA) else ''
transB = ' --transB=' + opts.transB if (opts.transB) else ''
trans  = ' --trans='  + opts.trans  if (opts.trans)  else ''
geuplo = ' --uplo='   + opts.geuplo if (opts.geuplo) else ''
uplo   = ' --uplo='   + opts.uplo   if (opts.uplo)   else ''
diag   = ' --diag='   + opts.diag   if (opts.diag)   else ''
side   = ' --side='   + opts.side   if (opts.side)   else ''
a      = ' --alpha='  + opts.alpha  if (opts.alpha)  else ''
ab     = a+' --beta=' + opts.beta   if (opts.beta)   else a
incx   = ' --incx='   + opts.incx   if (opts.incx)   else ''
#incy   = ' --incy='   + opts.incy   if (opts.incy)   else ''
#align  = ' --align='  + opts.align  if (opts.align)  else ''
check  = ' --test='   + opts.check  if (opts.check)  else ''  # todo: PLASMA uses `test`, others use `check`
ref    = ' --ref='    + opts.ref    if (opts.ref)    else ''
verbose = ' --verbose=' + opts.verbose if (opts.verbose) else ''
repeat  = ' --repeat='  + opts.repeat  if (opts.repeat)  else ''

# LAPACK only
#itype  = ' --itype='  + opts.itype  if (opts.itype)  else ''
#factored = ' --factored=' + opts.factored if (opts.factored)  else ''
#equed  = ' --equed='  + opts.equed  if (opts.equed)  else ''
#pivot  = ' --pivot='  + opts.pivot  if (opts.pivot)  else ''
#direction = ' --direction=' + opts.direction if (opts.direction) else ''
#storev = ' --storev=' + opts.storev if (opts.storev) else ''
norm   = ' --norm='   + opts.norm   if (opts.norm)   else ''
#ijob   = ' --ijob='   + opts.ijob   if (opts.ijob)   else ''
#jobz   = ' --jobz='   + opts.jobz   if (opts.jobz)   else ''
job    = ' --job='    + opts.job    if (opts.job)    else ''
#jobu   = ' --jobu='   + opts.jobu   if (opts.jobu)   else ''
#jobvt  = ' --jobvt='  + opts.jobvt  if (opts.jobvt)  else ''
#jobvl  = ' --jobvl='  + opts.jobvl  if (opts.jobvl)  else ''
#jobvr  = ' --jobvr='  + opts.jobvr  if (opts.jobvr)  else ''
#jobvs  = ' --jobvs='  + opts.jobvs  if (opts.jobvs)  else ''
#balanc = ' --balanc=' + opts.balanc if (opts.balanc) else ''
#sort   = ' --sort='   + opts.sort   if (opts.sort)   else ''
#sense  = ' --sense='  + opts.sense  if (opts.sense)  else ''
#vect   = ' --vect='   + opts.vect   if (opts.vect)   else ''
#l      = ' --l='      + opts.l      if (opts.l)      else ''
nb     = ' --nb='     + opts.nb     if (opts.nb)     else ''
#ka     = ' --ka='     + opts.ka     if (opts.ka)     else ''
#kb     = ' --kb='     + opts.kb     if (opts.kb)     else ''
#kd     = ' --kd='     + opts.kd     if (opts.kd)     else ''
kl     = ' --kl='     + opts.kl     if (opts.kl)     else ''
ku     = ' --ku='     + opts.ku     if (opts.ku)     else ''
kd     = kl + ku  # todo: add kd to PLASMA tester
#vl     = ' --vl='     + opts.vl     if (opts.vl)     else ''
#vu     = ' --vu='     + opts.vu     if (opts.vu)     else ''
#il     = ' --il='     + opts.il     if (opts.il)     else ''
#iu     = ' --iu='     + opts.iu     if (opts.iu)     else ''
#mtype  = ' --matrixtype=' + opts.matrixtype if (opts.matrixtype) else ''

# general options for all routines
# todo: + tol + repeat (see SLATE)
gen = check + ref + verbose + repeat + nb

#-------------------------------------------------------------------------------
# Filters a comma separated list csv based on items in list values.
# If no items from csv are in values, returns '?'.
def filter_csv( values, csv ):
    f = list( filter( lambda x: x in values, csv.split( ',' ) ) )
    if (not f):
        return '?'
    return ','.join( f )
# end

#-------------------------------------------------------------------------------
# Filters a list based on items in list values.
def filter_list( values, iterable ):
    return list( filter( lambda x: x in values, iterable ) )
# end

#-------------------------------------------------------------------------------
# Limit options to specific values.
dtype_real    = filter_list( ('s', 'd'), dtype )
dtype_complex = filter_list( ('c', 'z'), dtype )
dtype_double  = filter_list( ('d', 'z'), dtype )
print( 'dtype_real',    list( dtype_real    ) )
print( 'dtype_complex', list( dtype_complex ) )
print( 'dtype_double',  list( dtype_double  ) )

trans_nt = ' --trans=' + filter_csv( ('n', 't'), opts.trans )
trans_nc = ' --trans=' + filter_csv( ('n', 'c'), opts.trans )

# positive inc
incx_pos = ' --incx=' + filter_csv( ('1', '2'), opts.incx )
#incy_pos = ' --incy=' + filter_csv( ('1', '2'), opts.incy )

#-------------------------------------------------------------------------------
cmds = []

# Level 1
#if (opts.blas1):
#    cmds += [
#    [ 'asum',  dtype, n + incx_pos ],
#    [ 'axpy',  dtype, n + incx + incy ],
#    [ 'copy',  dtype, n + incx + incy ],
#    [ 'dot',   dtype, n + incx + incy ],
#    [ 'dotu',  dtype, n + incx + incy ],
#    [ 'iamax', dtype, n + incx_pos ],
#    [ 'nrm2',  dtype, n + incx_pos ],
#    [ 'rot',   dtype, n + incx + incy ],
#    [ 'rotg',  dtype, '' ],
#    [ 'rotm',  dtype_real, n + incx + incy ],
#    [ 'rotmg', dtype_real, '' ],
#    [ 'scal',  dtype, n + incx_pos ],
#    [ 'swap',  dtype, n + incx + incy ],
#    ]

# Level 2
#if (opts.blas2):
#    cmds += [
#    [ 'gemv',  dtype,       trans + mn + incx + incy ],
#    [ 'ger',   dtype,       mn + incx + incy ],
#    [ 'geru',  dtype,       mn + incx + incy ],
#    [ 'hemv',  dtype,       uplo + n + incx + incy ],
#    [ 'her',   dtype,       uplo + n + incx ],
#    [ 'her2',  dtype,       uplo + n + incx + incy ],
#    [ 'symv',  dtype,       uplo + n + incx + incy ],
#    [ 'syr',   dtype,       uplo + n + incx ],
#    [ 'syr2',  dtype,       uplo + n + incx + incy ],
#    [ 'trmv',  dtype,       uplo + trans + diag + n + incx ],
#    [ 'trsv',  dtype,       uplo + trans + diag + n + incx ],
#    ]

# Level 3
# Unlike LAPACK++, PLASMA doesn't support hemm, herk, her2k on
# real-valued matrices.
if (opts.blas3):
    cmds += [
    [ 'gemm',  dtype,         transA + transB + mnk ],
    [ 'hemm',  dtype_complex, side + uplo + mn ],
    [ 'symm',  dtype,         side + uplo + mn ],
    [ 'trmm',  dtype,         side + uplo + trans + diag + mn ],
    [ 'trsm',  dtype,         side + uplo + trans + diag + mn ],
    #['herk',  dtype_real,    uplo + trans    + mn ],
    [ 'herk',  dtype_complex, uplo + trans_nc + mn ],
    [ 'syrk',  dtype_real,    uplo + trans_nt + mn ],  # PLASMA doesn't allow trans=c
    [ 'syrk',  dtype_complex, uplo + trans_nt + mn ],
    #['her2k', dtype_real,    uplo + trans    + mn ],
    [ 'her2k', dtype_complex, uplo + trans_nc + mn ],
    [ 'syr2k', dtype_real,    uplo + trans_nt + mn ],  # PLASMA doesn't allow trans=c
    [ 'syr2k', dtype_complex, uplo + trans_nt + mn ],
    ]

# LU
if (opts.lu):
    cmds += [
    [ 'gesv',  dtype, gen + n ],
    #['gesvx', dtype, gen + n + factored + trans ],
    [ 'getrf', dtype, gen + mn ],
    [ 'getrs', dtype, gen + n + trans ],
    [ 'getri', dtype, gen + n ],
    #['gecon', dtype, gen + n ],
    #['gerfs', dtype, gen + n + trans ],
    #['geequ', dtype, gen + n ],

    # Banded
    [ 'gbsv',  dtype, gen + n  + kl + ku ],
    [ 'gbtrf', dtype, gen + mn + kl + ku ],
    #['gbtrs', dtype, gen + n  + kl + ku + trans ],
    #['gbcon', dtype, gen + n  + kl + ku ],
    #['gbrfs', dtype, gen + n  + kl + ku + trans ],
    #['gbequ', dtype, gen + n  + kl + ku ],
    ]

# Cholesky
if (opts.chol):
    cmds += [
    [ 'posv',  dtype, gen + n + uplo ],
    [ 'potrf', dtype, gen + n + uplo ],
    [ 'potrs', dtype, gen + n + uplo ],
    [ 'potri', dtype, gen + n + uplo ],
    #['pocon', dtype, gen + n + uplo ],
    #['porfs', dtype, gen + n + uplo ],
    #['poequ', dtype, gen + n ],  # only diagonal elements (no uplo)

    # Banded
    [ 'pbsv',  dtype, gen + n + kd + uplo ],
    [ 'pbtrf', dtype, gen + n + kd + uplo ],
    #['pbtrs', dtype, gen + n + kd + uplo ],
    #['pbcon', dtype, gen + n + kd + uplo ],
    #['pbrfs', dtype, gen + n + kd + uplo ],
    #['pbequ', dtype, gen + n + kd + uplo ],
    ]

# symmetric indefinite, Aasen
# todo: only lower supported, add upper
# todo: only real supported, add [cz]sysv, etc.
if (opts.sysv):
    cmds += [
    [ 'sysv',  dtype_real, gen + n + ' --uplo=l' ],
    [ 'sytrf', dtype_real, gen + n + ' --uplo=l' ],
    #['sytrs', dtype_real, gen + n + ' --uplo=l' ],
    #['sytri', dtype_real, gen + n + ' --uplo=l' ],
    #['sycon', dtype_real, gen + n + ' --uplo=l' ],
    #['syrfs', dtype_real, gen + n + ' --uplo=l' ],
    ]

# Hermitian indefinite
# todo: only lower supported, add upper
if (opts.hesv):
    cmds += [
    [ 'hesv',  dtype_complex, gen + n + ' --uplo=l' ],
    [ 'hetrf', dtype_complex, gen + n + ' --uplo=l' ],
    #['hetrs', dtype_complex, gen + n + ' --uplo=l' ],
    #['hetri', dtype_complex, gen + n + ' --uplo=l' ],
    #['hecon', dtype_complex, gen + n + ' --uplo=l' ],
    #['herfs', dtype_complex, gen + n + ' --uplo=l' ],
    ]

# least squares
if (opts.least_squares):
    cmds += [
    [ 'gels',  dtype_real,    gen + mn + trans_nt ],
    [ 'gels',  dtype_complex, gen + mn + trans_nc ],
    #['gelsy', dtype, gen + mn ],
    #['gelsd', dtype, gen + mn ],
    #['gelss', dtype, gen + mn ],
    #['getsls', dtype, gen + mn + trans_nc ],

    # Generalized
    #['gglse', dtype, gen + mnk ],
    #['ggglm', dtype, gen + mnk ],
    ]

# QR
if (opts.qr):
    cmds += [
    #['geqr',  dtype, gen + n + wide + tall ],
    [ 'geqrf', dtype, gen + n + wide + tall ],
    #['ggqrf', dtype, gen + mnk ],
    #['ungqr', dtype, gen + mn ],  # m >= n
    # todo: PLASMA doesn't support all combinations of m x n x k (mnk).
    [ 'ormqr', dtype_real,    gen + n + side + trans_nt ],  # real does trans = N, T; PLASMA doesn't allow C
    [ 'unmqr', dtype_complex, gen + n + side + trans_nc ],  # complex does trans = N, C, not T

    #['orhr_col', dtype_real, gen + n + tall ],
    #['unhr_col', dtype,      gen + n + tall ],

    #['gemqrt', dtype_real,    gen + n + nb + side + trans    ],  # real does trans = N, T, C
    #['gemqrt', dtype_complex, gen + n + nb + side + trans_nc ],  # complex does trans = N, C, not T

    # Triangle-pentagon
    # todo: -l argument
    #['tpqrt',  dtype,         gen + mn + l + nb ],
    #['tpqrt2', dtype,         gen + mn + l ],
    #['tpmqrt', dtype_real,    gen + mn + l + nb + side + trans    ],  # real does trans = N, T, C
    #['tpmqrt', dtype_complex, gen + mn + l + nb + side + trans_nc ],  # complex does trans = N, C, not T
    #['tprfb', dtype, gen + mn + l ],  # TODO: bug in LAPACKE crashes tester
    ]

# LQ
if (opts.lq):
    cmds += [
    [ 'gelqf', dtype, gen + mn ],
    #['gglqf', dtype, gen + mn ],
    #['unglq', dtype, gen + mn ],  # m <= n, k <= m  TODO Fix the input sizes to match constraints
    # todo: PLASMA doesn't support all combinations of m x n x k (mnk).
    [ 'ormlq', dtype_real,    gen + n + side + trans_nt ],  # real does trans = N, T; PLASMA doesn't allow C
    [ 'unmlq', dtype_complex, gen + n + side + trans_nc ],  # complex does trans = N, C, not T

    # Triangle-pentagon
    #['tplqt',  dtype,         gen + mn + l + nb ],
    #['tplqt2', dtype,         gen + mn + l ],
    #['tpmlqt', dtype_real,    gen + mn + l + nb + side + trans    ],  # real does trans = N, T, C
    #['tpmlqt', dtype_complex, gen + mn + l + nb + side + trans_nc ],  # complex does trans = N, C, not T
    ]

# QL
if (opts.ql):
    cmds += [
    #['geqlf', dtype, gen + mn ],
    #['ggqlf', dtype, gen + mn ],
    #['ungql', dtype, gen + mn ],
    #['unmql', dtype_real,    gen + mnk + side + trans    ],  # real does trans = N, T, C
    #['unmql', dtype_complex, gen + mnk + side + trans_nc ],  # complex does trans = N, C, not T
    ]

# RQ
if (opts.rq):
    cmds += [
    #['gerqf', dtype, gen + mn ],
    #['ggrqf', dtype, gen + mnk ],
    #['ungrq', dtype, gen + mnk ],
    #['unmrq', dtype_real,    gen + mnk + side + trans    ],  # real does trans = N, T, C
    #['unmrq', dtype_complex, gen + mnk + side + trans_nc ],  # complex does trans = N, C, not T
    ]

# Hermitian/symmetric eigenvalues
if (opts.heev):
    cmds += [
    #['heev',  dtype, gen + n + jobz + uplo ],
    #['heevx', dtype, gen + n + jobz + uplo + vl + vu ],
    #['heevx', dtype, gen + n + jobz + uplo + il + iu ],
    #['heevd', dtype, gen + n + jobz + uplo ],
    #['heevr', dtype, gen + n + jobz + uplo + vl + vu ],
    #['heevr', dtype, gen + n + jobz + uplo + il + iu ],
    #['hetrd', dtype, gen + n + uplo ],
    #['ungtr', dtype,         gen + n + uplo ],
    #['unmtr', dtype_real,    gen + mn + uplo + side + trans    ],  # real does trans = N, T, C
    #['unmtr', dtype_complex, gen + mn + uplo + side + trans_nc ],  # complex does trans = N, C, not T

    # Banded
    #['hbev',  dtype, gen + n + jobz + uplo ],
    #['hbevx', dtype, gen + n + jobz + uplo + vl + vu ],
    #['hbevx', dtype, gen + n + jobz + uplo + il + iu ],
    #['hbevd', dtype, gen + n + jobz + uplo ],
    #['hbevr', dtype, gen + n + jobz + uplo + vl + vu ],
    #['hbevr', dtype, gen + n + jobz + uplo + il + iu ],
    #['hbtrd', dtype, gen + n + uplo ],
    #['ubgtr', dtype, gen + n + uplo ],
    #['ubmtr', dtype_real,    gen + la + mn + uplo + side + trans    ],
    #['ubmtr', dtype_complex, gen + la + mn + uplo + side + trans_nc ],
    ]

# generalized Hermitian/symmetric eigenvalues
if (opts.hegv):
    cmds += [
    #['hegv',  dtype, gen + n + itype + jobz + uplo ],
    #['hegvx', dtype, gen + n + itype + jobz + uplo + vl + vu ],
    #['hegvx', dtype, gen + n + itype + jobz + uplo + il + iu ],
    #['hegvd', dtype, gen + n + itype + jobz + uplo ],
    #['hegvr', dtype, gen + n + uplo ],
    #['hegst', dtype, gen + n + itype + uplo ],

    # Banded
    #['hbgv',  dtype, gen + n + jobz + uplo + ka + kb ],
    #['hbgvx', dtype, gen + n + jobz + uplo + ka + kb + vl + vu ],
    #['hbgvx', dtype, gen + n + jobz + uplo + ka + kb + il + iu ],
    #['hbgvd', dtype, gen + n + jobz + uplo + ka + kb ],
    #['hbgvr', dtype, gen + n + uplo + ka + kb ],
    #['hbgst', dtype, gen + n + vect + uplo + ka + kb ],
    ]

# non-symmetric eigenvalues
if (opts.geev):
    cmds += [
    #['geev',  dtype, gen + n + jobvl + jobvr ],
    #['ggev',  dtype, gen + n + jobvl + jobvr ],
    #['geevx', dtype, gen + n + balanc + jobvl + jobvr + sense ],
    #['gehrd', dtype, gen + n ],
    #['unghr', dtype, gen + n ],
    #['unmhr', dtype_real,    gen + mn + side + trans    ],  # real does trans = N, T, C
    #['unmhr', dtype_complex, gen + mn + side + trans_nc ],  # complex does trans = N, C, not T
    #['trevc', dtype, gen + n + side + howmany + select ],
    #['geesx', dtype, gen + n + jobvs + sort + select + sense ],
    #['tgexc', dtype, gen + n + jobvl + jobvr ],
    #['tgsen', dtype, gen + n + jobvl + jobvr + ijob ],
    ]

# svd
if (opts.svd):
    cmds += [
    # todo: MKL seems to have a bug with jobu=o,s and jobvt=o,s,a
    # for tall matrices, e.g., dim=100x50. Skip failing combinations for now.
    #['gesvd', dtype, gen + mn + jobu + jobvt ],
    #['gesvd', dtype, gen + mn + " --jobu=n,a" + jobvt ],
    #['gesvd', dtype, gen + mn + " --jobu=o,s --jobvt=n" ],
    [ 'gesdd', dtype, gen + mn + job ],
    #['gesvdx', dtype, gen + mn + jobz + jobvr + vl + vu ],
    #['gesvdx', dtype, gen + mn + jobz + jobvr + il + iu ],
    #['gesvd_2stage', dtype, gen + mn ],
    #['gesdd_2stage', dtype, gen + mn ],
    #['gesvdx_2stage', dtype, gen + mn ],
    #['gejsv', dtype, gen + mn ],
    #['gesvj', dtype, gen + mn + joba + jobu + jobv ],
    ]

# auxilary
if (opts.aux):
    cmds += [
    [ 'lacpy', dtype,      gen + mn + geuplo ],
    [ 'lascl', dtype,      gen + mn + geuplo ],
    [ 'laset', dtype,      gen + mn + geuplo ],
    #['laswp', dtype,      gen + mn ],
    ]

# norms
if (opts.norms):
    cmds += [
    [ 'lange', dtype, gen + mn + norm ],
    #['lanhe', dtype, gen + n  + norm + uplo ],
    [ 'lansy', dtype, gen + n  + norm + uplo ],
    [ 'lantr', dtype, gen + mn + norm + uplo + diag ],
    #['lanhs', dtype, gen + n  + norm ],

    # Banded
    [ 'langb', dtype, gen + mn + kl + ku + norm ],
    #['lanhb', dtype, gen + n + kd + norm + uplo ],
    #['lansb', dtype, gen + n + kd + norm + uplo ],
    #['lantb', dtype, gen + n + kd + norm + uplo + diag ],

    # Tri-diagonal
    #['langt', dtype, gen + n + norm ],
    #['lanht', dtype, gen + n + norm ],
    #['lanst', dtype, gen + n + norm ],
    ]

#-------------------------------------------------------------------------------
# When stdout is redirected to file instead of TTY console,
# and  stderr is still going to a TTY console,
# print extra summary messages to stderr.
output_redirected = sys.stderr.isatty() and not sys.stdout.isatty()

#-------------------------------------------------------------------------------
# if output is redirected, prints to both stderr and stdout;
# otherwise prints to just stdout.
def print_tee( *args ):
    global output_redirected
    print( *args )
    if (output_redirected):
        print( *args, file=sys.stderr )
# end

#-------------------------------------------------------------------------------
# cmd is a tuple: (function, dtypes, args)
# function is string, like 'gemm'
# dtypes is list of precisions, like ['s', 'd', 'c', 'z']
# args is string, like '--dim=100'
# returns pair: (error, output-string), where error is the result from
# subprocess wait, so error == 0 is success.
#
def run_test( cmd ):
    err_all = 0
    output = ''
    print( 'run_test cmd', cmd )
    for dtype in cmd[1]:
        cmd_ = opts.test +' '+ dtype + cmd[0] +' '+ cmd[2]
        print_tee( cmd_ )
        if (opts.dry_run):
            continue

        p = subprocess.Popen( cmd_.split(), stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT )
        p_out = p.stdout
        if (sys.version_info.major >= 3):
            p_out = io.TextIOWrapper(p.stdout, encoding='utf-8')
        # Read unbuffered ("for line in p.stdout" will buffer).
        for line in iter(p_out.readline, ''):
            print( line, end='' )
            output += line
        err = p.wait()
        if (err != 0):
            err_all = err
            print_tee( 'FAILED: exit code', err )
        else:
            print_tee( 'pass' )
    # end
    return (err_all, output)
# end

#-------------------------------------------------------------------------------
# Utility to pretty print XML.
# See https://stackoverflow.com/a/33956544/1655607
#
def indent_xml( elem, level=0 ):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent_xml( elem, level+1 )
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
# end

#-------------------------------------------------------------------------------
# run each test

print_tee( ' '.join( sys.argv ) )
start = time.time()
print_tee( time.ctime() )

failed_tests = []
passed_tests = []
ntests = len(opts.tests)
run_all = (ntests == 0)

seen = set()
for cmd in cmds:
    if ((run_all or cmd[0] in opts.tests) and cmd[0] not in opts.exclude):
        if (start_routine and cmd[0] != start_routine):
            print_tee( 'skipping', cmd[0] )
            continue
        start_routine = None

        seen.add( cmd[0] )
        (err, output) = run_test( cmd )
        if (err):
            failed_tests.append( (cmd[0], err, output) )
        else:
            passed_tests.append( cmd[0] )
print( '-' * 80 )

not_seen = list( filter( lambda x: x not in seen, opts.tests ) )
if (not_seen):
    print_tee( 'Warning: unknown routines:', ' '.join( not_seen ))

# print summary of failures
nfailed = len( failed_tests )
if (nfailed > 0):
    print_tee( '\n' + str(nfailed) + ' routines FAILED:',
               ', '.join( [x[0] for x in failed_tests] ) )
else:
    print_tee( '\n' + 'All routines passed.' )

# generate jUnit compatible test report
if opts.xml:
    print( 'writing XML file', opts.xml )
    root = ET.Element("testsuites")
    doc = ET.SubElement(root, "testsuite",
                        name="test_suite",
                        tests=str(ntests),
                        errors="0",
                        failures=str(nfailed))

    for (test, err, output) in failed_tests:
        testcase = ET.SubElement(doc, "testcase", name=test)

        failure = ET.SubElement(testcase, "failure")
        if (err < 0):
            failure.text = "exit with signal " + str(-err)
        else:
            failure.text = str(err) + " tests failed"

        system_out = ET.SubElement(testcase, "system-out")
        system_out.text = output
    # end

    for test in passed_tests:
        testcase = ET.SubElement(doc, 'testcase', name=test)
        testcase.text = 'PASSED'

    tree = ET.ElementTree(root)
    indent_xml( root )
    tree.write( opts.xml )
# end

elapsed = time.time() - start
print_tee( 'Elapsed %.2f sec' % elapsed )
print_tee( time.ctime() )

exit( nfailed )
