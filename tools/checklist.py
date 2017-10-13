#!/usr/bin/env python
#
# This does some simple searches to look for common formatting issues,
# and a few potential bugs. As it uses regular expressions, it may have
# both false positives and false negatives.
#
# Usage: from top level of plasma directory:
# ./tools/checklist.py [files]
#
# If no files given, checks all files currently in Mercurial.
#
# See the ICL style guide
# https://bitbucket.org/icl/guides/wiki/icl_style_guide

import sys
import os
import re

# ------------------------------------------------------------------------------
def error( help, lineno=None, line=None ):
    '''
    The first time it sees this file, prints filename.
    Then prints error message.
    Uses globals first and filename, set in process().
    '''
    global first, filename
    if (first):
        print "====", filename
        first = False
    if (lineno):
        print "%4d: %s: %s" % (lineno, help, line)
    else:
        print help
# end

# ------------------------------------------------------------------------------
regexps = {}

# ------------------------------------------------------------------------------
def grep( lines, regexp, help, flags=0, exclude=None ):
    '''
    Prints error for all lines that match regexp (using flags),
    and don't match exclude, if given.
    '''
    if (not regexps.has_key( regexp )):
        regexps[ regexp ] = re.compile( regexp, flags=flags )
    regexp = regexps[ regexp ]

    if (exclude):
        if (not regexps.has_key( exclude )):
            regexps[ exclude ] = re.compile( exclude )
        exclude = regexps[ exclude ]
    # end

    lineno = 0
    for line in lines:
        lineno += 1
        if (re.search( regexp, line )
            and not (exclude and re.search( exclude, line ))):
            error( help, lineno, line )
        # end
    # end
# end

# ------------------------------------------------------------------------------
def process( fname ):
    '''
    Processes one file, named fname.
    Reads the file and looks for various patterns that violate style or
    conventions.
    '''
    # to ease argument passing, error() takes first and filename as globals
    global first, filename
    first = True
    filename = fname

    # read file
    infile = open( filename )
    txt = ''    # while text in one string
    lines = []  # array of lines, without newline
    for line in infile:
        txt += line
        lines.append( line.rstrip('\n') )
    # end

    # ----------------------------------------
    # rules for all files (Makefile, src, header, etc.)
    grep( lines, r'\r',
          help="Remove Windows returns; use Unix newlines only" )

    grep( lines, r' +$', flags=re.I,
          help="Remove trailing space" )

    grep( lines, r'@version|@date|@author',
          help="@version, @date, @author deprecated" )

    grep( lines, r'@file.+',
          help="@file should not have filename" )

    # ----------------------------------------
    # rules for src files (including headers)
    if (re.search( r'\.(c|cc|cpp|h|hh|hpp)$', filename )):
        # -------------------- spaces
        grep( lines, r'\t',
              help="Remove tabs; use 4 spaces" )

        # Check that at least if, for, while, and braces are indented by
        # multiple of 4 spaces.
        # More rigorous indentation checks are hard without parsing C.
        grep( lines, r'^(    )* {1,3}(#if|#define|#pragma|if|for|while|\{|\})', flags=re.I,
              help="Not 4 space indent" )

        grep( lines, r'^#pragma omp',
              help="#pragma omp should be indented" )

        #grep( lines, r'^ +#(if|define)',
        #      help="#if and #define should not be indented" )

        grep( lines, r'[^ @_]\{',
              help="Missing space before opening brace" )

        grep( lines, r' [,;]',
              help="Extra space before semicolon or comma" )

        grep( lines, r';[^ \n]',
              help="Missing space after semicolon" )

        grep( lines, r'^ +(if|for|while|switch)\(',
              help="Missing space after if, for, while, switch" )

        grep( lines, r'^ +(if|for|while|switch) *\( ',
              help="Extra space inside if, for, while, switch condition" )

        # Relational (==, !=, <=, >=, <, >), assignment (+=, -=, *=, /=),"
        # and boolean (&&, ||) operators should have spaces on both sides."
        # Other operators may have spaces, which should be consistent, e.g., not (x +y)."
        grep( lines,
              r'([\w\[\]\(\)])(==|!=|<=|>=|<|>|\+=|-=|\*=|/=|&&)([\w\[\]\(\)-])',
              help="Missing space around relational, assignment, and boolean operators, e.g., (x==y) should be (x == y).\n    " )

        # In (x #y) check, can't check (x -y) because of unary - minus,
        # nor (x *y) because of pointer declarations, float *x.
        # todo: omitting || for now because of norms, e.g., ||PA - LU||
        grep( lines, r'([\w\[\]\(\)]) +(==|!=|<=|>=|<|>|\+=|-=|\*=|/=|&&|\+|\/)([a-zA-Z0-9\[\]\(\)-])',
              exclude='#include <',
              help="Missing space after  operators, e.g., (x ==y) should be (x == y).\n    " )

        grep( lines, r'([\w\[\]\(\)])(==|!=|<=|>=|<|>|\+=|-=|\*=|/=|&&|\|\||\+|\-|\*|\/) +([\w\[\]\(\)-])',
              help="Missing space before operators, e.g., (x== y) should be (x == y).\n    " )

        # -------------------- newlines
        # some of these are global searches;
        # in some cases we find the only first occurrence

        grep( lines, r'^.{81}',
              help="Line exceeds 80 characters" )

        grep( lines, r'\} *else',
              help="Missing newline between { and else (don't cuddle)" )

        # first search finds any; second only
        #m = re.search( r'^(.*?\n)[^\n]*\{ *\n *\n', txt, flags=re.S )
        m = re.search( r'^(.*?\n)( *(if|else|for|while|do|switch)[^\n]*|)\{ *\n *\n', txt, flags=re.S )
        if (m):
            lineno = m.group(1).count('\n') + 1
            line = lines[lineno - 1]
            error( "Extra blank lines after {.", lineno, line )
        # end

        m = re.search( r'^(.*?\n\n)( *\}.*)', txt, flags=re.S )
        if (m):
            lineno = m.group(1).count('\n') + 1
            line = lines[lineno - 1]
            error( "Extra blank lines before }.", lineno, line )

        if (not re.search( r'\n$', txt )):
            error( "Missing newline at end of file (should be exactly one)." )

        if (re.search( r'\n\n$', txt )):
            error( "Extra newlines at end of file (should be exactly one)." )

        m = re.search( r'^(.*?\n)\}\n\S', txt, flags=re.S )
        if (m):
            lineno = m.group(1).count('\n') + 1
            line = lines[lineno - 1]
            error( "Missing blank line between functions (should be exactly one)." )
        # end

        if (re.search( r'^\}\n\n\n+\S', txt, flags=re.M )):
            error( "Extra blank lines between functions (should be exactly one)." )

        # -------------------- size_t usage
        # we assume core_blas routines operate on small tiles,
        # so don't need (size_t) casts.
        #if (not re.search( r'core_blas', filename )):
        #    grep( lines, r'ld\w+\*|\*ld\w+',
        #          exclude='\(size_t\)ld\w+\*',
        #          help="put lda first and use (size_t) cast, e.g., (size_t)lda*n.\n    " )
        ## end

        # -------------------- precision generation bugs
        if (not re.search( r'^ *#define PRECISION_\w', txt, flags=re.M )):
            grep( lines, r'defined\( *PRECISION_\w *\)|ifdef +PRECISION_\w',
                  help="Using PRECISION_[sdcz] without #define PRECISION_[sdcz]." )
        # end

        if (not re.search( r'^ *#define (REAL|COMPLEX)', txt, flags=re.M )):
            grep( lines, r'defined\( *(REAL|COMPLEX) *\)|ifdef +(REAL|COMPLEX)',
                  help="Using REAL or COMPLEX without #define REAL or COMPLEX." )
        # end

        # -------------------- comments & docs
        grep( lines, r'/\*[^*]',
              help="C style /* ... */ comments (excluding Doxygen). Use C++ // style." )

        grep( lines, r'\*\*[TH].',
              help="Using Fortran-style A**H or A**T. Please use A^H or A^T." )

        grep( lines, r" [A-Z]'",
              help="Might be using Matlab-style A'. Please use A^H or A^T. (This is really hard to search for.)" )

        grep( lines, r"'",
              exclude=r"'[a-zA-Z=:+]'|'\\0'|LAPACK's|PLASMA's|can't",
              help="Might be using Matlab-style A'. Please use A^H or A^T. (Second search.)" )

        grep( lines, r"@ingroup",
              exclude=r'@ingroup (plasma|core)_[^z]\w+',
              help="Use @ingroup (plasma|core)_{routine}, without precision: e.g., plasma_gemm.\n"
                  +"      See docs/doxygen/groups.dox for available groups.\n"
                  +"    " )

        grep( lines, r"\*\*\*",
              exclude=r'^.{80}$',
              help="Rule lines (/****/) should be exactly 80 characters" )

        grep( lines, r'@return',
              help="Use @retval; delete @return" )

        grep( lines, r'\\retval',
              help="Use @retval instead of \\retval" )

        grep( lines, r'\b(ld\w+|\w) by \w',
              help="Use hyphens in \"m-by-n\", instead of \"m by n\"" )

        if (re.search( r'\b(z|pz|core_z)\S+\.(c|cc|cpp)', filename )
            and not re.search( r'zsy|zlansy', filename )):
            grep( lines, r'symmetric', flags=re.I,
                  help="Term \"symmetric\" should not occur in complex (z) routines (except zsy routines); use \"Hermitian\"." )
        # end

        if (re.search( r'\b(zsy|pzsy|core_zsy|zlansy)\S+\.(c|cc|cpp)', filename )):
            grep( lines, r'Hermitian', flags=re.I,
                  help="Term \"Hermitian\" should not occur in complex-symmetric (zsy) routines; use \"symmetric\"." )
        # end

        if (re.search( r'\b(d|pd|core_d)\S+\.(c|cc|cpp)', filename )):
            grep( lines, r'Hermitian', flags=re.I,
                  help="Term \"Hermitian\" should not occur in real (d) routines; codegen should fix this." )
        # end

        grep( lines, r"hermitian",
              help="Hermitian should be capitalized" )

        # -------------------- deprecated routines
        grep( lines, r'_Tile\b',
              help="_Tile versions are removed" )

        # -------------------- conventions
        grep( lines, r'__real__|__imag__\b',
              help="Replace __real__/__imag__ (GNU) with standard creal/cimag" )
    # end src

    # ----------------------------------------
    # rules for header files
    if (re.search( r'\.(h|hh|hpp)$', filename )):
        if (not re.search( r'^#ifndef \w+_(H|HPP)$', txt, flags=re.M )):
            error( "header missing multiple-inclusion guard (#ifndef FOO_H)" )

        if (not re.search( r'^#define \w+_(H|HPP)$', txt, flags=re.M )):
            error( "header missing multiple-inclusion guard (#define FOO_H)" )
    # end hdr

    if (not first):
        print
# end

# ------------------------------------------------------------------------------
def main():
    '''
    Processes files given on command line, or all files known in Mercurial.
    Excludes checklist.py and files.txt.
    '''
    if (len(sys.argv) > 1):
        files = sys.argv[1:]
    else:
        files = []
        p = os.popen("hg st -a -c -m")
        for line in p:
            fname = re.sub( r'^[MAC] ', '', line.strip() )
            if (not re.search( 'checklist.sh|checklist.py|files.txt|\.gz', fname )):
                files.append( fname )
        # end
    # end
    for fname in files:
        if (os.path.isfile( fname )):
            process( fname )
# end

# ------------------------------------------------------------------------------
if (__name__ == '__main__'):
    main()
