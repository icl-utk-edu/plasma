# PLASMA is a software package provided by:
# University of Tennessee, US,
# University of Manchester, UK.

from __future__ import print_function

# standard python modules
import os
import sys
import re
import shlex
import subprocess
from   subprocess import PIPE

# local modules
from environment import Environment


# ==============================================================================
# initialization and settings

# ------------------------------------------------------------------------------
def init():
	'''
	Initialize config module.
	'''
	global env, top_dir, auto, verbose
	env = Environment()
	
	top_dir = os.getcwd()
	if (not os.path.exists( 'config-src' )):
		os.mkdir( 'config-src' )
	os.chdir( 'config-src' )
	
	auto = False
	verbose = 0
	
	args = sys.argv[1:]
	for arg in args:
		if (arg == '--auto'):
			auto = True
		elif (arg == '--verbose' or arg == '-v'):
			verbose += 1
		else:
			match = re.search( r'^(\w+)=(.*)', arg )
			if (match):
				env[ match.group(1) ] = match.group(2)
			else:
				print_comment( 'unrecognized argument: ' + arg )
				#raise Error( "unrecognized command line argument: " + arg )
		# end
	# end
# end


# ------------------------------------------------------------------------------
def set_auto( in_auto=True ):
	'''
	Sets automatic, non-interactive mode on or off.
	Auto always selects the first available choice.
	Auto is off by default.
	'''
	global auto
	auto = in_auto
# end


# ------------------------------------------------------------------------------
def set_verbose( in_verbose=1 ):
	'''
	Sets verbose level.
	Verbose = 0 (off) by default.
	'''
	global verbose
	verbose = in_verbose
# end


# ==============================================================================
# support

# ------------------------------------------------------------------------------
# All errors this code raises.
# Using this allows Python Exceptions to fall through, give tracebacks.
class Error( Exception ):
	pass


# ------------------------------------------------------------------------------
# ANSI codes
esc     = chr(0x1B) + '['

red     = esc + '31m'
green   = esc + '32m'
yellow  = esc + '33m'
blue    = esc + '34m'
magenta = esc + '35m'
cyan    = esc + '36m'
white   = esc + '37m'

font_bold    = esc + '1m'
font_normal  = esc + '0m'

font_header  = font_bold
font_subhead = blue
font_comment = cyan

str_yes = font_bold + green + 'yes' + font_normal
str_no  = red + 'no' + font_normal

dots_width = 50
subdots_width = 58


# ------------------------------------------------------------------------------
# Current programming language
lang_stack = [ None ]

lang = None


def get_lang():
	global lang
	return lang


def set_lang( in_lang ):
	global lang, lang_stack
	assert( in_lang in ('C', 'C++', 'Fortran', 'F77') )
	lang_stack[-1] = in_lang
	lang = in_lang


def push_lang( in_lang ):
	global lang, lang_stack
	assert( in_lang in ('C', 'C++', 'Fortran', 'F77') )
	lang_stack.append( in_lang )
	lang = in_lang


def pop_lang():
	global lang, lang_stack
	if (len(lang_stack) == 1):
		raise Error("popping last language")
	pop = lang_stack.pop()
	lang = lang_stack[-1]
	return pop


# ==============================================================================
# output files

output_headers_done = False

# ------------------------------------------------------------------------------
# todo: in autoconf, this changes @DEFS@ to be -DHAVE_CONFIG_H instead of
# -Dfoo1=bar1 -Dfoo2=bar2 ... for all variables foo{i}.
def output_headers( files ):
	'''
	Create each file in files from file.in, substituting @foo@ with
	variable foo.
	files can be a single header file or an iterable list of files.
	If the contents of file are not modified, it does not touch it,
	to avoid unnecesary recompilation.
	Ex:
		cfg.output_headers( "config.h" )
		cfg.output_headers( ["config.h", "meta.h"] )
	'''
	output_headers_done = True
	print_header( 'Output files' )
	if (isinstance( files, str )):
		files = [ files ]
	pwd = os.getcwd()
	os.chdir( top_dir )
	for fname in files:
		txt = read( fname + '.in' )
		out = re.sub( r'@(\w+)@', sub_env, txt )
		out = re.sub( r'#undef (\w+)', sub_define, txt )
		if (os.path.exists( fname ) and out == read( fname )):
			print( fname, 'is unchanged' )
		else:
			print( 'creating', fname )
			write( fname, out )
	# end
	
	env['DEFS'] = '-DHAVE_CONFIG_H'
	
	os.chdir( pwd )
# end


# ------------------------------------------------------------------------------
def output_files( files ):
	'''
	Create each file in files from file.in, substituting @foo@ with
	variable foo.
	files can be a single header file or an iterable list of files.
	Unlike output_headers(), this always recreates the file even if the
	contents are the same.
	Ex:
		cfg.output_files( "make.inc" )
		cfg.output_files( ["Makefile", "src/Makefile"] )
	'''
	if (not output_headers_done):
		print_header( 'Output files' )
	if (isinstance( files, str )):
		files = [ files ]
	pwd = os.getcwd()
	os.chdir( top_dir )
	for fname in files:
		txt = read( fname + '.in' )
		out = re.sub( r'@(\w+)@', sub_env, txt )
		print( 'creating', fname )
		write( fname, out )
	# end
	os.chdir( pwd )
# end


# ==============================================================================
# compilers

# ------------------------------------------------------------------------------
def prog_cc( extra=[],
             default=['gcc', 'cc', 'icc', 'xlc_r', 'xlc', 'clang'],
             required=True ):
	'''
	Detect C compilers.
	
	extra       List of additional compilers to check, before default.
	default     List of compilers to check, currently:
		GNU         gcc
		generic     cc
		Intel       icc
		IBM         xlc_r
		IBM         xlc
		Clang       clang
	
	If $CC is set in environment or on command line,
	uses that instead of extra & default.
	If $CFLAGS, $LDFLAGS, or $LIBS are set in environment or on command line,
	uses those flags when compiling.
	
	It is recommended to override extra and not override default.
	'''
	print_header( 'Detecting C compilers' )
	cc = env['CC']
	if (cc):
		print_comment( 'Test using $CC=' + cc )
		compilers = [ cc ]
	else:
		compilers = unique( extra + default )
	
	set_lang('C')
	src = 'prog_cc.c'
	choices = []
	for cc in compilers:
		env.push()
		env['CC'] = cc
		try:
			try_compiler( src )
			
			print_subhead( '    Required features:' )
			prog_cc_c99()
			openmp()
			openmp_depend()
			choices.append( env.top() )  # passed required features
			
			print_subhead( '    Optional features:' )
			openmp_priority( required=False )
			#for flag in ('-O2', '-Wall', '-Wshadow', '-Wno-unused-function', '-pedantic'):
			#	compiler_flag( flag, required=False )
		except Error, e:
			if (verbose): print( e )
		env.pop()
		#print()
	# end
	val = choose( 'C compiler', 'CC', choices )
	if (val is not None):
		env.push( val )
	elif (required):
		raise Error( 'C compiler not found' )
# end


# ------------------------------------------------------------------------------
def prog_cxx( extra=[],
              default=['g++', 'c++', 'CC', 'cxx',
                       'icpc', 'xlC_r', 'xlC', 'clang++'],
              required=True ):
	'''
	Detect C++ compilers.
	
	extra       List of additional compilers to check, before default.
	default     List of compilers to check, currently:
		GNU         g++
		generic     c++
		generic     CC
		generic     cxx
		Intel       icpc
		IBM         xlC_r
		IBM         xlC
		Clang		clang++
	
	If $CXX is set in environment or on command line,
	uses that instead of extra & default.
	If $CXXFLAGS, $LDFLAGS, or $LIBS are set in environment or on command line,
	uses those flags when compiling.
	
	It is recommended to override extra and not override default.
	'''
	print_header( 'Detecting C++ compilers' )
	cxx = env['CXX']
	if (cxx):
		print_comment( 'Test using $CXX=' + cxx )
		compilers = [ cxx ]
	else:
		compilers = unique( extra + default )
	
	set_lang('C++')
	src = 'prog_cxx.cxx'
	choices = []
	for cxx in compilers:
		env.push()
		env['CXX'] = cxx
		try:
			try_compiler( src )
			
			print_subhead( '    Required features:' )
			prog_cxx_cxx11()
			openmp()
			openmp_depend()
			choices.append( env.top() )  # passed required features
			
			print_subhead( '    Optional features:' )
			openmp_priority( required=False )
			#for flag in ('-O2', '-Wall', '-Wshadow', '-Wno-unused-function', '-pedantic'):
			#	compiler_flag( flag, required=False )
		except Error, e:
			if (verbose): print( e )
		env.pop()
		#print()
	# end
	val = choose( 'C++ compiler', 'CXX', choices )
	if (val is not None):
		env.push( val )
	elif (required):
		raise Error( 'C++ compiler not found' )
# end


# ------------------------------------------------------------------------------
# todo: autoconf also has [ifc efc lf95 epcf90 frt cf77 fort77 fl32 af77]
fortran_compilers = [
	'gfortran', 'g95', 'g77',
	'fort', 'f95', 'f90', 'f77',
	'ftn', 'nagfor', 'ifort',
	'xlf95', 'xlf90', 'xlf',
	'pgfortran', 'pgf95', 'pgf90', 'pghpf', 'pgf77'
]

def prog_fortran( extra=[], default=fortran_compilers, required=True ):
	'''
	Detect modern Fortran compilers (Fortran 90 and newer).
	
	extra       List of additional compilers to check, before default.
	default     List of compilers to check, currently:
		GNU         gfortran
		GNU         g95
		GNU         g77
		generic     fort
		generic     f95
		generic     f90
		generic     f77
		Cray        ftn
		NAG         nagfor
		Intel       ifort
		IBM         xlf
		IBM         xlf95
		IBM         xlf90
		PGI         pgfortran
		PGI         pgf95
		PGI         pgf90
		PGI         pghpf
		PGI         pgf77
	
	If $FC is set in environment or on command line,
	uses that instead of extra & default.
	If $FCFLAGS, $LDFLAGS, or $LIBS are set in environment or on command line,
	uses those flags when compiling.
	
	It is recommended to override extra and not override default.
	'''
	print_header( 'Detecting Fortran compilers' )
	fc = env['FC']
	if (fc):
		print_comment( 'Test using $FC=' + fc )
		compilers = [ fc ]
	else:
		compilers = unique( extra + default )
	
	set_lang('Fortran')
	src = 'prog_fortran.f90'
	choices = []
	for fc in compilers:
		env.push()
		env['FC'] = fc
		try:
			try_compiler( src )
			
			print_subhead( '    Required features:' )
			prog_fortran_f2008()
			openmp()
			choices.append( env.top() )  # passed required features
			
			# already checked for C compiler; checking again makes duplicate
			# flags in @DEFS@
			#print_subhead( '    Optional features:' )
			#openmp_depend(   required=False )
			#openmp_priority( required=False )
			
			#for flag in ('-O2', '-Wall', '-Wshadow', '-Wno-unused-function', '-pedantic'):
			#	compiler_flag( flag, required=False )
		except Error, e:
			if (verbose): print( e )
		env.pop()
		#print()
	# end
	val = choose( 'Fortran compiler', 'FC', choices )
	if (val is not None):
		env.push( val )
	elif (required):
		raise Error( 'Fortran compiler not found' )
# end


# ------------------------------------------------------------------------------
def prog_f77( extra=[], default=fortran_compilers, required=True ):
	'''
	Detect Fortran 77 compilers.
	
	extra       List of additional compilers to check, before default.
	default     List of compilers to check, currently:
		GNU         gfortran
		GNU         g95
		GNU         g77
		generic     fort
		generic     f95
		generic     f90
		generic     f77
		Cray        ftn
		NAG         nagfor
		Intel       ifort
		IBM         xlf
		IBM         xlf95
		IBM         xlf90
		PGI         pgfortran
		PGI         pgf95
		PGI         pgf90
		PGI         pghpf
		PGI         pgf77
	
	If $F77 is set in environment or on command line,
	uses that instead of extra & default.
	If $FFLAGS, $LDFLAGS, or $LIBS are set in environment or on command line,
	uses those flags when compiling.
	
	It is recommended to override extra and not override default.
	'''
	print_header( 'Detecting Fortran 77 compilers' )
	f77 = env['F77']
	if (f77):
		print_comment( 'Test using $F77=' + f77 )
		compilers = [ f77 ]
	else:
		compilers = unique( extra + default )
	
	set_lang('F77')
	src = 'prog_f77.f'
	choices = []
	for f77 in compilers:
		env.push()
		env['F77'] = f77
		try:
			try_compiler( src )
			
			print_subhead( '    Required features:' )
			openmp()
			choices.append( env.top() )  # passed required features
			
			print_subhead( '    Optional features:' )
			openmp_depend(   required=False )
			openmp_priority( required=False )
			#for flag in ('-O2', '-Wall', '-Wshadow', '-Wno-unused-function', '-pedantic'):
			#	compiler_flag( flag, required=False )
		except Error, e:
			if (verbose): print( e )
		env.pop()
		#print()
	# end
	val = choose( 'Fortran 77 compiler', 'F77', choices )
	if (val is not None):
		env.push( val )
	elif (required):
		raise Error( 'Fortran 77 compiler not found' )
# end


# ==============================================================================
# compiler features

# ------------------------------------------------------------------------------
# While many compilers support C99 by default, it's best to explicitly set
# -std=c99 if it exists, to exclude non-standard extensions.
# gcc's current default is gnu11.
def prog_cc_c99( flags=['-std=c99', ''], required=True ):
	assert( lang == 'C' )
	src = 'prog_cc_c99.c'
	
	found = False
	for flag in flags:
		print_dots( '    C99 support: ' + flag, subdots_width )
		save_cflags = env.append( 'CFLAGS', flag )
		try:
			try_compile_run( src )
			found = True
			print( str_yes )
			break
		except Error, e:
			env['CFLAGS'] = save_cflags
			print( str_no )
	# end
	if (not found):
		print_error( '    C99 not supported' )
		if (required):
			raise Error( 'C99 not supported' )
# end


# ------------------------------------------------------------------------------
# While many compilers support C++11 by default, it's best to explicitly set
# -std=c++11 if it exists, to exclude non-standard extensions.
# g++'s current default is gnu++14.
def prog_cxx_cxx11( flags=['-std=c++11', ''], required=True ):
	assert( lang == 'C++' )
	src = 'prog_cxx_cxx11.cxx'
	
	found = False
	for flag in flags:
		print_dots( '    C++11 support: ' + flag, subdots_width )
		save_cxxflags = env.append( 'CXXFLAGS', flag )
		try:
			try_compile_run( src )
			found = True
			print( str_yes )
			break
		except Error, e:
			env['CXXFLAGS'] = save_cxxflags
			print( str_no )
	# end
	if (not found):
		print_error( '    C++11 not supported' )
		if (required):
			raise Error( 'C++11 not supported' )
# end


# ------------------------------------------------------------------------------
# While many compilers support f2008 by default, it's best to explicitly set
# -std=f2008 if it exists, to exclude non-standard extensions.
# gfortran's current default is gnu.
# todo: ifort gives warning: ignoring unknown option '-std=f2008'
def prog_fortran_f2008( flags=['-stand=f08', '-std=f2008'], required=True ):
	assert( lang == 'Fortran' )
	src = 'prog_fortran_f2008.f90'  # without C binding
	
	if (env['CC']):
		push_lang('C')
		c_src = 'prog_fortran_f2008_foo.c'
		try:
			c_obj = try_compile_obj( c_src )
			src = 'prog_fortran_f2008_cbind.f90'  # with C binding
		except Exception:
			c_obj = ''
			print_comment( "    prog_fortran_f2008: compiling C failed; skipping C binding test" )
		pop_lang()
	else:
		c_obj = ''
		print_comment( "    prog_fortran_f2008: no C compiler set; skipping C binding test" )
	# end
	
	found = False
	for flag in flags:
		print_dots( '    Fortran 2008 support: ' + flag, subdots_width )
		save_fcflags = env.append( 'FCFLAGS', flag )
		try:
			try_compile_run( src, [c_obj] )
			found = True
			print( str_yes )
			break
		except Error, e:
			env['FCFLAGS'] = save_fcflags
			print( str_no )
	# end
	if (not found):
		print_error( '    Fortran 2008 not supported' )
		if (required):
			raise Error( 'Fortran 2008 not supported' )
# end


# ------------------------------------------------------------------------------
def compiler_flag( flag, required=True ):
	(cc, flagname) = lang_vars()
	if (lang == 'C'):
		src = 'prog_cc.c'
	elif (lang == 'C++'):
		src = 'prog_cxx.cxx'
	elif (lang == 'Fortran'):
		src = 'prog_fortran.f90'
	elif (lang == 'F77'):
		src = 'prog_f77.f'
	else:
		raise Error( "unknown language " + lang )
	# end
	
	# skip test if flag already included, e.g., by user setting $CFLAGS
	save = env[ flagname ]
	if (flag in save):
		return
	save = env.append( flagname, flag )
	try:
		print_dots( '    Accepts ' + flag, subdots_width )
		try_compile_obj( src )
		print( str_yes )
	except Error, e:
		env[ flagname ] = save
		print( str_no )
		if (required):
			raise Error
# end


# ------------------------------------------------------------------------------
def openmp( flags=['-fopenmp', '-qopenmp', '-openmp', '-omp', ''], required=True ):
	(cc, flagname) = lang_vars()
	if (lang == 'C'):
		src = 'omp_cc.c'
	elif (lang == 'C++'):
		src = 'omp_cc.c'  # same as C
	elif (lang == 'Fortran'):
		src = 'omp_fortran.f90'
	elif (lang == 'F77'):
		src = 'omp_f77.f'
	else:
		raise Error( "unknown language " + lang )
	# end
	
	found = False
	for flag in flags:
		# typically -fopenmp must be specified both when compiling & linking
		save_flags   = env.append( flagname,  flag )
		save_ldflags = env.append( 'LDFLAGS', flag )
		try:
			print_dots( '    OpenMP: ' + flag, subdots_width )
			try_compile_run( src )
			env['OPENMP_' + flagname] = flag
			found = True
			print( str_yes )
			break
		except Error, e:
			print( str_no )
		finally:
			env[ flagname ] = save_flags
			env['LDFLAGS']  = save_ldflags
	# end
	if (not found):
		print_error( '    OpenMP not supported' )
		if (required):
			raise Error( 'OpenMP not supported' )
# end


# ------------------------------------------------------------------------------
def openmp_depend( required=True ):
	(cc, flagname) = lang_vars()
	if (lang == 'C'):
		src = 'omp_depend_cc.c'
	elif (lang == 'C++'):
		src = 'omp_depend_cc.c'  # same as C
	elif (lang == 'Fortran'):
		src = 'omp_depend_fortran.f90'
	elif (lang == 'F77'):
		src = 'omp_depend_f77.f'
	else:
		raise Error( "unknown language " + lang )
	# end
	
	# temporarily add, e.g., OPENMP_CFLAGS to CFLAGS and LDFLAGS
	save_flags   = env.append( flagname,  env['OPENMP_'+flagname] )
	save_ldflags = env.append( 'LDFLAGS', env['OPENMP_'+flagname] )
	try:
		print_dots( '    OpenMP 4 task depend', subdots_width )
		try_compile_run( src )
		env.append( 'DEFS', '-DHAVE_OPENMP_DEPEND' )  # todo compiler-specific DEFS?
		print( str_yes )
	except Error, e:
		print( str_no )
		if (required):
			raise e
	finally:
		env[ flagname ] = save_flags
		env['LDFLAGS']  = save_ldflags
# end


# ------------------------------------------------------------------------------
def openmp_priority( required=True ):
	(cc, flagname) = lang_vars()
	if (lang == 'C'):
		src = 'omp_priority_cc.c'
	elif (lang == 'C++'):
		src = 'omp_priority_cc.c'  # same as C
	elif (lang == 'Fortran'):
		src = 'omp_priority_fortran.f90'
	elif (lang == 'F77'):
		src = 'omp_priority_f77.f'
	else:
		raise Error( "unknown language " + lang )
	# end
	
	# temporarily add, e.g., OPENMP_CFLAGS to CFLAGS and LDFLAGS
	save_flags   = env.append( flagname,  env['OPENMP_'+flagname] )
	save_ldflags = env.append( 'LDFLAGS', env['OPENMP_'+flagname] )
	try:
		print_dots( '    OpenMP 4.5 task priority', subdots_width )
		try_compile_run( src )
		env.append( 'DEFS', '-DHAVE_OPENMP_PRIORITY' )  # todo compiler-specific DEFS?
		print( str_yes )
	except Error, e:
		print( str_no )
		if (required):
			raise e
	finally:
		env[ flagname ] = save_flags
		env['LDFLAGS']  = save_ldflags
# end


# ==============================================================================
# (C)BLAS and LAPACK(E) libraries

# ------------------------------------------------------------------------------
def blas( ilp64=False, required=True ):
	print_header( 'Detecting BLAS libraries' )
	push_lang('C')
	src = 'blas.c'
	
	tests = []
	
	# ----------
	# build list of tests; each test is an environment (hash)
	# of LAPACK_CFLAGS and LAPACK_LIBS.
	cflags = env['LAPACK_CFLAGS']
	libs   = env['LAPACK_LIBS']
	if (cflags or libs):
		# user specified
		print_comment( 'Test using $LAPACK_CFLAGS="' + cflags +
		               '" and $LAPACK_LIBS="' + libs + '"' )
		libs = re.sub( '-L +', '-L', libs )  # keep -Ldir together, not -L dir
		libs = re.split( ' +', libs )
		libdir = filter( lambda x:     x.startswith( '-L' ), libs )
		libs   = filter( lambda x: not x.startswith( '-L' ), libs )
		libdir = join( *libdir )
		libs   = join( *libs )
		tests.append({
			'CFLAGS':  cflags,
			'LIBS':    libs,
		})
	else:
		# included by default (e.g., with Cray cc compilers)
		tests.append( {} )
		
		# plain BLAS
		tests.append( {'LIBS': '-lblas'} )
		
		# MacOS Accelerate
		if (sys.platform == 'darwin'):
			tests.append({
				'DEFS':    '-DHAVE_ACCELERATE',
				'LIBS':    '-framework Accelerate -lm',
			})
		# end
		
		# OpenBLAS
		(inc, libdir) = get_inc_lib( ['OPENBLAS', 'OPENBLASDIR', 'OPENBLAS_DIR'] )
		tests.append({
			'DEFS':    '-DHAVE_OPENBLAS',
			'CFLAGS':  inc,
			'LDFLAGS': libdir,
			'LIBS':    '-lopenblas -lm',
		})
		
		# ATLAS
		(inc, libdir) = get_inc_lib( ['ATLAS', 'ATLASDIR', 'ATLAS_DIR'] )
		for libs in ('-latlas -lm',
		             '-lf77blas -latlas -lm',
		             '-lf77blas -latlas -lgfortran -lm',):
			tests.append({
				'DEFS':    '-DHAVE_ATLAS',
				'CFLAGS':  inc,
				'LDFLAGS': libdir,
				'LIBS':    libs,
			})
		# end
		
		# ACML
		# may be in a subdirectory if user sets
		# $ACMLDIR = /path/to/acml-5.3.1            instead of
		# $ACMLDIR = /path/to/acml-5.3.1/gfortran64
		# modules on titan use ACML_BASE_DIR
		for gf in ('', 'gfortran64', 'gfortran64_mp',
		               'gfortran32', 'gfortran32_mp'):
			(inc, libdir) = get_inc_lib( ['ACML', 'ACMLDIR', 'ACML_DIR', 'ACML_BASE_DIR'], gf )
			if (libdir or not gf):  # if gf != '', require we found lib directory
				if (re.search( '_mp', gf )):
					libs = '-lacml_mp -lm'
				else:
					libs = '-lacml -lm'
				tests.append({
					'DEFS':    '-DHAVE_ACML',
					'CFLAGS':  inc,
					'LDFLAGS': libdir,
					'LIBS':    libs,
				})
			# end
		# end
		
		# MKL has combination of 3 libs:
		# interface lib (gf for gfortran, intel for ifort,
		#                lp64  for 64-bit long-pointer,
		#                ilp64 for 64-bit int-long-pointer),
		# thread lib (gnu_thread for libgomp, intel_thread for iomp5),
		# core lib
		# Not recommended to mix mkl_gf_*lp64 with mkl_intel_thread,
		# or mkl_intel_*lp64 with mkl_gnu_thread.
		# todo: if compiler is GNU, suppress intel_thread version,
		# and if compiler is Intel, suppress gnu_thread versions?
		libs = [
			'-lmkl_gf_lp64 -lmkl_sequential',
			'-lmkl_gf_lp64 -lmkl_gnu_thread',
			'-lmkl_intel_lp64 -lmkl_sequential',
			'-lmkl_intel_lp64 -lmkl_intel_thread',
		]
		if (ilp64):
			libs.extend([
			'-lmkl_gf_ilp64 -lmkl_sequential',
			'-lmkl_gf_ilp64 -lmkl_gnu_thread',
			'-lmkl_intel_ilp64 -lmkl_sequential',
			'-lmkl_intel_ilp64 -lmkl_intel_thread',
			])
		# end
		mkl_set_library_path()
		(inc, libdir) = get_inc_lib( ['MKLROOT'] )
		for lib in libs:
			defs = '-DHAVE_MKL'
			if (re.search( 'ilp64', lib )):
				defs += ' -DMKL_ILP64'
			tests.append({
				'DEFS':    defs,
				'CFLAGS':  inc,
				'LDFLAGS': libdir,
				'LIBS':    join( lib, '-lmkl_core -lm' ),
			})
		# end
	# end
	
	# ----------
	choices = []
	for test in tests:
		save = merge_env( test )
		try:
			print_dots( join_vars( test, 'CFLAGS', 'LDFLAGS', 'LIBS' ))
			try_compile_run( src )
			choices.append( test )
			print( str_yes )
		except Error, e:
			print( str_no )
		finally:
			restore_env( save )
	# end
	pop_lang()
	
	val = choose( 'BLAS library', 'LIBS', choices )
	if (val is not None):
		merge_env( val )
	elif (required):
		raise Error( 'BLAS not found' )
# end


# ------------------------------------------------------------------------------
def cblas( required=True ):
	print_header( 'Detecting CBLAS library' )
	push_lang('C')
	src = 'cblas.c'
	
	# test if (1) cblas in blas library, or (2) cblas in -lcblas
	tests = [
		{},
		{'LIBS': '-lcblas'},
	]
	
	# MacOS Accelerate has cblas.h header buried in Frameworks
	if ('-framework Accelerate' in env['LIBS']):
		try:
			(stdout, stderr) = run(
				'find /System/Library/Frameworks/Accelerate.framework -name cblas.h')
			(path, fname) = os.path.split( stdout )
			tests.append( {'CFLAGS':  '-I'+path} )
		except Error:
			pass
	# end
	
	# add lapack directory
	(inc, libdir) = get_inc_lib( ['LAPACK', 'LAPACKDIR', 'LAPACK_DIR'] )
	if (inc or libdir):
		tests.append({
			'CFLAGS':  inc + '/CBLAS/include',
			'LDFLAGS': libdir,
			'LIBS':    '-lcblas',
		})
	# end
	
	# add cblas directory
	(inc, libdir) = get_inc_lib( ['CBLAS', 'CBLASDIR', 'CBLAS_DIR'] )
	if (inc or libdir):
		tests.append({
			'CFLAGS':  inc,
			'LDFLAGS': libdir,
			'LIBS':    '-lcblas',
		})
	# end
	
	found = False
	for test in tests:
		save = merge_env( test )
		try:
			print_dots( join_vars( test, 'CFLAGS', 'LDFLAGS', 'LIBS' ))
			try_compile_run( src )
			found = True
			print( str_yes )
			break
		except Error, e:
			print( str_no )
			restore_env( save )
	# end
	pop_lang()
	
	if (not found):
		print_error( 'CBLAS not found; see http://www.netlib.org/lapack/' )
		if (required):
			raise Error( 'CBLAS not found' )
# end


# ------------------------------------------------------------------------------
def cblas_enum( required=True ):
	print_dots( 'CBLAS needs typedef enum' )
	src = 'cblas_enum.c'
	
	push_lang('C')
	try:
		try_compile_run( src )
		print( 'no' )
		return
	except Exception, e:
		pass
	finally:
		pop_lang()
	
	push_lang('C')
	try:
		env.append( 'DEFS', '-DCBLAS_ADD_TYPEDEF' )
		try_compile_run( src )
		print( 'yes' )
	except Exception, e:
		print_error( 'unknown' )
		if (required):
			raise e
	finally:
		pop_lang()
# end


# ------------------------------------------------------------------------------
def lapack( required=True ):
	print_header( 'Detecting LAPACK library' )
	push_lang('C')
	src = 'lapack.c'
	
	# test if (1) lapack in blas library, or (2) lapack in -llapack
	tests = [
		{},
		{'LIBS': '-llapack'},
	]
	
	# add lapack directory
	(inc, libdir) = get_inc_lib( ['LAPACK', 'LAPACKDIR', 'LAPACK_DIR'] )
	if (inc or libdir):
		tests.append({
			'CFLAGS':  inc,
			'LDFLAGS': libdir,
			'LIBS':    '-llapack'
		})
		tests.append({
			'CFLAGS':  inc,
			'LDFLAGS': libdir,
			'LIBS':    '-llapack -lgfortran'
		})
	# end
	
	found = False
	for test in tests:
		save = merge_env( test )
		try:
			print_dots( join_vars( test, 'CFLAGS', 'LDFLAGS', 'LIBS' ))
			try_compile_run( src )
			found = True
			print( str_yes )
			break
		except Error, e:
			print( str_no )
			restore_env( save )
	# end
	pop_lang()
	
	if (not found):
		print_error( 'LAPACK not found; see http://www.netlib.org/lapack/' )
		if (required):
			raise Error( 'LAPACK not found' )
# end


# ------------------------------------------------------------------------------
def lapacke( required=True ):
	print_header( 'Detecting LAPACKE library' )
	push_lang('C')
	src = 'lapacke.c'
	
	# test if (1) lapacke in blas library, or (2) lapacke in -llapacke
	tests = [
		{},
		{'LIBS': '-llapacke'},
	]
	
	# add lapack directory
	(inc, libdir) = get_inc_lib( ['LAPACK', 'LAPACKDIR', 'LAPACK_DIR'] )
	if (inc or libdir):
		tests.append({
			'CFLAGS':  inc + '/LAPACKE/include',
			'LDFLAGS': libdir,
			'LIBS':    '-llapacke',
		})
		tests.append({
			'CFLAGS':  inc + '/LAPACKE/include',
			'LDFLAGS': libdir,
			'LIBS':    '-llapacke -lgfortran',
		})
	# end
	
	found = False
	for test in tests:
		save = merge_env( test )
		try:
			print_dots( join_vars( test, 'CFLAGS', 'LDFLAGS', 'LIBS' ))
			try_compile_run( src )
			found = True
			print( str_yes )
			break
		except Error, e:
			print( str_no )
			restore_env( save )
	# end
	pop_lang()
	
	if (not found):
		print_error( 'LAPACKE not found; see http://www.netlib.org/lapack/' )
		if (required):
			raise Error( 'LAPACKE not found' )
# end


# ------------------------------------------------------------------------------
def blas_return_float( required=True ):
	print_dots( 'BLAS return float (e.g., sdot)' )
	
	push_lang('C')
	src = 'blas_return_float.c'
	try:
		try_compile_run( src )
		print( 'returns float (standard)' )
		return
	except Exception, e:
		pass
	finally:
		pop_lang()
	
	push_lang('C')
	src = 'blas_return_float_f2c.c'
	try:
		try_compile_run( src )
		print( 'returns double (f2c, clapack, MacOS Accelerate)' )
		env.append( 'DEFS', '-DBLAS_RETURN_FLOAT_AS_DOUBLE' )
	except Exception, e:
		print_error( 'unknown' )
		if (required):
			raise e
	finally:
		pop_lang()
# end


# ------------------------------------------------------------------------------
def blas_return_complex( required=True ):
	print_dots( 'BLAS return complex (e.g., zdotc)' )
	
	push_lang('C')
	src = 'blas_return_complex.c'
	try:
		try_compile_run( src )
		print( 'returns complex' )
		return
	except Exception, e:
		pass
	finally:
		pop_lang()
	
	push_lang('C')
	src = 'blas_return_complex_intel.c'
	try:
		try_compile_run( src )
		print( 'complex result is first argument' )
		env.append( 'DEFS', '-DBLAS_RETURN_COMPLEX_AS_ARGUMENT' )
	except Exception, e:
		print_error( 'unknown' )
		if (required):
			raise e
	finally:
		pop_lang()
# end


# ------------------------------------------------------------------------------
def lapacke_dlascl( required=True ):
	print_dots( 'LAPACKE_dlascl exists (LAPACK >= 3.6.0)' )
	push_lang('C')
	src = 'lapacke_dlascl.c'
	try:
		try_compile_run( src )
		print( str_yes )
		env.append( 'DEFS', '-DHAVE_LAPACKE_DLASCL' )
	except Exception, e:
		print( str_no )
		if (required):
			raise e
	finally:
		pop_lang()
# end


# ------------------------------------------------------------------------------
def lapacke_dlantr( required=True ):
	print_dots( 'LAPACKE_dlantr works  (LAPACK >= 3.6.1)' )
	push_lang('C')
	src = 'lapacke_dlantr.c'
	try:
		try_compile_run( src )
		print( str_yes )
		env.append( 'DEFS', '-DHAVE_LAPACKE_DLANTR' )
	except Exception, e:
		print( str_no )
		if (required):
			raise e
	finally:
		pop_lang()
# end


# ------------------------------------------------------------------------------
def lapacke_dlassq( required=True ):
	print_dots( 'LAPACKE_dlassq exists (LAPACK >= 3.8.0)' )
	push_lang('C')
	src = 'lapacke_dlassq.c'
	try:
		try_compile_run( src )
		print( str_yes )
		env.append( 'DEFS', '-DHAVE_LAPACKE_DLASSQ' )
	except Exception, e:
		print( str_no )
		if (required):
			raise e
	finally:
		pop_lang()
# end


# ==============================================================================
# utilities for (C)BLAS and LAPACK(E) tests

# ------------------------------------------------------------------------------
def join_vars( env2, *variables ):
	'''
	For variables v1, ..., vn, joins env2[v1], ..., env2[vn] with spaces.
	Ignores variables that don't exist in env2.
	If result is empty, returns "[default flags]".
	Ex: join_vars( test, 'CFLAGS', 'LIBS' ) is similar to:
		join( test.has_key('CFLAGS') and test['CFLAGS'] or '',
			  test.has_key('LIBS')   and test['LIBS']   or '' ) or '[default]'.
	'''
	txt = ''
	for var in variables:
		if (env2.has_key( var )):
			txt += ' ' + env2[ var ]
	txt = txt.strip()
	if (not txt):
		txt = '[default flags]'
	return txt
# end

# ------------------------------------------------------------------------------
def merge_env( env2 ):
	'''
	Appends all key-value pairs in env2 to corresponding key-value pair in env
	(except prepend for $LIBS). Return original key-value pairs from env in a
	map, to restore using restore_env.
	'''
	save = {}
	for key in env2.keys():
		if (key == 'LIBS'):
			save[key] = env.prepend( key, env2[key] )
		else:
			save[key] = env.append( key, env2[key] )
	return save
# end


# ------------------------------------------------------------------------------
def restore_env( save ):
	'''
	Restores all key-value pairs in save to env.
	'''
	for key in save.keys():
		env[key] = save[key]
# end


# ------------------------------------------------------------------------------
def get_inc_lib( variables, subdir=None ):
	'''
	Determines include and library paths using environment variables.
	Ex: (inc, libdir) = get_inc_lib( ['ACML', 'ACML_DIR'], 'gfortran64' )
	Checks if ${ACML} or ${ACML_DIR} exists, uses that as PATH.
	Then sets inc to the first of these that exists:
		PATH/gfortran64/include
		PATH/gfortran64
	and sets libdir to the first of these that exists:
		PATH/gfortran64/lib64
		PATH/gfortran64/lib/intel64
		PATH/gfortran64/lib
		PATH/gfortran64
	'''
	for var in variables:
		inc    = ''
		libdir = ''
		path = env[ var ]
		if (path and os.path.exists( path )):
			inc    = '-I${' + var + '}'
			libdir = '-L${' + var + '}'
			if (subdir):
				path = os.path.join( path, subdir )
				if (not os.path.exists( path )):
					inc    = ''
					libdir = ''
					continue
				inc    += '/' + subdir
				libdir += '/' + subdir
			# end
			for lib in ('lib64', 'lib/intel64', 'lib'):
				path_lib = os.path.join( path, lib )
				if (os.path.exists( path_lib )):
					libdir += '/' + lib
					break
			# end
			path_inc = os.path.join( path, 'include' )
			if (os.path.exists( path_inc )):
				inc += '/include'
			break
		# end
	# end
	return (inc, libdir)
# end


# ------------------------------------------------------------------------------
def mkl_set_library_path():
	'''
	MKL needs (DY)LD_LIBRARY_PATH set or it won't run,
	but (DY)LD_LIBRARY_PATH may not be passed to python or make,
	so if it isn't set, try setting it ourselves.
	'''
	mklroot = env['MKLROOT']
	LD_LIBRARY_PATH = 'LD_LIBRARY_PATH'
	if (sys.platform == 'darwin'):
		LD_LIBRARY_PATH = 'DYLD_LIBRARY_PATH'
	if (mklroot and not os.environ.has_key( LD_LIBRARY_PATH )):
		(inteldir, mkl) = os.path.split( mklroot )
		intel_compiler = os.path.join( inteldir, 'compiler' )
		if (os.path.exists( intel_compiler )):
			inteldir = intel_compiler
		paths = []
		for lib in ('lib64', 'lib/intel64', 'lib'):
			intel_lib = os.path.join( inteldir, lib )
			if (os.path.exists( intel_lib )):
				paths.append( intel_lib )
				break
		# end
		for lib in ('lib64', 'lib/intel64', 'lib'):
			mkl_lib = os.path.join( mklroot, lib )
			if (os.path.exists( mkl_lib )):
				paths.append( mkl_lib )
				break
		# end
		if (paths):
			paths = ':'.join( paths )
			os.environ[ LD_LIBRARY_PATH ] = paths
			print_comment( 'Setting '+ LD_LIBRARY_PATH +'='+ paths )
		# end
	# end
# end


# ==============================================================================
# lower level compilation & execution

# ------------------------------------------------------------------------------
def lang_vars():
	if (lang == 'C'):
		cc = env['CC']
		flagname = 'CFLAGS'
	elif (lang == 'C++'):
		cc = env['CXX']
		flagname = 'CXXFLAGS'
	elif (lang == 'Fortran'):
		cc = env['FC']
		flagname = 'FCFLAGS'
	elif (lang == 'F77'):
		cc = env['F77']
		flagname = 'FFLAGS'
	else:
		raise Error( "unknown language " + lang )
	# end
	return (cc, flagname)
# end


# ------------------------------------------------------------------------------
def try_compiler( src, required=True ):
	(cc, flagname) = lang_vars()
	try:
		print_dots( cc )
		(stdout, stderr) = run([ 'which', cc ])
		ccpath = stdout.strip()
		if (sys.platform == 'darwin'):
			# due to case-insensitive filesystem, `which CC` returns
			# non-existent /usr/bin/CC on MacOS; ignore it
			(ccdir, exe) = os.path.split( ccpath )
			if (not cc in os.listdir( ccdir )):
				raise Error
		# end
		print( ccpath )
		
		print_dots( '    compile and run test', subdots_width )
		try_compile_run( src )
		print( str_yes )
	except Error, e:
		print( str_no )
		if (required):
			raise e
# end


# ------------------------------------------------------------------------------
def try_compile_run( src, extra_objs=[] ):
	exe = try_compile_exe( src, extra_objs )
	run( './'+exe )
# end


# ------------------------------------------------------------------------------
def try_compile_obj( src ):
	(cc, flagname) = lang_vars()
	flags   = env[ flagname ]
	openmp  = env['OPENMP_' + flagname]
	defs    = env['DEFS']
	
	write_test( src )
	
	(exe, ext) = os.path.splitext( src )
	exe += '_' + cc
	obj = exe + '.o'
	run([ cc, flags, defs, '-c', src, '-o', obj ])
	
	return obj
# end


# ------------------------------------------------------------------------------
def try_compile_exe( src, extra_objs=[] ):
	(cc, flagname) = lang_vars()
	flags   = env[ flagname ]
	openmp  = env['OPENMP_' + flagname]
	defs    = env['DEFS']
	ldflags = env['LDFLAGS']
	libs    = env['LIBS']
	
	write_test( src )
	
	(exe, ext) = os.path.splitext( src )
	exe += '_' + cc
	obj = exe + '.o'
	run([ cc, flags, openmp, defs, '-c', src, '-o', obj ])
	run([ cc, ldflags, openmp, obj, extra_objs, libs, '-o', exe ])
	
	return exe
# end


# ------------------------------------------------------------------------------
def run( cmd ):
	if (not isinstance( cmd, str )):
		cmd = ' '.join( flatten( cmd ))
	
	if (verbose):
		dots = ' ' * max( 0, 71 - len(cmd) )
		print( font_comment + '    >>', cmd, dots, font_normal, end='' )
	
	# Popen propogates (DY)LD_LIBRARY_FLAGS, unlike os.system,
	# but it doesn't substitute variables, so do that ourselves.
	# Prefer ${xyz} syntax, which both shell and make recognize.
	cmd = re.sub( r'\$\{(\w+)\}', sub_env, cmd )  # shell & make
	cmd = re.sub( r'\$\((\w+)\)', sub_env, cmd )  # make only
	cmd = re.sub( r'\$(\w+)',     sub_env, cmd )  # shell only
	
	cmd_list = shlex.split( cmd )
	proc = subprocess.Popen( cmd_list, stdout=PIPE, stderr=PIPE )
	(stdout, stderr) = proc.communicate()
	if (verbose > 1):
		sys.stdout.write( stdout )
		sys.stderr.write( stderr )
	rc = proc.wait()
	if (verbose):
		if (rc != 0):
			print_comment( 'failed' )
		else:
			print_comment( 'ok' )
	if (rc != 0):
		raise Error
	return (stdout, stderr)
# end


# ==============================================================================
# utilities

# ------------------------------------------------------------------------------
def choose( name, key, choices ):
	'''
	Asks user to choose one of choices, and returns it.
	
	name        Used for prompts, e.g.,
				name='C++ compiler' generates prompt
				'Which C++ compiler to use?'
	choices     List to choose from.
	'''
	#print( 'choose', name, choices )
	comment = ''
	num = len(choices)
	if (num == 0):
		print_error( name + ' not found' )
		return None
	elif (num == 1):
		i = 0
	elif (auto):
		i = 0
		comment = ' [auto]'
	else:
		print( 'Available choices for', name +':' )
		for i in xrange(num):
			print( '%3d) %s' % (i+1, choices[i][key]) )
		# end
		while (True):
			print( 'Which', name, 'to use [1-%d, or quit]? ' % (num), end='' )  # no newline
			reply = raw_input()
			if (re.search( 'q|quit', reply, re.I )):
				raise Error('cancelled')
			try:
				i = int(reply) - 1
				assert( i >= 0 and i < num )
				break
			except:
				print( 'invalid input' )
		# end
	# end
	print( font_bold + green + 'Using '+ name + ': ' + choices[i][key] +
	       comment + font_normal )
	return choices[i]
# end


# ------------------------------------------------------------------------------
def print_dots( txt, width=dots_width ):
	dots = '.' * max( 3, width - len(txt) )
	end = ' ' if (not verbose) else '\n'
	print( txt, dots, end=end )


def print_header( txt ):
	print( font_header + '\n' + '='*80 + '\n' + txt + font_normal )


def print_subhead( txt ):
	print( font_subhead + txt + font_normal )


def print_comment( txt ):
	print( font_comment + txt + font_normal )


def print_error( txt ):
	print( red + txt + font_normal )


# ------------------------------------------------------------------------------
def sub_env_empty( match ):
	'''
	Replaces an environment variable.
	For use in: re.sub( pattern, sub_env_empty, txt ).
	Returns contents of variable match.group(1),
	or '' if variable doesn't exist.
	'''
	return env[ match.group(1) ]
# end


# ------------------------------------------------------------------------------
def sub_env( match ):
	'''
	Replaces an environment variable.
	For use in: re.sub( pattern, sub_env_empty, txt ).
	Returns contents of variable match.group(1),
	or the original match.group(0) if variable doesn't exist.
	'''
	key = match.group(1)
	val = env.get( key )
	if (val is None):
		return match.group(0)  # no change
	else:
		return val
# end


# ------------------------------------------------------------------------------
def sub_define( match ):
	'''
	Called as repl in re.sub( pattern, sub_define, txt ).
	Returns:
		#define variable value
	for variable match.group(1), or commented out:
		/* variable */
	if variable doesn't exist.
	'''
	key = match.group(1)
	defs = env['DEFS']
	m = re.search( r'-D' + key + r'(\S*)', defs )
	if (m):
		return join( '#define', key, m.group(1) )
	else:
		return join( '/*', match.group(0), '*/' )  # comment out
# end


# ------------------------------------------------------------------------------
g_written = {}

def write_test( filename ):
	'''
	Write g_test_files[ filename ] to file, the first time this is called with
	that filename. Subsequent calls with the same filename do nothing.
	'''
	if (not g_written.has_key( filename )):
		g_written[ filename ] = True
		write( filename, g_test_files[ filename ] )
	# end
# end


# ------------------------------------------------------------------------------
def write( filename, txt ):
	'''
	Write txt to file.
	'''
	f = open( filename, 'w' )
	f.write( txt )
	f.close()
# end


# ------------------------------------------------------------------------------
def read( filename ):
	'''
	Read file and return its contents.
	'''
	f = open( filename, 'r' )
	txt = f.read()
	f.close()
	return txt
# end


# ------------------------------------------------------------------------------
def join( *args ):
	'''
	Joins its arguments with space.
	Ex: join( "foo", "bar", "baz" ) returns "foo bar baz"
	'''
	return ' '.join( args ).strip()
# end


# ------------------------------------------------------------------------------
def unique( lst ):
	'''
	Returns first of each unique item from lst, without changing order.
	Ex: unique( [ 1, 2, 1, 2, 3, 4, 3 ] ) returns [ 1, 2, 3, 4 ]
	'''
	lst2 = []
	for x in lst:
		if (not x in lst2):
			lst2.append( x )
	return lst2
# end


# ------------------------------------------------------------------------------
def flatten( l, ltypes=(list, tuple) ):
	'''
	Flattens nested list or tuple.
	Ex: flatten( [1, 2, [3, [4, 5], 6]] ) returns [1, 2, 3, 4, 5, 6]
	
	see http://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html
	'''
	ltype = type(l)
	l = list(l)
	i = 0
	while i < len(l):
		while isinstance(l[i], ltypes):
			if not l[i]:
				l.pop(i)
				i -= 1
				break
			else:
				l[i:i + 1] = l[i]
		i += 1
	return ltype(l)
# end


# ==============================================================================
# common code

# ------------------------------------------------------------------------------
# utilities prepended to some tests that call Fortran
fortran_mangling = r'''
#if defined( MKL_ILP64 ) || defined( ILP64 )
    typedef long long myint;
#else
    typedef int myint;
#endif

#if defined( LOWERCASE )
    #define FORTRAN_NAME( lower, UPPER ) lower
#elif defined( UPPERCASE )
    #define FORTRAN_NAME( lower, UPPER ) UPPER
#else
    #define FORTRAN_NAME( lower, UPPER ) lower ## _
#endif

#ifdef __cplusplus
    #define EXTERN_C extern "C"
#else
    #define EXTERN_C
#endif
'''



# ==============================================================================
# All the test codes are saved in g_test_files map, with filenames as keys.
# These are written to disk using write_test().
# ==============================================================================

g_test_files = {

# ==============================================================================
# compilers

# ------------------------------------------------------------------------------
'prog_cc.c': r'''
#include <stdio.h>

int main( int argc, char** argv )
{
    int x = 1;
    printf( "hello, x=%d\n", x );
    return 0;
}
''',

# ------------------------------------------------------------------------------
'prog_cc_c99.c': r'''
#include <stdio.h>

#if __STDC_VERSION__ >= 199901L
    // supports C99
#else
    choke function();
#endif

int main( int argc, char** argv )
{
    printf( "hello, __STDC_VERSION__ = %ld\n", __STDC_VERSION__ );
    for (int i = 0; i < 10; ++i) {
        // pass
    }
    return 0;
}
''',

# ------------------------------------------------------------------------------
'prog_cxx.cxx': r'''
#include <iostream>

class Simple {
public:
    Simple(): x(1) {}
    int x;
};

int main( int argc, char** argv )
{
    Simple s;
    std::cout << "hello, x=" << s.x << "\n";
    return 0;
}
''',

# ------------------------------------------------------------------------------
'prog_cxx_cxx11.cxx': r'''
#include <iostream>

#if __cplusplus >= 201103L
    // supports C++11
#else
    choke function();
#endif

enum class Foo { a, b };
enum Bar : char { c, d };

int main( int argc, char** argv )
{
    int *y = nullptr;
    y = new int[3] { 0, 1, 2 };
    auto x = 1.23;
    std::cout << "hello, x=" << x << ", __cplusplus = " << __cplusplus << "\n";
    return 0;
}
''',

# ------------------------------------------------------------------------------
'prog_fortran.f90': r'''
program main
    implicit none
    print '(a)', 'hello'
end program main
''',

# ------------------------------------------------------------------------------
'prog_fortran_f2008.f90': r'''
program main
    use iso_c_binding
    implicit none

    integer(c_int), parameter :: x = 100
    print '(a,i3)', 'hello', x
end program main
''',

# ------------------------------------------------------------------------------
'prog_fortran_f2008_cbind.f90': r'''
program main
    use iso_c_binding
    implicit none

interface
    subroutine foo( x ) &
    bind( C, name="foo" )
        use iso_c_binding
        integer(c_int), value :: x
    end subroutine foo
end interface

    integer(c_int), parameter :: x = 100
    print '(a,i3)', 'hello', x
    call foo( x )
end program main
''',

# ------------------------------------------------------------------------------
'prog_fortran_f2008_foo.c': r'''
#include <stdio.h>

void foo( int x )
{
    printf( "%s( %d )\n", __func__, x );
}
''',

# ------------------------------------------------------------------------------
'prog_f77.f': r'''
      program main
          implicit none
          print '(a)', 'hello'
      end program main
''',

# ==============================================================================
# OpenMP

# ------------------------------------------------------------------------------
'omp_cc.c': r'''
#include <stdio.h>
#include <omp.h>

int main( int argc, char** argv )
{
    int x[10], nt;

    #pragma omp parallel
    nt = omp_get_num_threads();

    #pragma omp parallel for
    for (int i = 0; i < 10; ++i) {
        x[i] = i;
    }
    printf( "openmp x[0]=%d, nt=%d\n", x[0], nt );
    return 0;
}
''',

# ------------------------------------------------------------------------------
'omp_fortran.f90': r'''
program main
    use omp_lib
    implicit none
    integer :: x(10), nt, i

    !$omp parallel
    nt = omp_get_num_threads()
    !$omp end parallel

    !$omp parallel do
    do i = 1, 10
        x(i) = i
    end do
    print '(a,i3,a,i3)', 'openmp x(1)=', x(1), 'nt=', nt
end program
''',

# ------------------------------------------------------------------------------
'omp_f77.f': r'''
      program main
          implicit none
          integer  omp_get_num_threads
          external omp_get_num_threads
          integer x(10), nt, i

!$omp     parallel
          nt = omp_get_num_threads()
!$omp       end parallel

!$omp     parallel do
          do i = 1, 10
              x(i) = i
          end do
          print '(a,i3,a,i3)', 'openmp x(1)=', x(1), 'nt=', nt
      end program
''',

# ------------------------------------------------------------------------------
'omp_depend_cc.c': r'''
#include <stdio.h>

void task( int n, int* x, int id )
{
    for (int i = 0; i < n; ++i) {
        x[i] = id + i;
    }
}

int main( int argc, char** argv )
{
    int n = 1000, x[1000] = { 0 };
    int last = 1000;
    for (int iter = 0; iter < 100; ++iter) {
        // inserts last/10 tasks that update x
        #pragma omp parallel
        {
            for (int i = 0; i <= last; i += 10) {
                #pragma omp task depend(inout:x[0:n])
                task( n, x, i );
            }
        }
        // verify that updates worked
        for (int i = 0; i < n; ++i) {
            int expect = last + i;
            if (x[i] != expect) {
                printf( "openmp task depend failed, x[%d] = %d, expected %d (iter %d)\n",
                        i, x[i], expect, iter );
                return 1;
            }
        }
    }
    printf( "openmp task depend seems ok\n" );
    return 0;
}
''',

# ------------------------------------------------------------------------------
'omp_depend_fortran.f90': r'''
program main
    use omp_lib
    implicit none
    integer, parameter :: n = 1000, last = 1000
    integer :: x(n), iter, i, expect

    do iter = 1, 100
        !! inserts last/10 tasks that update x
        !$omp parallel
        do i = 0, last, 10
            !$omp task depend(inout:x(1:n))
            call task( n, x, i )
            !$omp end task
        end do
        !$omp end parallel

        !! verify that updates worked
        do i = 1, n
            expect = last + i
            if (x(i) .ne. expect) then
                print '(a,i4,a,i4)', 'openmp task depend failed, x(', i, ') = ', x(i)
                stop 1
            endif
        end do
    end do
    print '(a)', 'openmp task depend seems ok'
end program

subroutine task( n, x, id )
    integer :: n, x(n), id
    integer :: i
    do i = 1, n
        x(i) = id + i
    end do
end subroutine task
''',

# ------------------------------------------------------------------------------
'omp_depend_f77.f': r'''
      program main
          implicit none
          integer n, last
          parameter (n = 1000, last = 1000)
          integer x(n), iter, i, expect

          do iter = 1, 100
!!            inserts last/10 tasks that update x
!$omp         parallel
              do i = 0, last, 10
!$omp             task depend(inout:x(1:n))
                  call task( n, x, i )
!$omp             end task
              end do
!$omp         end parallel

!!            verify that updates worked
              do i = 1, n
                  expect = last + i
                  if (x(i) .ne. expect) then
                      print '(a,i4,a,i4)',
     c                      'openmp task depend failed, x(',
     c                      i, ') = ', x(i)
                      stop 1
                  endif
              end do
          end do
          print '(a)', 'openmp task depend seems ok'
      end program

      subroutine task( n, x, id )
          integer n, x(n), id
          integer i
          do i = 1, n
              x(i) = id + i
          end do
      end subroutine task
''',

# ------------------------------------------------------------------------------
'omp_priority_cc.c': r'''
#include <stdio.h>

void task( int n, int* x, int id )
{
    for (int i = 0; i < n; ++i) {
        x[i] = id + i;
    }
}

int main( int argc, char** argv )
{
    int n = 1000, x[1000] = { 0 };
    #pragma omp parallel
    {
        #pragma omp task depend(inout:x[0:n]) priority(1)
        task( n, x, 0 );

        #pragma omp task depend(inout:x[0:n]) priority(2)
        task( n, x, 100 );
    }
    for (int i = 0; i < n; ++i) {
        if (x[i] != 100 + i) {
            printf( "openmp task priority failed, x[%d] = %d, expected %d\n",
                    i, x[i], 100 + i );
            return 1;
        }
    }
    printf( "openmp task priority ok\n" );
    return 0;
}
''',

# ------------------------------------------------------------------------------
'omp_priority_fortran.f90': r'''
program main
    use omp_lib
    implicit none
    integer, parameter :: n = 1000
    integer :: x(n), i, expect

    !$omp parallel
        !! todo: verify priority syntax
        !$omp task depend(inout:x(1:n)), priority(1)
        call task( n, x, 0 );
        !$omp end task

        !$omp task depend(inout:x(1:n)), priority(2)
        call task( n, x, 100 );
        !$omp end task
    !$omp end parallel

    do i = 1, n
        expect = 100 + i
        if (x(i) .ne. expect) then
            print '(a,i4,a,i4,a,i4)', 'openmp task priority failed, x(', &
                  i, ') = ', x(i), ', expected ', expect
            stop 1
        endif
    end do
    print '(a)', 'openmp task priority ok'
end program

subroutine task( n, x, id )
    integer :: n, x(n), id
    integer :: i
    do i = 1, n
        x(i) = id + i
    end do
end subroutine
''',

# ------------------------------------------------------------------------------
'omp_priority_f77.f': r'''
      program main
          implicit none
          integer n
          parameter (n = 1000)
          integer x(n), i, expect

!$omp     parallel
!!        todo: verify priority syntax
!$omp         task depend(inout:x(1:n)), priority(1)
              call task( n, x, 0 );
!$omp         end task

!$omp         task depend(inout:x(1:n)), priority(2)
              call task( n, x, 100 );
!$omp         end task
!$omp     end parallel

          do i = 1, n
              expect = 100 + i
              if (x(i) .ne. expect) then
                  print '(a,i4,a,i4,a,i4)',
     c                  'openmp task priority failed, x(',
     c                  i, ') = ', x(i), ', expected ', expect
                  stop 1
              endif
          end do
          print '(a)', 'openmp task priority ok'
      end program

      subroutine task( n, x, id )
          integer n, x(n), id
          integer i
          do i = 1, n
              x(i) = id + i
          end do
      end subroutine
''',

# ==============================================================================
# library

# ------------------------------------------------------------------------------
'blas.c': fortran_mangling + r'''
#include <stdio.h>

#define sgemm FORTRAN_NAME( sgemm, SGEMM )

EXTERN_C
void sgemm( const char* transA, const char* transB,
            const myint* m, const myint* n, const myint* k,
            const float* alpha,
            const float* A, const myint* lda,
            const float* B, const myint* ldb,
            const float* beta,
                  float* C, const myint* ldc );

int main( int argc, char** argv )
{
    // A is 4x2 embedded in 4x2 array
    // B is 2x3 embedded in 3x3 array
    // C is 4x3 embedded in 5x3 array
    // D = alpha*A*B + beta*C
    myint i, j;
    myint m = 4, n = 3, k = 2, lda = 4, ldb = 3, ldc = 5;
    float alpha = 2, beta = -1;
    float A[ 5*2 ] = { 1, 2, 3, 4,   4, 1, 2, 3 };
    float B[ 3*3 ] = { 1, 3, 0,   2, 1, 0,   3, 2, 0 };
    float C[ 5*3 ] = { 1, 2, 3, 4, 0,   4, 1, 2, 3, 0,   3, 4, 1, 2, 0 };
    float D[ 5*3 ] = { 25, 8, 15, 22, 0,   8, 9, 14, 19, 0,   19, 12, 25, 34 };
    sgemm( "no", "no", &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
    // check C == D
    for (i = 0; i < ldc*n; ++i) {
        if (C[i] != D[i]) {
            printf( "sgemm failed: C[%d] %.2f != D[%d] %.2f\n",
                    i, C[i], i, D[i] );
            return 1;
        }
    }
    printf( "sgemm ok\n" );
    return 0;
}
''',

# ------------------------------------------------------------------------------
'cblas.c': r'''
#include <stdio.h>

#ifdef HAVE_MKL
    #include <mkl_cblas.h>
#else
    #include <cblas.h>
#endif

int main( int argc, char** argv )
{
    // A is 4x2 embedded in 4x2 array
    // B is 2x3 embedded in 3x3 array
    // C is 4x3 embedded in 5x3 array
    // D = alpha*A*B + beta*C
    int i, j;
    int m = 4, n = 3, k = 2, lda = 4, ldb = 3, ldc = 5;
    float alpha = 2, beta = -1;
    float A[ 5*2 ] = { 1, 2, 3, 4,   4, 1, 2, 3 };
    float B[ 3*3 ] = { 1, 3, 0,   2, 1, 0,   3, 2, 0 };
    float C[ 5*3 ] = { 1, 2, 3, 4, 0,   4, 1, 2, 3, 0,   3, 4, 1, 2, 0 };
    float D[ 5*3 ] = { 25, 8, 15, 22, 0,   8, 9, 14, 19, 0,   19, 12, 25, 34 };
    cblas_sgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
                 m, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
    // check C == D
    for (i = 0; i < ldc*n; ++i) {
        if (C[i] != D[i]) {
            printf( "cblas_sgemm failed: C[%d] %.2f != D[%d] %.2f\n",
                    i, C[i], i, D[i] );
            return 1;
        }
    }
    printf( "cblas_sgemm ok\n" );
    return 0;
}
''',

# ------------------------------------------------------------------------------
'cblas_enum.c': r'''
#include <stdio.h>

#ifdef HAVE_MKL
    #include <mkl_cblas.h>
#else
    #include <cblas.h>
#endif

#ifdef CBLAS_ADD_TYPEDEF
typedef enum CBLAS_TRANSPOSE CBLAS_TRANSPOSE;
#endif

int main( int argc, char** argv )
{
    CBLAS_TRANSPOSE trans;
    printf( "CBLAS_TRANSPOSE ok\n" );
    return 0;
}
''',

# ------------------------------------------------------------------------------
'blas_return_float.c': fortran_mangling + r'''
#include <stdio.h>

#define sdot FORTRAN_NAME( sdot, SDOT )

EXTERN_C
float sdot( const myint* n,
            const float* x, const myint* incx,
            const float* y, const myint* incy );

int main( int argc, char** argv )
{
    myint n = 5, ione = 1;
    float x[5] = { 1, 2, 3, 4, 5 };
    float y[5] = { 5, 4, 3, 2, 1 };
    float expect = 35;
    float result = sdot( &n, x, &ione, y, &ione );
    myint okay = (result == expect);
    printf( "sdot result %.2f, expect %.2f, %s\n",
            result, expect, (okay ? "ok" : "failed"));
    return ! okay;
}
''',

# ------------------------------------------------------------------------------
# f2c has sdot return double instead of float; appears in MacOS Accelerate
'blas_return_float_f2c.c': fortran_mangling + r'''
#include <stdio.h>

#define sdot FORTRAN_NAME( sdot, SDOT )

EXTERN_C
double sdot( const myint* n,
             const float* x, const myint* incx,
             const float* y, const myint* incy );

int main( int argc, char** argv )
{
    myint n = 5, ione = 1;
    float x[5] = { 1, 2, 3, 4, 5 };
    float y[5] = { 5, 4, 3, 2, 1 };
    float expect = 35;
    float result = sdot( &n, x, &ione, y, &ione );
    myint okay = (result == expect);
    printf( "sdot result %.2f, expect %.2f, %s\n",
            result, expect, (okay ? "ok" : "failed"));
    return ! okay;
}
''',

# ------------------------------------------------------------------------------
'blas_return_complex.c': fortran_mangling + r'''
#include <stdio.h>
#include <complex.h>

#define zdotc FORTRAN_NAME( zdotc, ZDOTC )

EXTERN_C
double _Complex zdotc( const myint* n,
                       const double _Complex* x, const myint* incx,
                       const double _Complex* y, const myint* incy );

int main( int argc, char** argv )
{
    myint n = 5, ione = 1;
    double _Complex x[5] = { 1, 2, 3, 4, 5 };
    double _Complex y[5] = { 5, 4, 3, 2, 1 };
    double _Complex expect = 35;
    double _Complex result = zdotc( &n, x, &ione, y, &ione );
    myint okay = (result == expect);
    printf( "zdotc result %.2f, expect %.2f, %s\n",
            creal(result), creal(expect), (okay ? "ok" : "failed"));
    return ! okay;
}
''',

# ------------------------------------------------------------------------------
# Intel Fortran complex number return convention
# see https://software.intel.com/en-us/node/528406
# "Calling BLAS Functions that Return the Complex Values in C/C++ Code"
'blas_return_complex_intel.c': fortran_mangling + r'''
#include <stdio.h>
#include <complex.h>

#define zdotc FORTRAN_NAME( zdotc, ZDOTC )

EXTERN_C
void zdotc( double _Complex* result,
            const myint* n,
            const double _Complex* x, const myint* incx,
            const double _Complex* y, const myint* incy );

int main( int argc, char** argv )
{
    myint n = 5, ione = 1;
    double _Complex x[5] = { 1, 2, 3, 4, 5 };
    double _Complex y[5] = { 5, 4, 3, 2, 1 };
    double _Complex expect = 35;
    double _Complex result;
    zdotc( &result, &n, x, &ione, y, &ione );
    myint okay = (result == expect);
    printf( "zdotc result %.2f, expect %.2f, %s\n",
            creal(result), creal(expect), (okay ? "ok" : "failed"));
    return ! okay;
}
''',

# ------------------------------------------------------------------------------
'lapack.c': fortran_mangling + r'''
#include <stdio.h>

#define dpotrf FORTRAN_NAME( dpotrf, DPOTRF )

EXTERN_C
void dpotrf( const char* uplo, const myint* n,
             double* A, const myint* lda,
             myint* info );

int main( int argc, char** argv )
{
    myint i, n = 2, info = 0;
    double A[2*2] = { 16, 4,   -1, 5 };
    double L[2*2] = {  4, 1,   -1, 2 };
    double work[1];
    dpotrf( "lower", &n, A, &n, &info );
    if (info != 0) {
        printf( "dpotrf failed: info %d\n", info );
        return 1;
    }
    for (i = 0; i < n*n; ++i) {
        if (A[i] != L[i]) {
            printf( "dpotrf failed: A[%d] %.2f != L[%d] %.2f\n",
                    i, A[i], i, L[i] );
            return 1;
        }
    }
    printf( "dpotrf ok\n" );
    return 0;
}
''',

# ------------------------------------------------------------------------------
'lapacke.c': r'''
#include <stdio.h>

#ifdef HAVE_MKL
    #include <mkl_lapacke.h>
#else
    #include <lapacke.h>
#endif

int main( int argc, char** argv )
{
    int i, n = 2, info = 0;
    double A[2*2] = { 16, 4,   -1, 5 };
    double L[2*2] = {  4, 1,   -1, 2 };
    double work[1];
    info = LAPACKE_dpotrf_work( LAPACK_COL_MAJOR, 'L', n, A, n );
    if (info != 0) {
        printf( "dpotrf failed: info %d\n", info );
        return 1;
    }
    for (i = 0; i < n*n; ++i) {
        if (A[i] != L[i]) {
            printf( "dpotrf failed: A[%d] %.2f != L[%d] %.2f\n",
                    i, A[i], i, L[i] );
            return 1;
        }
    }
    printf( "dpotrf ok\n" );
    return 0;
}
''',

# ------------------------------------------------------------------------------
'lapacke_dlascl.c': r'''
#include <stdio.h>

#ifdef HAVE_MKL
    #include <mkl_lapacke.h>
#else
    #include <lapacke.h>
#endif

int main( int argc, char** argv )
{
    int i, m = 4, n = 3, info = 0;
    double A[4*3] = { 1, 2, 3, 4,    5,  6,  7,  8,    9, 10, 11, 12 };
    double D[4*3] = { 2, 4, 6, 8,   10, 12, 14, 16,   18, 20, 22, 24 };
    info = LAPACKE_dlascl_work( LAPACK_COL_MAJOR, 'g', -1, -1, 1.0, 2.0, m, n, A, m );
    if (info != 0) {
        printf( "dlascl failed: info %d\n", info );
        return 1;
    }
    for (i = 0; i < m*n; ++i) {
        if (A[i] != D[i]) {
            printf( "dlascl failed: A[%d] %.2f != D[%d] %.2f\n",
                    i, A[i], i, D[i] );
            return 1;
        }
    }
    printf( "dlascl ok\n" );
    return 0;
}
''',

# ------------------------------------------------------------------------------
'lapacke_dlantr.c': r'''
#include <stdio.h>

#ifdef HAVE_MKL
    #include <mkl_lapacke.h>
#else
    #include <lapacke.h>
#endif

int main( int argc, char** argv )
{
    int n = 3;
    double A[3*3] = { 1, 2, 3,   4, 5, 6,   7, 8, 9 };
    double work[1];
    double expect = 11;
    double result = LAPACKE_dlantr_work( LAPACK_COL_MAJOR, '1', 'L', 'N', n, n, A, n, work );
    int okay = (result == expect);
    printf( "dlantr result %.2f, expect %.2f, %s\n",
            result, expect, (okay ? "ok" : "failed"));
    return ! okay;
}
''',

# ------------------------------------------------------------------------------
'lapacke_dlassq.c': fortran_mangling + r'''
#include <stdio.h>

#ifdef HAVE_MKL
    #include <mkl_lapacke.h>
#else
    #include <lapacke.h>
#endif

#define dlassq FORTRAN_NAME( dlassq, DLASSQ )

//EXTERN_C
//void dlassq( const myint* n, const double* x, const myint* incx,
//             double* scale, double* sumsq );

int main( int argc, char** argv )
{
    int n = 3, incx = 1;
    double x[3] = { 1, 2, 2 };
    double scale, sumsq;
    double expect = 9;
    LAPACKE_dlassq_work( n, x, incx, &scale, &sumsq );
    //dlassq( &n, x, &incx, &scale, &sumsq );
    double result = sumsq*scale*scale;
    int okay = (result == expect);
    printf( "dlassq result %.2f, expect %.2f, %s\n",
            result, expect, (okay ? "ok" : "failed"));
    return ! okay;
}
''',

}  # end g_test_files
