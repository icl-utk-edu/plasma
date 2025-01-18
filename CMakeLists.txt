cmake_minimum_required( VERSION 3.12 FATAL_ERROR )

project( PLASMA VERSION 24.8.7 LANGUAGES C
    HOMEPAGE_URL "https://github.com/icl-utk-edu/plasma" DESCRIPTION "Software library for solving dense linear algebra systems using OpenMP")

set(CMAKE_SUPPRESS_REGENERATION on)

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.31.0)
    cmake_policy( SET CMP0171 NEW ) # recognize CMake's `codegen` target
    set( CODEGEN "CODEGEN" )
endif()

if (${CMAKE_VERSION} VERSION_GREATER 3.11.99)
  cmake_policy(PUSH)
  cmake_policy(SET CMP0074 NEW) # allows to use CBLAS_ROOT and LAPACKE_ROOT
endif()

#set( CMAKE_THREAD_PREFER_PTHREAD 1 )
#find_package( Threads )

include( FindPython )  # requires CMake 3.12
if (Python_FOUND)
    message( STATUS "Found Python interpreter wth ID ${Python_INTERPRETER_ID} and EXE ${Python_EXECUTABLE}" )
else()
    message( FATAL_ERROR "Couldn't find Python interpreter, cannot generate all precision files." )
endif()

# PLASMA uses C99 features (in-loop definition of for loop variables)
if (CMAKE_VERSION VERSION_LESS "3.1")
  if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
     set (CMAKE_C_FLAGS "-std=gnu99 ${CMAKE_C_FLAGS}")
   endif ( CMAKE_C_COMPILER_ID )
else ()
   set ( CMAKE_C_STANDARD 99 )
endif ()

set(BLA_PREFER_PKGCONFIG ON)  # for CMake 3.11 or newer for BLAS and 3.20+ for LAPACK

# use standard module to find BLAS (calls CMake's FindThreads.cmake if necessary)
# typically succeeds when `sgemm' is found
# use BLA_VENDOR (by default "All") "Intel" "Intel10_32" "Intel10_64lp" "Intel10_64p_seq" "IBMESSL" "Generic"
find_package( BLAS )
if (BLAS_FOUND)
  message( STATUS "Found BLAS libraries ${BLAS_LIBRARIES}" )
else ()
  message( STATUS "BLAS not found. Set BLAS_LIBRARIES in command line with -D option or with GUI." )
  find_package( BLAS REQUIRED ) # this is guaranteed to fail with more info for the user
endif ()

# use standard module to find LAPACK (calls FindBLAS.cmake if necessary)
# typically succeeds when `cheev' is found
find_package( LAPACK REQUIRED )
if (LAPACK_FOUND)
  message( STATUS "Found LAPACK libraries ${LAPACK_LIBRARIES}" )
else ()
  message( STATUS "LAPACK not found. Set LAPACK_LIBRARIES in command line with -D option or with GUI." )
  find_package( LAPACK REQUIRED ) # this is guaranteed to fail with more info for the user
endif ()

list(APPEND CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

macro( plasma_found_openblas )
  add_definitions( -DPLASMA_WITH_OPENBLAS )
  set( PLASMA_WITH_OPENBLAS 1 )
endmacro( plasma_found_openblas )

find_package( OpenBLAS CONFIG ) # OpenBLAS 0.3.* provide CMake configuration file
if (OpenBLAS_FOUND)
  message( STATUS "Found OpenBLAS ${OpenBLAS_VERSION}")
  find_path(OpenBLAS_INCLUDE_DIRS openblas_config.h PATHS ${OpenBLAS_DIR} ENV OPENBLAS_ROOT PATH_SUFFIXES include DOC "Path to OpenBLAS include directory")
  find_library(OpenBLAS_LIBRARIES openblas PATHS ${OpenBLAS_DIR} ENV OpenBLAS_DIR)

  # make sure OpenBLAS headers are visible to CMake tests, compiler, and linker
  include_directories(${OpenBLAS_INCLUDE_DIRS})
  list(APPEND CMAKE_REQUIRED_INCLUDES ${OpenBLAS_INCLUDE_DIRS})
  list(APPEND CMAKE_REQUIRED_LIBRARIES ${OpenBLAS_LIBRARIES})

  set(CBLAS_PROVIDER openblas)
  set(LAPACKE_PROVIDER openblas)
  plasma_found_openblas()
else ()
  message( STATUS "OpenBLAS 0.3.* not found. Set OpenBLAS_DIR environment variable." )
  message( STATUS "Either file OpenBLASConfig.cmake or openblas-config.cmake, should be inside OpenBLAS_DIR directory tree." )
endif ()

message( STATUS "BLAS(${BLAS_FOUND})" )  # linker flags `${BLA_LINKER_FLAGS}'" )
message( STATUS "LAPACK(${LAPACK_FOUND})" )  # libraries: ${LAPACK_LIBRARIES}" )

# finds new CBLAS/LAPACKE using paths and libraries for old BLAS/LAPACK
macro(FindNewWithOld m_val m_old m_new m_inc m_lib m_dir m_root)
  foreach( lib ${m_val} )
    string(REPLACE "lib${m_old}" "lib${m_new}" librpl ${lib})  # Netlib has simply naming: lib$m_old and lib$m_new
    if (EXISTS ${librpl})
      message( STATUS "Found ${m_new} ${librpl}" )
      list( INSERT ${m_lib} 0 "${librpl}" )  # add newly found library to $m_lib unconditionally at position 0
      file( TO_CMAKE_PATH "${librpl}" libslh )  # turn back-slashes to slashes, even if not needed
      string( FIND "${libslh}" "/" idx REVERSE )  # find last slash
      if (${idx} GREATER 0)
        string( SUBSTRING "${libslh}" 0 ${idx} dir)  # extract directory name
        string( APPEND dir "/.." )  #  remove library directory: /path/to/blas/lib -> /path/to/blas/lib/..
        find_path( m_header_dir "${m_new}.h" PATHS ${m_inc} ${dir} ENV ${m_dir} ${m_root} PATH_SUFFIXES include DOC "Path to ${m_new} include directory" )
        if (m_header_dir)
          list( INSERT ${m_inc} 0 "${m_header_dir}" )  # insert newly found header location
          message( STATUS "Found ${m_new} header in ${m_header_dir}" )
        endif ()
      endif ()
    endif ()
  endforeach ()
endmacro ()

if (BLAS_FOUND)
  # BLA_VENDOR is the input variable to allow only checks for specific vendor BLAS
  #if ( BLA_VENDOR STREQUAL "Intel" OR BLA_VENDOR STREQUAL "Intel10_32" OR BLA_VENDOR STREQUAL "Intel10_64lp" OR BLA_VENDOR STREQUAL "Intel10_64p_seq" )
  if ( BLAS_LIBRARIES MATCHES "mkl_core" )

    message( STATUS "Found Intel MKL" )
    add_definitions( -DPLASMA_WITH_MKL ) # this is command line only
    set( PLASMA_HAVE_MKL 1 )

  elseif ( BLAS_LIBRARIES MATCHES "essl" )
    message( STATUS "Found IBM ESSL" )

    set( PLASMA_HAVE_ESSL 1 )
    set( PLASMA_WITH_ESSL 1 )

  elseif ( BLAS_LIBRARIES MATCHES "openblas" )
    message( STATUS "Found OpenBLAS" )
    plasma_found_openblas()

  elseif ( BLAS_LIBRARIES MATCHES "Accelerate" )
    message( STATUS "Found Apple Accelerate Framework" )
    add_definitions( -DPLASMA_WITH_ACCELERATE )
    set( PLASMA_WITH_ACCELERATE 1 )

  else ()
    message( STATUS "Found Generic BLAS" )
    message( STATUS "Vendor `${BLA_VENDOR}', linker flags `${BLA_LINKER_FLAGS}', libs ${BLAS_LIBRARIES}" )
    FindNewWithOld( "${BLAS_LIBRARIES}" blas cblas CBLAS_INCLUDE_DIRS CBLAS_LIBRARIES CBLAS_DIR CBLAS_ROOT)

    # save extra includes and set to generic CBLAS header
    set(PLASMA_CMAKE_EXTRA_INCLUDE_FILES "$CMAKE_EXTRA_INCLUDE_FILES")
    set(CMAKE_EXTRA_INCLUDE_FILES "cblas.h")
    set( PLASMA_REQUIRED_INCLUDES "${CMAKE_REQUIRED_INCLUDES}" )  # save current value
    set( CMAKE_REQUIRED_INCLUDES "${CBLAS_INCLUDE_DIRS}" )

    # check if CBLAS header defines types or enums
    include(CheckTypeSize)
    check_type_size( CBLAS_TRANSPOSE PLASMA_CBLAS_TRANSPOSE )

    if ( NOT HAVE_PLASMA_CBLAS_TRANSPOSE )
      check_type_size( "enum CBLAS_TRANSPOSE" PLASMA_ENUM_CBLAS_TRANSPOSE )
      if ( HAVE_PLASMA_ENUM_CBLAS_TRANSPOSE )
        set( PLASMA_CBLAS_ADD_TYPEDEF 1 )
      else ()
        message( FATAL_ERROR "CBLAS_TRANSPOSE missing as defined type or enumeration type" )
      endif ()
    endif ()
    set(CMAKE_EXTRA_INCLUDE_FILES "$PLASMA_CMAKE_EXTRA_INCLUDE_FILES")
    set( CMAKE_REQUIRED_INCLUDES "${PLASMA_REQUIRED_INCLUDES}" )  # restore old value
  endif ()

endif (BLAS_FOUND)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if (CBLAS_INCLUDE_DIRS)  # save non-empty value found above
  set( PLASMA_CBLAS_INCLUDE_DIRS "${CBLAS_INCLUDE_DIRS}" )
endif ()

if (CBLAS_LIBRARIES)  # save non-empty value found above
  set( PLASMA_CBLAS_LIBRARIES "${CBLAS_LIBRARIES}" )
endif ()

find_package( CBLAS REQUIRED )

if (NOT CBLAS_INCLUDE_DIRS)  # if the above package find failed to come up with anything
  set( CBLAS_INCLUDE_DIRS "${PLASMA_CBLAS_INCLUDE_DIRS}" )  # restore old value
endif ()

if (NOT CBLAS_LIBRARIES)  # if the above package find failed to come up with anything
  set( CBLAS_LIBRARIES "${PLASMA_CBLAS_LIBRARIES}" )  # restore old value
endif ()
include_directories(${CBLAS_INCLUDE_DIRS})

if (CBLAS_PROVIDER STREQUAL "mkl")
  add_definitions(-DPLASMA_WITH_MKL)
  set(PLASMA_WITH_MKL TRUE)
elseif (CBLAS_PROVIDER STREQUAL "netlib")
  add_definitions(-DPLASMA_WITH_NETLIB)
  set(PLASMA_WITH_NETLIB TRUE)
endif()

FindNewWithOld( "${LAPACK_LIBRARIES}" lapack lapacke LAPACKE_INCLUDE_DIRS LAPACKE_LIBRARIES LAPACKE_DIR LAPACKE_ROOT )

if (LAPACKE_INCLUDE_DIRS)  # save non-empty value found above
  set( PLASMA_LAPACKE_INCLUDE_DIRS "${LAPACKE_INCLUDE_DIRS}" )
endif ()

if (LAPACKE_LIBRARIES)  # save non-empty value found above
  set( PLASMA_LAPACKE_LIBRARIES "${LAPACKE_LIBRARIES}" )
endif ()

find_package( LAPACKE REQUIRED )

if (NOT LAPACKE_INCLUDE_DIRS)  # if the above package find failed to come up with anything
  set( LAPACKE_INCLUDE_DIRS "${PLASMA_LAPACKE_INCLUDE_DIRS}" )  # restore old value
endif ()

if (NOT LAPACKE_LIBRARIES)  # if the above package find failed to come up with anything
  set( LAPACKE_LIBRARIES "${PLASMA_LAPACKE_LIBRARIES}" )  # restore old value
endif ()

include_directories(${LAPACKE_INCLUDE_DIRS})

if (${CMAKE_VERSION} VERSION_GREATER 3.11.99)
  cmake_policy(POP)
endif()

set(PLASMA_LINALG_LIBRARIES ${LAPACKE_LIBRARIES} ${LAPACK_LIBRARIES} ${CBLAS_LIBRARIES} ${BLAS_LIBRARIES})

if (PLASMA_DETECT_LUA)
  find_package( Lua )

  if ( LUA_FOUND )
    include_directories( ${LUA_INCLUDE_DIR} )
    add_definitions( -DPLASMA_USE_LUA ) # this is command line only
    set( PLASMA_USE_LUA 1 ) # this will be substituted in the config file
  endif()
endif()

if (PLASMA_DETECT_MAGMA)
  find_package( CUDA ) # MAGMA requires CUDA
  if (CUDA_FOUND)
    include_directories( ${CUDA_INCLUDE_DIRS} )

    message( STATUS "Looking for MAGMA" )
    find_package( MAGMA )

    if ( MAGMA_FOUND )
      include_directories( ${MAGMA_INCLUDE_DIRS} )
      add_definitions( -DPLASMA_USE_MAGMA ) # this is command line only
      set( PLASMA_USE_MAGMA 1 ) # this will be substituted in the config file
    else()
      message( STATUS "MAGMA not found" )
    endif()

  endif()
endif()

# use standard module to find OpenMP
find_package( OpenMP )
if (OPENMP_FOUND)
  if (OpenMP_C_VERSION  VERSION_LESS 4.5)
    message(FATAL_ERROR "OpenMP C version ${OpenMP_C_VERSION} is too old. Version 4.5 or later is required.")
  endif()
  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  # set C++ flags in case C++ compiler is used to compiler PLASMA
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
  message(FATAL_ERROR "OpenMP not found.")
endif()

#-------------------------------------------------------------------------------
# Parses a list of template source files to find what files should be generate.
#
# @param[in,out] src
#   On input, list of template files (source and headers) for codegen to
#   process. May have non-template source files; codegen ignores them.
#   On output, the list of generated files is appended.
#
# Example:
#   set( src zgemm.c plasma_z.h )
#   generate_files( src )
#   # On output, src is zgemm.c plasma_z.h sgemm.c dgemm.c cgemm.c plasma_s.h
#   #                   plasma_d.h plasma_c.h
#   add_library( plasma ${src} )
#
function( generate_files src )
    message( DEBUG "----- generate_files -----" )
    message( DEBUG "src   is ${src}       = <${${src}}>" )
    message( DEBUG "cache is ${src}_cache = <${${src}_cache}>" )

    if (NOT "${${src}}" STREQUAL "${${src}_cache}")
        message( STATUS "Running codegen to find files to generate for ${src}" )
        execute_process(
            COMMAND "${Python_EXECUTABLE}" "tools/codegen.py" "--depend" ${${src}}
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            RESULT_VARIABLE error
            OUTPUT_VARIABLE ${src}_depends )
        message( DEBUG "codegen error ${error}" )
        message( DEBUG "depends is ${src}_depends = <<<\n${${src}_depends}>>>" )

        if (error)
            message( STATUS "codegen returned error; cannot generate source files." )
        else()
            # Cache src so we don't have to re-run codegen to get the
            # list of dependencies again if src doesn't change.
            set( ${src}_cache ${${src}} CACHE INTERNAL "" )

            # Split lines and cache it.
            string( REGEX REPLACE "\n" ";" ${src}_depends "${${src}_depends}" )
            set( ${src}_depends ${${src}_depends} CACHE INTERNAL "" )
            message( DEBUG "depends is ${src}_depends = <<<${${src}_depends}>>>" )
        endif()
    endif()

    message( STATUS "Adding codegen commands to generate files for ${src}" )
    foreach( depend ${${src}_depends} )
        message( DEBUG "depend = <${depend}>" )
        string( REGEX MATCH "^(.*): (.*)$" out "${depend}" )
        set( outputs ${CMAKE_MATCH_1} )
        set( input   ${CMAKE_MATCH_2} )
        string( REGEX REPLACE " " ";" outputs "${outputs}" )
        list( TRANSFORM outputs PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/"
              OUTPUT_VARIABLE src_outputs )
        message( DEBUG "    input:       <${input}>" )
        message( DEBUG "    outputs:     <${outputs}>" )
        message( DEBUG "    src_outputs: <${src_outputs}>" )
        add_custom_command(
            OUTPUT   ${src_outputs}
            COMMAND "${Python_EXECUTABLE}" "tools/codegen.py" "${input}"
            DEPENDS "${input}"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            VERBATIM ${CODEGEN} )

        list( APPEND ${src} "${outputs}" )
        message( DEBUG "    src:  <${${src}}>" )
        message( DEBUG "" )
    endforeach()
    set( ${src} ${${src}} PARENT_SCOPE ) # propagate changes
    message( DEBUG "src is ${src} = <${${src}}>" )
endfunction()

#-------------------------------------------------------------------------------
# List all template files (sources and headers) and non-template source
# files, e.g., zgemm.c, plasma_z.h, test.c.
# Do not list generated files, e.g., sgemm.c, plasma_s.h.
# Please add files in alphabetical order.
set( plasma_src
    compute/clag2z.c
    compute/dlaebz2.c
    compute/dlaneg2.c
    compute/dstevx2.c
    compute/dzamax.c
    compute/pclag2z.c
    compute/pdzamax.c
    compute/pzdesc2ge.c
    compute/pzdesc2pb.c
    compute/pzdesc2tr.c
    compute/pzgb2desc.c
    compute/pzgbbrd_static.c
    compute/pzgbtrf.c
    compute/pzge2desc.c
    compute/pzge2gb.c
    compute/pzgeadd.c
    compute/pzgecpy_tile2lapack_band.c
    compute/pzgelqf.c
    compute/pzgelqf_tree.c
    compute/pzgemm.c
    compute/pzgeqrf.c
    compute/pzgeqrf_tree.c
    compute/pzgeswp.c
    compute/pzgetrf.c
    compute/pzgetri_aux.c
    compute/pzhbtrd_static.c
    compute/pzhe2hb.c
    compute/pzheb2trd_static.c
    compute/pzhecpy_tile2lapack_band.c
    compute/pzhemm.c
    compute/pzher2k.c
    compute/pzherk.c
    compute/pzhetrf_aasen.c
    compute/pzlacpy.c
    compute/pzlag2c.c
    compute/pzlangb.c
    compute/pzlange.c
    compute/pzlanhe.c
    compute/pzlansy.c
    compute/pzlantr.c
    compute/pzlarft_blgtrd.c
    compute/pzlascl.c
    compute/pzlaset.c
    compute/pzlauum.c
    compute/pzpb2desc.c
    compute/pzpbtrf.c
    compute/pzpotrf.c
    compute/pzsymm.c
    compute/pzsyr2k.c
    compute/pzsyrk.c
    compute/pztbsm.c
    compute/pztr2desc.c
    compute/pztradd.c
    compute/pztrmm.c
    compute/pztrsm.c
    compute/pztrtri.c
    compute/pzunglq.c
    compute/pzunglq_tree.c
    compute/pzungqr.c
    compute/pzungqr_tree.c
    compute/pzunmlq.c
    compute/pzunmlq_tree.c
    compute/pzunmqr.c
    compute/pzunmqr_blgtrd.c
    compute/pzunmqr_tree.c
    compute/zcgbsv.c
    compute/zcgesv.c
    compute/zcposv.c
    compute/zdesc2ge.c
    compute/zdesc2pb.c
    compute/zdesc2tr.c
    compute/zgb2desc.c
    compute/zgbmm.c
    compute/zgbset.c
    compute/zgbsv.c
    compute/zgbtrf.c
    compute/zgbtrs.c
    compute/zge2desc.c
    compute/zgeadd.c
    compute/zgeinv.c
    compute/zgelqf.c
    compute/zgelqs.c
    compute/zgels.c
    compute/zgemm.c
    compute/zgeqrf.c
    compute/zgeqrs.c
    compute/zgesdd.c
    compute/zgesv.c
    compute/zgeswp.c
    compute/zgetrf.c
    compute/zgetri.c
    compute/zgetri_aux.c
    compute/zgetrs.c
    compute/zheevd.c
    compute/zhemm.c
    compute/zher2k.c
    compute/zherk.c
    compute/zhesv.c
    compute/zhetrf.c
    compute/zhetrs.c
    compute/zlacpy.c
    compute/zlag2c.c
    compute/zlangb.c
    compute/zlange.c
    compute/zlanhe.c
    compute/zlansy.c
    compute/zlantr.c
    compute/zlascl.c
    compute/zlaset.c
    compute/zlauum.c
    compute/zpb2desc.c
    compute/zpbsv.c
    compute/zpbtrf.c
    compute/zpbtrs.c
    compute/zpoinv.c
    compute/zposv.c
    compute/zpotrf.c
    compute/zpotri.c
    compute/zpotrs.c
    compute/zsymm.c
    compute/zsyr2k.c
    compute/zsyrk.c
    compute/ztr2desc.c
    compute/ztradd.c
    compute/ztrmm.c
    compute/ztrsm.c
    compute/ztrtri.c
    compute/zunglq.c
    compute/zungqr.c
    compute/zunmlq.c
    compute/zunmqr.c

    control/constants.c
    control/context.c
    control/descriptor.c
    control/tree.c
    control/tuning.c
    control/version.c
    control/workspace.c

    include/core_lapack_z.h
    include/plasma.h
    include/plasma_internal_z.h
    include/plasma_internal_zc.h
    include/plasma_z.h
    include/plasma_zc.h
    include/plasma_zlaebz2_work.h
)

generate_files( plasma_src )
add_library( plasma SHARED ${plasma_src} )

# CMake knows about "plasma" library at this point so inform CMake where the headers are
target_include_directories(plasma PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
)

#-------------------------------------------------------------------------------
# See note above on plasma_src.
# Please add files in alphabetical order.
set( plasma_core_blas_src
    control/async.c
    control/barrier.c

    core_blas/core_clag2z.c
    core_blas/core_dcabs1.c
    core_blas/core_dzamax.c
    core_blas/core_zgbtype1cb.c
    core_blas/core_zgbtype2cb.c
    core_blas/core_zgbtype3cb.c
    core_blas/core_zgeadd.c
    core_blas/core_zgelqt.c
    core_blas/core_zgemm.c
    core_blas/core_zgeqrt.c
    core_blas/core_zgessq.c
    core_blas/core_zgeswp.c
    core_blas/core_zgetrf.c
    core_blas/core_zhbtype1cb.c
    core_blas/core_zhbtype2cb.c
    core_blas/core_zhbtype3cb.c
    core_blas/core_zhegst.c
    core_blas/core_zhemm.c
    core_blas/core_zher2k.c
    core_blas/core_zherfb.c
    core_blas/core_zherk.c
    core_blas/core_zhessq.c
    core_blas/core_zheswp.c
    core_blas/core_zlacpy.c
    core_blas/core_zlacpy_band.c
    core_blas/core_zlag2c.c
    core_blas/core_zlange.c
    core_blas/core_zlanhe.c
    core_blas/core_zlansy.c
    core_blas/core_zlantr.c
    core_blas/core_zlarfb_gemm.c
    core_blas/core_zlarfy.c
    core_blas/core_zlascl.c
    core_blas/core_zlaset.c
    core_blas/core_zlauum.c
    core_blas/core_zpamm.c
    core_blas/core_zparfb.c
    core_blas/core_zpemv.c
    core_blas/core_zpotrf.c
    core_blas/core_zsymm.c
    core_blas/core_zsyr2k.c
    core_blas/core_zsyrk.c
    core_blas/core_zsyssq.c
    core_blas/core_ztradd.c
    core_blas/core_ztrmm.c
    core_blas/core_ztrsm.c
    core_blas/core_ztrssq.c
    core_blas/core_ztrtri.c
    core_blas/core_ztslqt.c
    core_blas/core_ztsmlq.c
    core_blas/core_ztsmlq_corner.c
    core_blas/core_ztsmlq_hetra1.c
    core_blas/core_ztsmqr.c
    core_blas/core_ztsmqr_corner.c
    core_blas/core_ztsmqr_hetra1.c
    core_blas/core_ztsqrt.c
    core_blas/core_zttlqt.c
    core_blas/core_zttmlq.c
    core_blas/core_zttmqr.c
    core_blas/core_zttqrt.c
    core_blas/core_zunmlq.c
    core_blas/core_zunmqr.c

    include/core_lapack_z.h
    include/plasma_core_blas.h
    include/plasma_core_blas_z.h
    include/plasma_core_blas_zc.h
    include/plasma.h
    include/plasma_internal_z.h
    include/plasma_internal_zc.h
    include/plasma_z.h
    include/plasma_zc.h
)

generate_files( plasma_core_blas_src )
add_library( plasma_core_blas SHARED ${plasma_core_blas_src} )

target_include_directories(plasma_core_blas PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
)

#-------------------------------------------------------------------------------
# See note above on plasma_src.
# Please add files in alphabetical order.
set( plasma_test_src
    include/plasma.h

    test/test.c
    test/test.h
    test/test_clag2z.c
    test/test_dstevx2.c
    test/test_dzamax.c
    test/test_z.h
    test/test_zc.h
    test/test_zcgbsv.c
    test/test_zcgesv.c
    test/test_zcposv.c
    test/test_zgbmm.c
    test/test_zgbsv.c
    test/test_zgbtrf.c
    test/test_zgeadd.c
    test/test_zgeinv.c
    test/test_zgelqf.c
    test/test_zgelqs.c
    test/test_zgels.c
    test/test_zgemm.c
    test/test_zgeqrf.c
    test/test_zgeqrs.c
    test/test_zgesdd.c
    test/test_zgesv.c
    test/test_zgeswp.c
    test/test_zgetrf.c
    test/test_zgetri.c
    test/test_zgetri_aux.c
    test/test_zgetrs.c
    test/test_zheevd.c
    test/test_zhemm.c
    test/test_zher2k.c
    test/test_zherk.c
    test/test_zhesv.c
    test/test_zhetrf.c
    test/test_zlacpy.c
    test/test_zlag2c.c
    test/test_zlangb.c
    test/test_zlange.c
    test/test_zlanhe.c
    test/test_zlansy.c
    test/test_zlantr.c
    test/test_zlascl.c
    test/test_zlaset.c
    test/test_zlauum.c
    test/test_zpbsv.c
    test/test_zpbtrf.c
    test/test_zpoinv.c
    test/test_zposv.c
    test/test_zpotrf.c
    test/test_zpotri.c
    test/test_zpotrs.c
    test/test_zsymm.c
    test/test_zsyr2k.c
    test/test_zsyrk.c
    test/test_ztradd.c
    test/test_ztrmm.c
    test/test_ztrsm.c
    test/test_ztrtri.c
    test/test_zunmlq.c
    test/test_zunmqr.c
)

generate_files( plasma_test_src )
add_executable( plasmatest ${plasma_test_src} )

#-------------------------------------------------------------------------------
find_library(MATH_LIBRARY m)
if( MATH_LIBRARY )
  # OpenBLAS needs to link C math library (usually -lm) but MKL doesn't
  set(PLASMA_LIBRARIES ${PLASMA_LINALG_LIBRARIES} ${LUA_LIBRARIES} ${MATH_LIBRARY})
else( MATH_LIBRARY )
  set(PLASMA_LIBRARIES ${PLASMA_LINALG_LIBRARIES} ${LUA_LIBRARIES})
endif( MATH_LIBRARY )

target_link_libraries( plasmatest plasma plasma_core_blas ${PLASMA_LIBRARIES} )
if ( MAGMA_FOUND )
    target_link_libraries( plasma plasma_core_blas ${PLASMA_LIBRARIES} ${MAGMA_LIBRARIES} ${CUDA_LIBRARIES} )
else()
  target_link_libraries( plasma plasma_core_blas ${PLASMA_LIBRARIES} )
endif()
target_link_libraries( plasma_core_blas ${PLASMA_LIBRARIES} )

target_include_directories(plasmatest PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
)

set_target_properties( plasma_core_blas plasma PROPERTIES VERSION ${CMAKE_PROJECT_VERSION} SOVERSION 1.0)

configure_file( include/plasma_config.hin ${CMAKE_CURRENT_SOURCE_DIR}/include/plasma_config.h @ONLY NEWLINE_STYLE LF )

include(GNUInstallDirs)

install(TARGETS plasma plasma_core_blas EXPORT plasmaTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
file( GLOB plasma_headers include/plasma*.h)
install(FILES ${plasma_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS plasmatest EXPORT plasmaTargets RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# generage config files for upstream CMake projects
include(CMakePackageConfigHelpers)

write_basic_package_version_file( "${CMAKE_CURRENT_BINARY_DIR}/plasmaExport/plasmaConfigVersion.cmake" VERSION ${PLASMA_VERSION} COMPATIBILITY AnyNewerVersion )

export(EXPORT plasmaTargets FILE "${CMAKE_CURRENT_BINARY_DIR}/plasmaExport/plasmaTargets.cmake" NAMESPACE plasma::)

configure_file( ${PROJECT_SOURCE_DIR}/share/cmake/plasma.cmakein "${CMAKE_CURRENT_BINARY_DIR}/plasmaExport/plasmaConfig.cmake" @ONLY )
install( EXPORT plasmaTargets FILE plasmaTargets.cmake NAMESPACE plasma:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/plasma NAMESPACE plasma:: CONFIGURATIONS Debug Release RelWithDebInfo MinSizeRel )

install( FILES "${CMAKE_CURRENT_BINARY_DIR}/plasmaExport/plasmaConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/plasmaExport/plasmaConfigVersion.cmake" DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/plasma )

# generate pkg-config .pc file from a template
string(TOLOWER ${PROJECT_NAME} lproj)
string(REPLACE ";" "\ " plasma_libs_spaced "${PLASMA_LIBRARIES}")
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/share/pkgconfig/${lproj}.pcin ${CMAKE_CURRENT_BINARY_DIR}/${lproj}.pc @ONLY NEWLINE_STYLE LF )
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${lproj}.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

# Copy run_tests script to build directory.
add_custom_command(
    TARGET plasmatest POST_BUILD
    COMMAND
        cp ${CMAKE_CURRENT_SOURCE_DIR}/test/run_tests.py
           ${CMAKE_CURRENT_BINARY_DIR}/run_tests.py
)
