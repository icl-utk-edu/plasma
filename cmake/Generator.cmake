#.rst:
# Generator
# ---------
#
# Calls Python `codegen.py` script to generate source code for different
# data types.

#-------------------------------------------------------------------------------
# Parses a list of template source files to find what files should be generated.
#
# @param[in,out] src
#   On input, variable that is a list of template files (source and
#   headers) for codegen to process. May have non-template source files;
#   codegen ignores them.
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
