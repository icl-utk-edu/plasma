# ------------------------------------------------------------------------------
# Usage:
#   make [all]      --  make lib test
#   make lib        --  make lib/libplasma.{a,so} lib/libcoreblas.{a,so}
#   make lua        --  make tools/lua-5.3.4/src/liblua.a
#   make test       --  make test/test
#   make docs       --  make docs/html
#   make generate   --  generate precisions
#   make clean      --  remove objects, libraries, and executables
#   make cleangen   --  remove generated precision files
#   make distclean  --  remove above, Makefile.*.gen, and anything else that can be generated
#
# Options:
#   quiet=1         --  abbreviate long compilation commands
#   static=0        --  build only shared libraries (no static)
#   static=1        --  build only static libraries (no shared)
#                       By default, builds both static and shared libraries.


# ------------------------------------------------------------------------------
# Define default rule first, before including Makefiles

all: lib test


# ------------------------------------------------------------------------------
# Tools and flags
# Get from make.inc, or use these defaults

include make.inc

# define FPIC to make shared libraries
#FPIC      = -fPIC

CC        ?= cc
CFLAGS    ?= -std=c99 -fopenmp $(FPIC) -O3 -Wall -Wno-unused-variable -Wno-unused-function
LDFLAGS   ?= -fopenmp $(FPIC)

ARCH      ?= ar
ARCHFLAGS ?= cr
RANLIB    ?= ranlib

# INC and LIBS indicate where to find LAPACK, LAPACKE, and CBLAS
INC       ?= -I$(LAPACKDIR)/LAPACKE/include -I$(CBLASDIR)/include
LIBS      ?= -L$(LAPACKDIR) -llapack -llapacke -L$(CBLASDIR)/lib -lcblas -lblas

prefix    ?= /usr/local/plasma

# one of: aix bsd c89 freebsd generic linux macosx mingw posix solaris
# usually generic is fine
lua_platform ?= generic


# ------------------------------------------------------------------------------
# Internal tools and flags

codegen     := ./tools/codegen.py

lua_version := lua-5.3.4
lua_dir     := tools/$(lua_version)/src
lua_lib     := lib/liblua.a

PLASMA_INC  := -Iinclude -I$(lua_dir)
PLASMA_LIBS := -Llib -lplasma -lcoreblas -llua

.DELETE_ON_ERROR:

.SUFFIXES:

ifeq ($(quiet),1)
   quiet_CC := @echo "$(CC) ... $@";
   quiet_AR := @echo "$(ARCH) ... $@";
endif


# ------------------------------------------------------------------------------
# Define sources, objects, libraries, executables.
# These makefiles define lists of source and header files in
# $(plasma_all), $(coreblas_all), and $(test_all).

makefiles_gen := \
	Makefile.plasma.gen   \
	Makefile.coreblas.gen \
	Makefile.test.gen     \

-include $(makefiles_gen)

plasma_hdr   := $(filter %.h, $(plasma_all))
coreblas_hdr := $(filter %.h, $(coreblas_all))
test_hdr     := $(filter %.h, $(test_all))
headers      := $(plasma_hdr) $(coreblas_hdr) $(test_hdr)

plasma_obj   := $(addsuffix .o, $(basename $(filter-out %.h, $(plasma_all))))
coreblas_obj := $(addsuffix .o, $(basename $(filter-out %.h, $(coreblas_all))))
test_obj     := $(addsuffix .o, $(basename $(filter-out %.h, $(test_all))))

test_exe     := test/test


# ------------------------------------------------------------------------------
# Build dependencies (Lua)

lua: $(lua_lib)

$(lua_dir): tools/$(lua_version).tar.gz
	cd tools && tar -zxvf $(lua_version).tar.gz

# clear TARGET_ARCH, which Intel's compilervars.sh script (erroneously?) sets to intel64
$(lua_lib): | $(lua_dir)
	cd $(lua_dir) && make $(lua_platform) MYCFLAGS="-fPIC" CC=$(CC) TARGET_ARCH=
	cp $(lua_dir)/liblua.a lib

# plasma needs lua include files
$(plasma_obj):   | $(lua_dir)
$(coreblas_obj): | $(lua_dir)

libfiles = $(lua_lib)


# ------------------------------------------------------------------------------
# Build static libraries

ifneq ($(static),0)
    libfiles += \
	lib/libplasma.a    \
	lib/libcoreblas.a  \

    # In case changing Makefile.gen changes $(obj), also depend on it,
    # which recreates the library if a file is removed.
    lib/libplasma.a: $(plasma_obj) Makefile.plasma.gen
	-rm -f $@
	$(quiet_AR) $(ARCH) $(ARCHFLAGS) $@ $(plasma_obj)
	$(RANLIB) $@

    lib/libcoreblas.a: $(coreblas_obj) Makefile.coreblas.gen
	-rm -f $@
	$(quiet_AR) $(ARCH) $(ARCHFLAGS) $@ $(coreblas_obj)
	$(RANLIB) $@
endif


# ------------------------------------------------------------------------------
# Build shared libraries

# if all FLAGS have -fPIC, allow compiling shared libraries
have_fpic = $(and $(findstring -fPIC, $(CFLAGS)),   \
                  $(findstring -fPIC, $(LDFLAGS)))

ifneq ($(static),1)
ifneq ($(have_fpic),)
   libfiles += \
	lib/libplasma.so    \
	lib/libcoreblas.so  \

   top = $(shell pwd)

   rpath = -Wl,-rpath,$(top)/lib


   # MacOS (darwin) needs shared library's path set
   ifneq ($(findstring darwin, $(OSTYPE)), '')
       install_name = -install_name @rpath/$(notdir $@)
   endif

   lib/libplasma.so: $(plasma_obj) Makefile.plasma.gen lib/libcoreblas.so $(lua_lib)
	$(quiet_CC) $(CC) -shared $(LDFLAGS) -o $@ $(plasma_obj) \
	-Llib -lcoreblas -llua $(LIBS) \
	$(install_name)

   lib/libcoreblas.so: $(coreblas_obj) Makefile.coreblas.gen $(lua_lib)
	$(quiet_CC) $(CC) -shared $(LDFLAGS) -o $@ $(coreblas_obj) \
	-Llib -llua $(LIBS) \
	$(install_name)
endif
endif

.PHONY: lib

lib: $(libfiles)


# ------------------------------------------------------------------------------
# Build tester

.PHONY: test

test: $(test_exe)

$(test_exe): $(test_obj) $(libfiles) Makefile.test.gen
	$(quiet_CC) $(CC) $(LDFLAGS) -o $@ $(test_obj) $(PLASMA_LIBS) $(LIBS) \
	$(rpath)


# ------------------------------------------------------------------------------
# Build objects
# Headers must exist before compiling, but use order-only prerequisite
# (after "|") so as not to force recompiling everything if a header changes.
# (Should use compiler's -MMD flag to create header dependencies.)

%.o: %.c | $(headers)
	$(quiet_CC) $(CC) $(CFLAGS) $(PLASMA_INC) $(INC) -c -o $@ $<

%.i: %.c | $(headers)
	$(quiet_CC) $(CC) $(CFLAGS) $(PLASMA_INC) $(INC) -E -o $@ $<


# ------------------------------------------------------------------------------
# Build documentation

.PHONY: docs

docs: generate
	doxygen docs/doxygen/doxyfile.conf


# ------------------------------------------------------------------------------
# Install
# TODO: need to install Lua headers?

.PHONY: install_dirs install

install_dirs:
	mkdir -p $(prefix)
	mkdir -p $(prefix)/include
	mkdir -p $(prefix)/lib
	mkdir -p $(prefix)/lib/pkgconfig

install: lib install_dirs
	cp include/*.h $(prefix)/include
	cp $(libfiles) $(prefix)/lib
	# pkgconfig
	cat lib/pkgconfig/plasma.pc.in         | \
	sed -e s:@INSTALL_PREFIX@:"$(prefix)": | \
	sed -e s:@CFLAGS@:"$(INC)":            | \
	sed -e s:@LIBS@:"$(LIBS)":             | \
	sed -e s:@REQUIRES@::                    \
	    > $(prefix)/lib/pkgconfig/plasma.pc

uninstall:
	rm -f $(prefix)/include/core_blas*.h
	rm -f $(prefix)/include/core_lapack*.h
	rm -f $(prefix)/include/plasma*.h
	rm -f $(prefix)/lib/libcoreblas*
	rm -f $(prefix)/lib/libplasma*
	rm -f $(prefix)/lib/liblua*
	rm -f $(prefix)/lib/pkgconfig/plasma.pc


# ------------------------------------------------------------------------------
# Maintenance rules
# makefiles_gen define generate and cleangen.

.PHONY: clean distclean

clean:
	-rm -f $(plasma_obj) $(coreblas_obj) $(test_obj) $(test_exe) $(libfiles)
	-cd $(lua_dir) && make clean

# cleangen removes generated files if the template still exists;
# grep for any stale generated files without a template.
distclean: clean cleangen
	grep -l @generated $(plasma_src) $(coreblas_src) $(test_src) | xargs rm -f
	-rm -f compute/*.o control/*.o core_blas/*.o test/*.o
	-rm -f $(makefiles_gen)
	-rm -rf docs/html
	-rm -rf tools/$(lua_version)


# ------------------------------------------------------------------------------
# Create dependencies to do precision generation.

plasma_src   := $(wildcard compute/*.c compute/*.h include/*.h)

coreblas_src := $(wildcard core_blas/*.c core_blas/*.h control/*.c control/*.h)

test_src     := $(wildcard test/*.c test/*.h)

Makefile.plasma.gen: $(codegen)
	$(codegen) --make --prefix plasma   $(plasma_src)   > $@

Makefile.coreblas.gen: $(codegen)
	$(codegen) --make --prefix coreblas $(coreblas_src) > $@

Makefile.test.gen: $(codegen)
	$(codegen) --make --prefix test     $(test_src)     > $@


# --------------------
# If the list of src files changes, then force remaking Makefile.gen
# To reduce unnecesary remaking, don't remake if either:
# 1) src == old:
#    src has same files now as when Makefile.gen was generated, or
# 2) src - generated == templates:
#    src has all the templates from Makefile.gen, and no new non-generated files.
ifneq ($(plasma_src),$(plasma_old))
ifneq ($(filter-out $(plasma_generated),$(plasma_src)),$(plasma_templates))
Makefile.plasma.gen: force_gen
endif
endif

ifneq ($(coreblas_src),$(coreblas_old))
ifneq ($(filter-out $(coreblas_generated),$(coreblas_src)),$(coreblas_templates))
Makefile.coreblas.gen: force_gen
endif
endif

ifneq ($(test_src),$(test_old))
ifneq ($(filter-out $(test_generated),$(test_src)),$(test_templates))
Makefile.test.gen: force_gen
endif
endif
# --------------------

force_gen: ;


# ------------------------------------------------------------------------------
# Debugging

echo:
	@echo "CC      $(CC)"
	@echo "CFLAGS  $(CFLAGS)"
	@echo "LDFLAGS $(LDFLAGS)"
	@echo
	@echo "test_exe           <$(test_exe)>"
	@echo "libfiles           <$(libfiles)>"
	@echo
	@echo "plasma_src         <$(plasma_src)>"
	@echo "plasma_old         <$(plasma_old)>"
	@echo "plasma_templates   <$(plasma_templates)>"
	@echo "plasma_filtered    <$(filter-out $(plasma_generated),$(plasma))>"
	@echo "plasma_hdr         <$(plasma_hdr)>"
	@echo
	@echo "coreblas_src       <$(coreblas_src)>"
	@echo "coreblas_old       <$(coreblas_old)>"
	@echo "coreblas_templates <$(coreblas_templates)>"
	@echo "coreblas_filtered  <$(filter-out $(coreblas_generated),$(coreblas))>"
	@echo "coreblas_hdr       <$(coreblas_hdr)>"
	@echo
	@echo "test_src           <$(test_src)>"
	@echo "test_old           <$(test_old)>"
	@echo "test_templates     <$(test_templates)>"
	@echo "test_filtered      <$(filter-out $(test_generated),$(test))>"
	@echo "test_hdr           <$(test_hdr)>"
	@echo
	@echo "headers            <$(headers)>"
