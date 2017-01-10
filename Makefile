# ----------------------------------------
# Usage:
#   make [all]      --  make lib test
#   make lib        --  make lib/libplasma.a lib/libcoreblas.a
#   make shared     --  make lib/libplasma.so lib/libcoreblas.so
#   make test       --  make test/test
#   make docs       --  make docs/html
#   make generate   --  generate precisions
#   make clean      --  remove objects, libraries, and executables
#   make cleangen   --  remove generated precision files
#   make distclean  --  remove above, Makefile.*.gen, and anything else that can be generated


# ----------------------------------------
# Define default rule first, before including Makefiles

all: lib test


# ----------------------------------------
# Tools and flags
# Get from make.inc, or use these defaults

include make.inc

CC        ?= cc

ARCH      ?= ar
ARCHFLAGS ?= cr
RANLIB    ?= ranlib

CFLAGS    ?= -std=c99 -fopenmp -O3 -Wall -Wno-unused-variable -Wno-unused-function
LDFLAGS   ?= -fopenmp

# INC and LIBS indicate where to find LAPACK, and LAPACKE, and CBLAS
INC       ?= -I$(LAPACKDIR)/LAPACKE/include -I$(CBLASDIR)/include
LIBS      ?= -L$(LAPACKDIR) -llapack -llapacke -L$(CBLASDIR)/lib -lcblas -lblas

prefix    ?= /usr/local/plasma


# ----------------------------------------
# Internal tools and flags

codegen     := ./tools/codegen.py

PLASMA_INC  := -Iinclude
PLASMA_LIBS := -Llib -lplasma -lcoreblas

.DELETE_ON_ERROR:

.SUFFIXES:


# ----------------------------------------
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


# ----------------------------------------
# Build libraries

.PHONY: lib

libs := \
	lib/libplasma.a    \
	lib/libcoreblas.a  \

lib: $(libs)

# In case changing Makefile.gen changes $(obj), also depend on it,
# which recreates the library if a file is removed.
# ----------
lib/libplasma.a: $(plasma_obj) Makefile.plasma.gen
	-rm -f $@
	$(ARCH) $(ARCHFLAGS) $@ $(plasma_obj)
	$(RANLIB) $@

# ----------
lib/libcoreblas.a: $(coreblas_obj) Makefile.coreblas.gen
	-rm -f $@
	$(ARCH) $(ARCHFLAGS) $@ $(coreblas_obj)
	$(RANLIB) $@


# ----------------------------------------
# Build shared libraries

# check whether all FLAGS have -fPIC
have_fpic = $(and $(findstring -fPIC, $(CFLAGS)),   \
                  $(findstring -fPIC, $(LDFLAGS)))

# -----
# if all FLAGS have -fPIC: allow compiling shared libraries
ifneq ($(have_fpic),)

shared := \
	lib/libplasma.so    \
	lib/libcoreblas.so  \

shared: $(shared)

lib/libplasma.so: $(plasma_obj) Makefile.plasma.gen lib/libcoreblas.so
	$(CC) -shared $(LDFLAGS) -o $@ $(plasma_obj) -Llib -lcoreblas $(LIBS)

lib/libcoreblas.so: $(coreblas_obj) Makefile.coreblas.gen
	$(CC) -shared $(LDFLAGS) -o $@ $(coreblas_obj) $(LIBS)

# -----
# else: some FLAGS missing -fPIC: print error for shared
else

shared:
	@echo "Error: 'make shared' requires CFLAGS and LDFLAGS to have -fPIC."
	@echo "Please edit your make.inc file, make clean && make shared."

endif


# ----------------------------------------
# Build tester

.PHONY: test

test: $(test_exe)

$(test_exe): $(test_obj) $(libs) Makefile.test.gen
	$(CC) $(LDFLAGS) -o $@ $(test_obj) $(PLASMA_LIBS) $(LIBS)


# ----------------------------------------
# Build objects
# Headers must exist before compiling, but use order-only prerequisite (after "|")
# so as not to force recompiling everything if a header changes.
# (Should use compiler's -MMD flag to create header dependencies.)

%.o: %.c | $(headers)
	$(CC) $(CFLAGS) $(PLASMA_INC) $(INC) -c -o $@ $<

%.i: %.c | $(headers)
	$(CC) $(CFLAGS) $(PLASMA_INC) $(INC) -E -o $@ $<


# ----------------------------------------
# Build documentation

.PHONY: docs

docs: generate
	doxygen docs/doxygen/doxyfile.conf


# ----------------------------------------
# Install

.PHONY: install_dirs install

install_dirs:
	mkdir -p $(prefix)
	mkdir -p $(prefix)/include
	mkdir -p $(prefix)/lib
	mkdir -p $(prefix)/lib/pkgconfig

install: lib install_dirs
	cp  include/*.h $(prefix)/include
	cp  $(libs)     $(prefix)/lib
	-cp $(shared)   $(prefix)/lib
	# pkgconfig
	cat lib/pkgconfig/plasma.pc.in         | \
	sed -e s:@INSTALL_PREFIX@:"$(prefix)": | \
	sed -e s:@CFLAGS@:"$(INC)":            | \
	sed -e s:@LIBS@:"$(LIBS)":             | \
	sed -e s:@REQUIRES@::                    \
	    > $(prefix)/lib/pkgconfig/plasma.pc


# ----------------------------------------
# Maintenance rules
# makefiles_gen define generate and cleangen.

.PHONY: clean distclean

clean:
	-rm -f $(plasma_obj) $(coreblas_obj) $(test_obj) $(test_exe) $(libs) $(shared)

# cleangen removes generated files if the template still exists;
# grep for any stale generated files without a template.
distclean: clean cleangen
	grep -l @generated $(plasma_src) $(coreblas_src) $(test_src) | xargs rm
	-rm -f compute/*.o control/*.o core_blas/*.o test/*.o
	-rm -f $(makefiles_gen)
	-rm -rf docs/html


# ----------------------------------------
# Create dependencies to do precision generation.

plasma_src   := $(wildcard compute/*.c compute/*.h control/*.c control/*.h include/*.h)

coreblas_src := $(wildcard core_blas/*.c core_blas/*.h)

test_src     := $(wildcard test/*.c test/*.h)

Makefile.plasma.gen: $(codegen)
	$(codegen) --make --prefix plasma   $(plasma_src)   > $@

Makefile.coreblas.gen: $(codegen)
	$(codegen) --make --prefix coreblas $(coreblas_src) > $@

Makefile.test.gen: $(codegen)
	$(codegen) --make --prefix test     $(test_src)     > $@


# ----------
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
# ----------

force_gen: ;


# ----------------------------------------------------------------------
# Debugging

echo:
	@echo "CC      $(CC)"
	@echo "CFLAGS  $(CFLAGS)"
	@echo "LDFLAGS $(LDFLAGS)"
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
