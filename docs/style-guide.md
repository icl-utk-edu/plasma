[TOC]

About this guide
================

This guide is mostly based on the
[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

> Sometimes the Google rules are tweaked, sometimes contradicted.
>
> This guide includes rules beyond those in the Google guide.

Other notable sources of best software engineering practices include:

* [JSF Air Vehicle C++ Coding Standards](http://www.stroustrup.com/JSF-AV-rules.pdf),
* [Code Complete](http://cc2e.com) by Steve McConnell,
* [Source codes of the Trilinos project](https://trilinos.org),
* [Microsoft .Net Guidelines](https://msdn.microsoft.com/en-us/library/ms229042(v=vs.110).aspx),
* [Linux Kernel Coding Style](https://www.kernel.org/doc/Documentation/CodingStyle).

Some conventions introduced in this guide originate from:

* NVIDIA CUDA,
* Intel MKL,
* Microsoft.

This guide is created using Markdown.
For Markdown documentation consult:

* http://daringfireball.net/projects/markdown/,
* https://en.wikipedia.org/wiki/Markdown,

General Guidelines
==================

* Be consistent in your own code.
* Follow the conventions already established by the project.
* If you spot inconsistencies, fix them.
* Break rules if it helps readability. This is only a guide.

> This guide was written for PLASMA. Other ICL projects have their own
> conventions. A few differences with SLATE's style are noted here.

Standard Compliance
===================

C codes should be C99 compliant and compiled with the `-std=c99` flag,
and C++ codes should be C++11 compliant and compiled with the `-std=c++11` flag
(or later).

Avoid features present only in C but not in C++. That is, C code should compile with either C or C++ compiler.

> Microsoft's C compiler doesn't support C99, so code must use C++11.

Header Files
============

Self-contained Headers
----------------------

Header files should be self-contained and end in `.h` for C and in `.hh` for C++.
Files that are meant for textual inclusion, but are not headers, should end in `.inc`.

> Google uses `.cc` for C++ source files, but `.h` for C++ header files.
> However, it is useful to have a distinction between C and C++ headers.
> Therefore, we use `.c` and `.h` for C and `.cc` and `.hh` for C++.
>
> Trilinos uses another common convention of `.cpp` and `.hpp`.
> However, in a long list of files, this puts a lot of p's on the screen.
> The `.cc` and `.hh` endings are shorter and cleaner.

\#define Guards
--------------

All header files should have `#define` guards to prevent multiple inclusion.
The format of the symbol name should be `<PROJECT>_<FILE>_H` for C
and `<PROJECT>_<FILE>_HH` for C++.

```
#!cpp

#ifndef PLASMA_BLAS_H
#define PLASMA_BLAS_H

...

#endif // PLASMA_BLAS_H
```

> Google uses an underscore at the end.
> Trilinos does not use underscores.
> In the case of header guards, beginning/ending underscores seem pointless.
> Underscores in front of a name are reserved for system-level headers.

extern C
--------

C library headers should add extern "C" around function definitions, to allow C++ codes to use the library.

```
#!cpp

#ifdef __cplusplus
extern "C" {
#endif

...

#ifdef __cplusplus
}  // extern "C"
#endif
```

Forward Declarations
--------------------

Avoid using forward declarations, i.e., declarations of classes, functions,
or templates without associated definitions.
Instead, `#include` the headers you need. This limits the number of places that
have to be modified if the members change.
Check [Google Style Guide](https://google.github.io/styleguide/cppguide.html#Forward_Declarations)
to see why.

Inline Functions
----------------

Mark small functions as inline, specifically those that serve as macro replacements.
Although the compiler will identify on its own functions
suitable for inlining, it is a good idea hint it directly.
Define as inline member accessors and functions that would otherwise be macros,
e.g., address arithmetic functions and alike.
Check [Google Style Guide](https://google.github.io/styleguide/cppguide.html#Inline_Functions)
for further guidelines.

Names and Order of Included Header Files
----------------------------------------

Use the following order of inclusion:

* your project headers,
* standard headers,
* other libraries' headers.

For example, the include section might look like this:

```
#!cpp

#include "common_magma.h"
#include "batched_kernel_param.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <omp.h>
#include <cuda.h>
#include <mkl_blas.h>
```

> This ordering is a little different from the one in the
> [Google Style Guide](https://google.github.io/styleguide/cppguide.html#Names_and_Order_of_Includes).

All of project's header files should be listed as descendants of the project's
source directory without use of UNIX directory shortcuts `.` (the current directory)
or `..` (the parent directory).

You should include all the headers that define the symbols you rely upon.
If you rely on symbols from `bar.h`, do not count on the fact that you included `foo.h`
which (currently) includes `bar.h`.
Include `bar.h` explicitly.
However, any includes present in the related header do not need to be included again
in the related `.cc` (i.e., `foo.cc` can rely on `foo.h`'s includes).

Sometimes, system-specific code needs conditional includes.
Such code can put conditional includes after other includes.
Keep your system-specific code small and localized.

```
#!cpp

#ifndef __APPLE__
#include <pthread.h>
#else
#include <libkern/OSAtomic.h>

...
#endif // __APPLE__
```

Scoping
=======

Namespaces
----------

With few exceptions, place C++ code in a namespace.
Use named namespaces as follows:

* Namespaces wrap the entire source file after includes.
* Do not declare anything in namespace `std`, including forward declarations
  of standard library classes. Declaring entities in namespace `std` is undefined
  behavior (unless specifically allowed in the C++ standard).
  To declare entities from the standard library, include the appropriate
  header file.
* Do not use a `using` directive to make all names from a namespace available.
* Do not use `using` declarations in `.hh` files, because anything imported
  into a namespace in a `.hh` file becomes part of the public API exported by that file.
* Use a `using` declaration anywhere in a `.cc` file (including in the global namespace),
  and in functions, methods, and classes.
* Minimize the use of namespace aliases.
* Do not use inline namespaces.
* Use Pascal case for namespace components.

Namespaces should have unique names based on the project name.
Generic components, that may be shared among multiple projects, such as
an efficient implementation of a thread-safe hash table, may be placed in
a namespace `icl`, while project-specific components must be placed
in a namespace, e.g., `plasma`.

Format namespaces in `.hh` files as follows:

```
#!cpp

#ifndef PLASMA_HH
#define PLASMA_HH

#include <cuda.h>

namespace plasma {

...

} // namespace plasma

#endif // PLASMA_HH
```

Nonmember, Static Member, and Global Functions
----------------------------------------------

Prefer placing nonmember functions in a namespace; use completely global functions rarely.
Prefer grouping functions with a namespace instead of using a class as if it was a namespace.

Sometimes it is useful to define a function not bound to a class instance.
Such a function can be either a static member or a nonmember function.
Nonmember functions should not depend on external variables,
and should nearly always exist in a namespace.
Rather than creating classes only to group static member functions
which do not share static data, use namespaces instead.

If you must define a nonmember function and it is only needed in its `.cc` file,
use an unnamed namespace or static linkage (e.g., `static int Foo() {...}`)
to limit its scope.

Local Variables
---------------

Place local variables in the narrowest scope possible.
Declare loop counters inside the `for` and `while` statements if possible.

Initialize variables that are initialized once in the declaration.
If the variable is reused, use declaration in a separate line preceeding its first use,
e.g.:

```
#!cpp

cublasStatus_t retval;

retval = cublasCreate(...);
assert(retval == CUBLAS_STATUS_SUCCESS);

retval = cublasDestroy(...);
assert(retval == CUBLAS_STATUS_SUCCESS);
```

Static and Global Variables
---------------------------

Do not use global variables.
Use static variables only for Plain Old Data (POD): only ints, chars, floats,
or pointers, or arrays/structs of POD. Check the
[Google Style Guide](https://google.github.io/styleguide/cppguide.html#Static_and_Global_Variables)
for detailed explanations.

Classes
=======

In the context of numerical software, many advanced features of C++ can,
and should be, ignored.

* Do not define implicit conversions.

* Do not use delegating and inheriting constructors. Use helper functions instead.

* Use a `struct` only for passive objects that carry data; everything else is a class.
> Note that member variables in structs and classes have different naming rules.

* Use composition instead of inheritance.
  Encapsulate rather than derive.
  If you end up using inheritance, try to limit `protected` to functions,
  don’t use for data members.

* Overload operators judiciously.
  Define overloaded operators only if their meaning is obvious, unsurprising,
  and consistent with the corresponding built-in operators.
  Define operators only on your own types.
> Operator overloading makes sense for numerical objects, such as complex numbers
> or extended precision numbers.

* Make data members private, unless they are `static const`.

* Use private before public. Within each section, use the following order:
    - constants,
    - data members,
    - constructors,
    - destructors,
    - methods.

Copyable and Movable Types
--------------------------

In numerical libraries, we basically have two types of classes:
small classes to represent complex numbers and extended precision numbers,
and big classes to represent contexts and descriptors.
Handling of the small classes is straightforward.
They can be copyable and movable, and use overloaded operators.
Handling of the big classes requires a little more caution, due to the issue
of ownership of pointers.
The issue basically boils down to the definition of a descriptor.

Numerical libraries often introduce descriptors, which describe the layout
of the data, but do not contain the actual memory references.
Instead, memory pointers are passed alongside descriptors to library routines.
ScaLAPACK can serve as a legacy example,
NVIDIA cuDNN can serve as a contemporary example.
In this case, the descriptor itself is perfectly copyable and mutable.

However, at some level, this separation goes against object oriented programming
(defies the principle encapsulation).
PLASMA already set a precedence by including the memory pointer in the
matrix descriptor.
Decoupling memory references from descriptors becomes even more troublesome
if a single descriptor describes data requiring more than one memory pointer,
e.g., a low-rank matrix approximation represented by its SVD.

Therefore, big classes, representing large mathematical objects,
should be handled as follows:

* Be self-contained, i.e., contain all memory pointers.

* Set all memory pointers to `NULL` (`nullptr` in C++) at the time of initialization
  (in the constructor in C++).
  Rely on constructors or factory methods or `Init()` methods to perform
  allocations and initializations.
  Free memory in destructors or `Finalize()` methods and include a check
  for NULL.

* Be non-copyable and non-movable and be passed by reference.

Functions
=========

* Write short functions.
  If possible, write functions that fit on one screen.
  If a function exceeds about 50 lines, think about whether it can be broken up
  without harming the structure of the program.

* All parameters passed by reference must be labeled `const` unless they are
  modified in the function.

* Use function overloading judiciously.

* Avoid default arguments.

* Avoid trailing return type syntax.

Parameter Ordering
------------------

Place all input-only parameters before any output parameters.
In particular, do not add new parameters to the end of the function
just because they are new; place new input-only parameters before the output parameters.

Use the following ordering of parameters:

* library handle/context,
* data layout specifier,
* input parameters,
* output parameters,
* flags.

Place array size before the memory pointer.

Other C++ Features
==================

* Use friend classes within reason.
  A common use of friend is to have a `FooBuilder` class be a friend of `Foo`
  so that it can construct the inner state of `Foo` correctly, without exposing
  this state to the world.
  In some cases it may be useful to make a unit test class a friend of the class it tests.

* Avoid using Run Time Type Information.

* Use traditional, C-style, casting.
  In numerical codes virtually all casts are conversions and there is no ambiguity.
  The syntax of C++ style casts is nasty.
  Also, `(long long)x` is the only way to convert to the `long long` type, because
  of the space.
  And also, the proper name of `long` is `long int`, which also includes the space.

> SLATE uses C++ style casts, which are clearer about what expression
> the cast applies to, e.g., `(int) x * y` is the same as `int( x ) * y`,
> not `int( x * y )`. For `long long`, SLATE has a typedef `llong` so that
> `llong( x )` works. Otherwise, `(long long)( x )` works.

* Do not use C++ streams.
  Use C standard IO functions instead.
  C++ streams are cumbersome and the
  [Google Style Guide](https://google.github.io/styleguide/cppguide.html#Streams)
  provides a long list of reasons why.

* Use preincrement/predecrement (`++i`) as opposed to postincrement/postdecrement (`i++`).
  This is a common C++ convention since postincrement introduces a
  needless temporary return value that can be expensive for C++ classes.

* Use the following notations when initializing variables:
  `1` for integers, `1.0` for doubles, `1.0f` for floats, `0x01` for bit patterns.

* Prefer `sizeof(type)` to `(sizeof varname)`.
  It is more traditional and more explicit.

* Use `auto` for long type names.

* Avoid complicated template programming.
  Use templates for handling numerical types, such as complex or extended precision.
  Some projects, such as BEAST, use templates for specialized purposes, such as
  parametrization of tunable codes.

* Do not use Boost for codes meant for public releases, such as numerical libraries.
  Use it for small projects.

* Feel free to use C++11 features.

Use of enum
-----------

Use `enum` whenever possible.
More type safety is good.
Wrap then in typedefs, so the word `enum` only shows up in the definition.

Use of const
------------

The use of `const` is strongly encouraged in external and internal interfaces.

Use `const` such that the declaration can be read from right to left.

> `const` is viral: if you pass a `const` variable to a function,
> that function must have `const` in its prototype.
>
> Intel MKL uses `const` for all input function parameters.
> NVIDIA cuBLAS does not use `const` for input parameters passed by value,
> only for input parameters passed by pointer.

Use of restrict
---------------

Avoid `restrict`. Only use if there is a clear performance benefit.

Exceptions
----------

You may use C++ exceptions.
[Google Style Guide](https://google.github.io/styleguide/cppguide.html#Exceptions)
argues against it, but Trilinos uses it.
Pick one way or the other and stick to it.

If you use exceptions, list exceptions in function comments, but not in signatures.
Listing in signatures is a bad idea (http://www.gotw.ca/publications/mill22.htm),
and it is deprecated in C++11.

Integer Types
-------------

Most of the time, use the `int` type and safely assume that it is at least 32 bits.
Do not assume that it is more than 32 bits, though.
If you need any other integer type than `int`, use a precise-width integer type
from `<cstdint>`.
Always use `size_t` to describe the size of a memory region or offset.

Do not use unsigned types unless specifically required.
In particular, do not use unsigned types to say a number will never be negative.
Use `int64_t` for integers for which 32 bits is not enough.
If you really need an unsigned type, use the width-specific type,
even for a 32-bit integer. I.e., use `uint32_t` instead of `unsigned int`.
Basically, never use the `unsigned` keyword.

> The use of unsigned types for loop counters can introduce bugs and prevent
> compiler optimizations.

Do not use a larger type than `int` for a variable, just because it will be used
in an intermediate calculation that may overflow, such as address calculation.
Use explicit casts to `size_t` for all address arithmetic that may overflow.
Declare variables only of the size required for the maximum value the variable
may store. Also, do not use integer types shorter than `int` unless specifically
required, e.g., to minimize the memory footprint of a large array.

NULL, nullptr, 0
----------------

In C++ use `nullptr`, which removes ambiguity. `NULL` is an integer (0);
`nullptr` is a pointer. You can take `sizeof(nullptr)`.

64-bit Addressing
-----------------

In numerical codes, the need for using 64-bit variables to index arrays is rare,
but the chance of the memory offset not fitting in 32 bits is high. Therefore:

* Always use a cast to `size_t` when indexing large arrays involves
  arithmetic operations.

* Use a cast to `size_t` inside all memory allocation functions.

```
#!cpp

array[(size_t)...

malloc((size_t)...
```

> Generally, we assume the matrix dimensions can be passed as 32-bit `int`,
> only the offset (i + j * lda) needs to be computed as 64-bit.
> So, we can use the 32-bit LAPACK interface.
> One place this fails is passing `lwork`, if it needs O(m * n) workspace,
> which it does for the `syev` and `gesvd` routines.

Preprocessing Macros
--------------------

Do not use macros for anything else than:

* conditionally including software dependencies,
* guards in header files preventing multiple inclusion.

This code summarizes the legitimate uses of macros:

```
#!cpp

#ifndef ICL_PTHREAD_H
#define ICL_PTHREAD_H

#ifndef __APPLE__
#include <pthread.h>
#else
#include <libkern/OSAtomic.h>

...

#endif // __APPLE__

#endif // ICL_PTHREAD_H
```

Check out the
[Google Style Guide](https://google.github.io/styleguide/cppguide.html#Preprocessor_Macros)
for more explanations.

Concurrency
===========

* POSIX Threads (Pthreads) and OpenMP are preferred ways of multithreading.
  Do not refrain from using recent features. The bottom line is: if it is
  supported in GCC, it is okay to use.

* Prefer spinlocks over regular locks.
  There is no reason for doing otherwise for high performance parallel codes.

* Use [GNU atomic builtins](https://gcc.gnu.org/onlinedocs/gcc-4.4.3/gcc/Atomic-Builtins.html)
  to implement low-level synchronization mechanisms.
  They are supported by all major compilers (GNU, Intel).

> SLATE uses C++ `std::atomic`.

* <strike> Declare all synchronization variables as `volatile`, even if you are
  only accessing them using atomic builtins.
  If a variable can be accessed by more than one thread, it needs to be
  `volatile` to prevent compiler from applying optimizations that might result
  in incorrect code. </strike>

> This misunderstands `volatile`. Read Scott Meyers, "Effective Modern C++",
> Item 40: Use `std::atomic` for concurrency, `volatile` for special memory
> (e.g. memory-mapped I/O).

* If you find yourself considering the use of memory barriers,
  you went too low-level. Use atomic builtins or spinlocks instead.

Building with Missing Dependencies
==================================

> !!! This section violates xSDK policies such as (M9) using a
> well-defined name space and (M12) linking with external dependencies.
> Consider a library linked with stub MPI or OpenMP functions; it cannot
> be used in an application that links with real MPI or OpenMP due to
> name collisions. !!!

If a certain environment is missing a mainstream component,
e.g., Pthread spinlocks on OSX, do not create a new abstraction layer,
but implement the missing functions on top of the available native functions
(e.g., PLASMA on MS Windows, PULSAR on OSX).

```
#!cpp

#ifndef __APPLE__
#include <pthread.h>
#else
#include <libkern/OSAtomic.h>

typedef OSSpinLock pthread_spinlock_t;

inline int pthread_spin_lock(pthread_spinlock_t *lock) {
    OSSpinLockLock(lock);
    return 0;
}
...
#endif // __APPLE__
```

Similar principle applies when a component is completely missing.
Use the API as if the component was there.
Provide a header file with stubs for the missing functions.
Write your code such that it works correctly if stubs are called
instead of the missing component.
PULSAR is written that way with respect to support for MPI and CUDA.
It may not always be possible, but at least investigate the possibility.

Naming
======

* Use abbreviations within reason. Follow common conventions.
  Name a variable `retval` rather than `return_value`.

* There are no special requirements for global variables,
  which should be rare anyway, but if you use them, consider prefixing them
  with `g_` or some other marker to easily distinguish it from local variables.

File Names
----------

Most projects already have a convention.
Follow the project's convention.

In C++ the file name should match the class name.
Consider using the Trilinos convention:
`<NameSpace>_<ClassName>`, i.e., namespace using Pascal case,
underscore, class name using Pascal case,
e.g., `Magma_SomeClass.hh`, `Magma_SomeClass.cc`.

Use `.h` and `.c` extensions for C files, and `.hh` and `.cc` for C++ files.

Type Names
----------

* In C++ use Pascal case for class names, e.g., `MyExcitingClass`.

* In C use camel case for type names followed by `_t`, e.g., `myExcitingType_t`
  (NVIDIA's convention).

Wrap enums and structs in typedefs and do not use `enum` and `struct` keywords
when declaring variables.

Variable Names
--------------

* The names of variables and data members are all lowercase,
  with underscores between words.
  Data members of classes (but not structs) additionally have trailing underscores.
  For instance:

    - `a_local_variable`,
    - `a_struct_data_member`,
    - `a_class_data_member_`.

Constant Names
--------------

Use Pascal case for constants, i.e., `const int ArraySize`.

> This is Microsoft's convention.
> Google's convention is to use a leading `k`, i.e., `kArraySize`,
> which seems a little arbitrary and looks somewhat awkward.

Prefix global variables with `g_` and global constants with `G_`, e.g.,
`g_scary_global_variable`, `G_ScaryGlobalConstant`.

Function Names
--------------

Use snake case for C function names and C++ method names,
e.g., `my_awesome_c_function`, `my_awesome_cpp_method`.

> SLATE used lowerCamelCase for methods, but there has been discussion
> to make them snake_case.

Namespace Names
---------------

Namespace names are Pascal case (Trilinos, Microsoft).

> SLATE uses lowercase for namespaces.

Enumerator Names
----------------

Individual enumerators should be named like constants, i.e., Pascal case.

Macro Names
-----------

1. Do not use macros.

2. If you really need to use a macro, see 1.

Comments
========

* Use C99/C++ style `//` comments.

* Comment your code heavily, but do not state the obvious.

* If a comment is a sentence, start with a capital letter and end with a period.

* If a comment is not a sentence, start with a small letter and do not end with a period.

* Start each file with a boilerplate.

* Do not duplicate comments in both the `.hh` and the `.cc` file.
  Duplicated comments diverge.

Function Comments
-----------------

Use Doxygen (`///`) for function comments:

* Start with `@brief`,
* Follow with extended description if necessary, indented one more space
  than `@`.
* Follow with parameters and indicate direction (in, out, inout),
* Follow with return values,
* Separate the name of the parameter from the description with a dash,
* Do not capitalize the description,
* Do not follow with a period.

```
#!cpp

///
/// @brief Solves a complex problem.
///
///  Uses such and such algorith
///  with such and such properties.
///
/// @param[in] n - array size
/// @param[in] array - array of input data
/// ...
/// @param[out] result - array of output data
///
/// @returns error code
///
```

Use `@retval` for a list of discrete return values.

```
#!cpp

/// @retval  0 - success
/// @retval -1 - failure
```

The descriptions should be declarative ("Solves a problem.")
rather than imperative ("Solve a problem").
This is the convention of Google, and also LAPACK.

If the function does something trivial, just skip the comment.
It is quite common for destructors not to have header comments.

> SLATE skips `@brief` in favor of Doxygen's autobrief feature.

Variable Comments
-----------------

* Local variables should have names descriptive enough to not require comments.

* Use Doxygen (`///<`) for data members of classes and structures.

Implementation Comments
-----------------------

Use standard C++ (`//`) comments (not Doxygen) for implementation comments.
Put the comments before the codes.
Do not use inline comments. They make the code harder to read and make it hard
to respect the 80-characters line limit.

TODO Comments
-------------

Use the Doxygen `@todo` tag for code that is temporary, a short-term solution,
or good-enough but not perfect.

```
#!cpp

// @todo Make it better.
// @todo (Jakub) Make it even better.
// @todo (kurzak@eecs.utk.edu) Make it yet better.
```

Formatting
==========

Line Length
-----------

Each line of text in your code should be at most 80 characters long.

> It is a Google rule and is easier to follow than you may think.
> [Google Style Guide](https://google.github.io/styleguide/cppguide.html#Line_Length)
> gives some good reasons for this rule.
> Many ICL projects are pretty good at following this rule.

Indentation
-----------

Use only spaces (no tabs), and indent 4 spaces at a time.

Most editors can be configured to insert spaces instead of tabs.
For vim, put these in ~/.vimrc:
```
"set shiftwidth=4
"set softtabstop=4
"set expandtab
```

For emacs, put these in ~/.emacs:
```
(setq-default c-default-style "k&r"
              c-basic-offset 4
              tab-width 4
              indent-tabs-mode nil)
```

For jedit, set Tab width to 4 and check "Soft (emulated with spaces) tabs"

Function Declarations and Definitions
-------------------------------------

Use one of the following styles in declarations and definitions of functions
that don't fit in a single line:

```
#!cpp

ReturnType ClassName::ReallyLongFunctionName(Type par_name1, Type par_name2,
                                             Type par_name3)
{
    DoSomething();
    ...
}
```

```
#!cpp

ReturnType ClassName::ReallyLongFunctionName(
    Type par_name1, Type par_name2, Type par_name3)
{
    DoSomething();
    ...
}
```

```
#!cpp

ReturnType ClassName::ReallyLongFunctionName(
    Type par_name1,
    Type par_name2,
    Type par_name3)
{
    DoSomething();
    ...
}
```

Some points to note:

* Choose good parameter names.
  Try to be consistent with existing codes (e.g., LAPACK) or literature.

* Never omit parameter names in declarations.

* The open parenthesis is always on the same line as the function name.

* There is never a space between the parentheses and the parameters.

* The open curly brace is always on the start of the next line after
  function declaration.
  The exception to the rule are really short functions, e.g., accessors
  in the body of a class.
  If this is the case, then everything can be in a single line
  (if it fits in the 80-characters limit).
  There should be a space between the close parenthesis
  and the open curly brace.

* The close curly brace is either on the last line by itself
  or on the same line as the open curly brace.

Function Calls
--------------

Use one of the following styles when calling functions
that don't fit in a single line:

```
#!cpp

bool result = ReallyLongFunctionName(ReallyLongArguent1,
                                     ReallyLongArguent2,
                                     ReallyLongArguent3);
```

```
#!cpp

bool result = ReallyLongFunctionName(
    ReallyLongArguent1, ReallyLongArguent2, ReallyLongArguent3);
```

```
#!cpp

bool result = ReallyLongFunctionName(
    ReallyLongArguent1,
    ReallyLongArguent2,
    ReallyLongArguent3);
```

```
#!cpp

bool result =
    ReallyLongFunctionName(
        ReallyLongArguent1,
        ReallyLongArguent2,
        ReallyLongArguent3);
```

Functions may have natural line breaking patterns, e.g.:

```
#!cpp

cblas_dgemm(
    CblasColMajor,
    CblasNoTrans, CblasNoTrans,
    m, n, k,
    1.0, a, lda,
         b, ldb,
    1.0, c, ldc);
```

Conditionals
------------

The boilerplate for conditionals is:

```
#!cpp

if (condition) {
    ...
}
else if (...) {
    ...
}
else {
    ...
}
```

Note:

* a space between `if` and the opening parenthesis,
* no spaces inside the parentheses,
* the opening curly brace in the same line,
* a space between the closing parenthesis and the opening curly brace,
* closing curly brace in a separate line,
* `else` in the next line after the closing curly brace,
* the opening curly brace of `else` in the same line after a space,
* same goes for `else if`.

Short conditional statements may be written on one line
if this enhances readability.

```
#!cpp

if (x == foo) return bar;
```

This is not allowed when the `if` statement has an `else`.

Curly braces are not required for single-line statements.
However, if one part of an if-else statement uses curly braces,
the other part must too.

Loops
-----

The boilerplates for loops are:

```
#!cpp

for (int i = 0; i < SomeNumber; ++i)
    ...

for (int i = 0; i < SomeNumber; ++i) {
    ...
}
```

```
#!cpp

while (condition)
    ...

while (condition) {
    ...
}

do {
    ...
} while (condition);
```

Switch
------

The boilerplate for `switch` is:

```
#!cpp

switch (var) {
case 0:
    ...
    break;
case 0:
    ...
    break;
default:
    assert(false);
}
```

If the default case should never execute, simply assert false.

> SLATE indents `case` statements one level, since they're inside a block.

Pointer and Reference Expressions
---------------------------------

No spaces around period or arrow. Pointer operators do not have trailing spaces.
The following are examples of correctly-formatted pointer and reference expressions:

```
#!cpp

x = *p;
p = &x;
x = r.y;
x = r->y;
```

Note that:

* There are no spaces around the period or arrow when accessing a member.
* Pointer operators have no space after the * or &.

When declaring a pointer variable or argument, always place the asterisk
adjacent to the variable name:

```
#!cpp

char *c;
const string &str;
```

> SLATE places * and & next to the type, since it is part of the type.

Boolean Expressions
-------------------

When you have a boolean expression that is longer than the standard line length,
break lines like this:

```
#!cpp

if (this_one_thing > this_other_thing &&
    a_third_thing == a_fourth_thing &&
    yet_another && last_one) {
    ...
}
```

Note that when the code wraps in this example, both of the && operators
are at the end of the line.
Also note that you should always use the punctuation operators,
such as && and ~, rather than the word operators, such as `and` and `compl`.

> SLATE places operators at the beginning of the line for clarity,
> consistent with math typesetting conventions.
> (Although older SLATE code doesn't.)
> See AMS "Mathematics into Type", section 3.3.5.

Return Values
-------------

Do not needlessly surround the `return` expression with parentheses.
Use parentheses in `return expr`; only where you would use them in `x = expr;`.

```
#!cpp

return result;

return (some_long_condition && another_condition);
```

Preprocessor Directives
-----------------------

The hash mark that starts a preprocessor directive should always be
at the beginning of the line.
Even when preprocessor directives are within the body of indented code,
the directives should start at the beginning of the line.

> SLATE indents preprocessor directives and code inside #if ... #else ... #endif
> for improved readability (excluding header guards).

Class Format
------------

The class boilerplate is:

```
#!cpp

class MyClass {
    friend class FriendClass;

public:
    MyClass(int value) : value_(value), pointer_(nullptr) {}
    ~MyClass();

    int get_value() { return value_; }
    int get_pointer() { return pointer_; }

    void awesome_public_method();
    void another_public_method();

private:
    void awesome_secret_method();
    void another_secret_method();

    int value_;
    int *poiter_;
};
```

Note that:

* The `friend` keyword is indented.
* The `private` and `public` keywords are not indented.
* Except for the first instance, these keywords are preceded by a blank line.
* There are no blank lines after these keywords.
* The `friend` section is first, followed by the `public` section
  and the `private` section.
* In the private section, methods are first, followed by attributes.

> This is the order of the
> [Google Style Guide](https://google.github.io/styleguide/cppguide.html#Class_Format),
> the Trilinos project, and also Doxygen produces documentation
> in this order.
>
> The order of methods is top to bottom, e.g., the deeper in the call tree
> the lower on the list. This applies both to the order of declarations
> and the order or definitions, which should be identical.

Constructor Initializer Lists
-----------------------------

Constructor initializer lists can be all on one line or with subsequent lines
indented four spaces.
The acceptable formats for initializer lists are:

```
#!cpp

    MyClass(int value)
        : value_(value), pointer_(nullptr)
    {
        ...
    }

    MyClass(int value)
        : value_(value),
          pointer_(nullptr)
    {
        ...
    }
```

In each case the closing curly brace can be in the same line as the opening
curly brace if it fits.

> Initialization list is part of constructor's definition,
> so you need to define it at the same place you define constructor's body.

Namespace Formatting
--------------------

The contents of namespaces are not indented.

When declaring nested namespaces, put each namespace on its own line.

```
#!cpp

namespace plasma {
namespace internal {

...

} // namespace internal
} // namespace plasma
```

Horizontal Whitespace
---------------------

* Never put trailing whitespace at the end of a line.
> Trailing whitespace can cause extra work for others editing the same file,
> when they merge.

* Opening curly braces should always have a space before them.

* Semicolons have no space before them.

* Put spaces around the colon in initializer lists.
  The same applies for inheritance if you end up using it.

* For inline function implementations, put spaces between the braces
  and the implementation itself.

* No spaces inside empty parentheses and curly braces.

* Put space after the keyword in conditions and loops.

* `for` loops always have a space after the semicolon,
  and never a space before the semicolon.

* No space before colon in a `switch` case.
  A space after a colon if there's code after it.

* Assignment operators always have spaces around them.

* Other binary operators usually have spaces around them, but it's
  okay to remove spaces around factors.
  Parentheses should have no internal padding.

* No spaces separating unary operators and their arguments.

```
#!cpp

x = 0;

v = w * x + y / z;
v = w*x + y/z;
v = w * (x + z);

x = -5;
++x;
if (x && !y)
    ...
```

* In templates, no spaces inside the angle brackets.
  No spaces between type and pointer.
  C++11 notation for nesting.

```
#!cpp

vector<string> x;

y = static_cast<char*>(x);

vector<char*> x;

set<list<string>> x;
```

Vertical Whitespace
-------------------

* Never use more than a single blank line.

Horizontal Rules
----------------

* Horizontal rules are great.
  Use whatever works for you.
  Be consistent throughout each project.
  Make them 80 characters wide.

```
#!cpp

//------------------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////

/******************************************************************************/
```
