#!/usr/bin/env python
import os
import re
import argparse

description = '''\
Generates Fortran 2003 interface from PLASMA header files.'''

help = '''\
----------------------------------------------------------------------
Example uses:

  fortran_gen.py plasma_*.h
      generates plasma_mod.f90 with module plasma

----------------------------------------------------------------------
'''

# ------------------------------------------------------------
# command line options
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=description,
    epilog=help )
parser.add_argument('--prefix',        action='store', help='Prefix for variables in Makefile', default='./')
parser.add_argument('args', nargs='*', action='store', help='Files to process')
opts = parser.parse_args()

# ------------------------------------------------------------
# set indentation in the f90 file
tab = "    "
indent = tab

# module name
module_name = "plasma"

# translation_table of types
types_dict = {
    "int":               ("integer(kind=c_int)"),
    "size_t":            ("integer(kind=c_size_t)"),
    "char":              ("character(kind=c_char)"),
    "double":            ("real(kind=c_double)"),
    "float":             ("real(kind=c_float)"),
    "plasma_complex64_t":("complex(kind=c_double_complex)"),
    "plasma_complex32_t":("complex(kind=c_float_complex)"),
    "plasma_enum_t":     ("integer(kind=c_int)"),
    "plasma_desc_t":     ("type(plasma_desc_t)"),
    "plasma_workspace_t":("type(plasma_workspace_t)"),
    "plasma_sequence_t": ("type(plasma_sequence_t)"),
    "plasma_request_t":  ("type(plasma_request_t)"),
    "plasma_context_t":  ("type(plasma_context_t)"),
    "plasma_barrier_t":  ("type(plasma_barrier_t)"),
    "pthread_t":         ("integer(kind=c_int)"),
    "lua_State":         ("integer(kind=c_int)"),
    "void":              ("type(c_ptr)"),
}

# translation_table with names of auxiliary variables
return_variables_dict = {
    "double":            ("value"),
    "float":             ("value"),
    "plasma_desc_t":     ("desc"),
    "plasma_workspace_t":("workspace"),
    "plasma_sequence_t": ("sequence"),
    "plasma_request_t":  ("request"),
    "plasma_context_t":  ("context"),
    "plasma_enum_t":     ("enum"),
    "lua_State":         ("lua_state"),
}

# name arrays which will be translated to assumed-size arrays, e.g. pA(*)
arrays_names_2D = ["pA", "pB", "pC", "pAB", "pQ", "pX", "pAs"]
arrays_names_1D = ["ipiv", "values", "work", "W"]

# exclude inline functions from the interface
exclude_list = ["inline"]

# ------------------------------------------------------------

# global list used to determine derived types
derived_types = []

def polish_file(whole_file):
    """Preprocessing and cleaning of the header file.
       Do not change the order of the regular expressions !
       Works with a long string."""

    clean_file = whole_file

    # borrowed from cfwrapper.py
    # Remove C comments:
    clean_file = re.sub(r"(?s)/\*.*?\*/", "", clean_file)
    clean_file = re.sub(r"//.*", "", clean_file)
    # Remove C directives (multilines then monoline):
    clean_file = re.sub(r"(?m)^#(.*[\\][\n])+.*?$", "", clean_file)
    clean_file = re.sub("(?m)^#.*$", "", clean_file)
    clean_file = re.sub("(?m)#.*", "", clean_file)
    # Remove TABs and overnumerous spaces:
    clean_file = clean_file.replace("\t", " ")
    clean_file = re.sub("[ ]{2,}", " ", clean_file)
    # Remove extern C statement:
    clean_file = re.sub("(?m)^(extern).*$", "", clean_file)
    # Remove empty lines:
    clean_file = re.sub(r"(?m)^\n$", "", clean_file)
    # Merge structs
    clean_file = re.sub(r"(?m)$", "", clean_file)

    # Merge string into single line
    clean_file = re.sub(r"\n", "", clean_file)

    # Split the line based on ";" and "}"
    clean_file = re.sub(r";", "\n", clean_file)
    clean_file = re.sub(r"}", "}\n", clean_file)

    return clean_file

def preprocess_list(initial_list):
    """Preprocessing and cleaning of the header file.
       Works with a list of strings.
       Produces a new list in which each function, enum or struct
       corresponds to a single item."""

    # merge braces
    list1 = []
    merged_line = ""
    nopen = 0
    inStruct = False
    for line in initial_list:

        if (line.find("struct") > -1):
            inStruct = True

        if (inStruct):
            split_character = ","
        else:
            split_character = ""

        nopen += line.count("{") - line.count("}")
        merged_line += line + split_character

        if (nopen <= 0):
            list1.append(merged_line)
            merged_line = ""
            isOpen   = False
            inStruct = False
            nopen = 0

    # merge structs
    list2 = []
    merged_line = ""
    for line in list1:

        merged_line += line

        if (line.find("struct") == -1):
            list2.append(merged_line)
            merged_line = ""

    # clean orphan braces
    list3 = []
    for line in list2:

        if (line.strip() != "}"):
            list3.append(line)

    #print '\n\n'.join(list3)

    return list3

def parse_triple(string):
    """Parse string of
       type (*)name
       into triple of [type, pointer, name]"""

    parts = string.split()
    if (len(parts) < 2 or len(parts) > 3):
        print("Error: Cannot detect type for ", string)

    type_part = str.strip(parts[0])

    if (len(parts) == 2):
        name_with_pointer = str.strip(parts[1])
        if (name_with_pointer.find("**") > -1):
            pointer_part = "**"
            name_part = name_with_pointer.replace("**", "")
        elif (name_with_pointer.find("*") > -1):
            pointer_part = "*"
            name_part    = name_with_pointer.replace("*", "")
        else:
            pointer_part = ""
            name_part    = name_with_pointer

    elif (len(parts) == 3):
        if (str.strip(parts[1]) == "**"):
            pointer_part = "**"
            name_part    = str.strip(parts[2])
        elif (str.strip(parts[1]) == "*"):
            pointer_part = "*"
            name_part    = str.strip(parts[2])
        else:
            print("Error: Too many parts for ", string)

    name_part = name_part.strip()

    return [type_part, pointer_part, name_part]


def parse_enums(preprocessed_list):
    """Each enum will be parsed into a list of its arguments."""

    enum_list = []
    for proto in preprocessed_list:

        # extract the part of the function from the prototype
        fun_parts = proto.split("{")

        if (fun_parts[0].strip() == "enum"):
            args_string = fun_parts[1];
            args_string = re.sub(r"}", "", args_string)

            args_list = args_string.split(",")
            params = [];
            for args in args_list:
                if (args != ""):
                    values = args.split("=")

                    name = values[0].strip()
                    if (len(values) > 1):
                       value = values[1].strip()
                    else:
                       if (len(params) > 0):
                          value = str(int(params[len(params)-1][1]) + 1)
                       else:
                          value = "0"

                    params.append([name, value])

            enum_list.append(params)

    return enum_list


def parse_structs(preprocessed_list):
    """Each struct will be parsed into a list of its arguments."""

    struct_list = []
    for proto in preprocessed_list:

        # extract the part of the function from the prototype
        fun_parts = proto.split("{")

        if (fun_parts[0].find("struct") > -1):
            args_string = fun_parts[1]
            parts = args_string.split("}")
            args_string = parts[0].strip()
            args_string = re.sub(r"volatile", "", args_string)
            if (len(parts) > 1):
                name_string = parts[1]
                name_string = re.sub(r"(?m),", "", name_string)
                name_string = name_string.strip()
            else:
                print("Error: Cannot detect name for ", proto)
                name_string = "name_not_recognized"

            args_list = args_string.split(",")
            params = [];
            params.append(["struct","",name_string])
            for arg in args_list:
                if (not (arg == "" or arg == " ")):
                    params.append(parse_triple(arg))

            struct_list.append(params)
            derived_types.append(name_string)

    # reorder the list so that only defined types are exported
    goAgain = True
    while (goAgain):
        goAgain = False
        for istruct in range(0,len(struct_list)-1):
            struct = struct_list[istruct]
            for j in range(1,len(struct)-1):
                type_name = struct_list[istruct][j][0]

                if (type_name in derived_types):

                    # try to find the name in the registered types
                    definedEarlier = False
                    for jstruct in range(0,istruct):
                        struct2 = struct_list[jstruct]
                        that_name = struct2[0][2]
                        if (that_name == type_name):
                            definedEarlier = True

                    # if not found, try to find it behind
                    if (not definedEarlier):
                        definedLater = False
                        for jstruct in range(istruct+1,len(struct_list)-1):
                            struct2 = struct_list[jstruct]
                            that_name = struct2[0][2]
                            if (that_name == type_name):
                                index = jstruct
                                definedLater = True

                        # swap the entries
                        if (definedLater):
                            print("Swapping " + struct_list[istruct][0][2] + " and " + struct_list[index][0][2])
                            tmp = struct_list[index]
                            struct_list[index] = struct_list[istruct]
                            struct_list[istruct] = tmp
                            goAgain = True
                            break
                        else:
                            print("Error: Cannot find a derived type " + type_name + " in imported structs.")

            if (goAgain):
                break

    return struct_list


def parse_prototypes(preprocessed_list):
    """Each prototype will be parsed into a list of its arguments."""

    function_list = []
    for proto in preprocessed_list:

        if (proto.find("(") == -1):
            continue

        # extract the part of the function from the prototype
        fun_parts = proto.split("(")
        fun_def   = str.strip(fun_parts[0])

        exclude_this_function = False
        for exclude in exclude_list:
            if (fun_def.find(exclude) != -1):
                exclude_this_function = True

        if (exclude_this_function):
            continue

        # clean keywords
        fun_def = fun_def.replace("^static\s", "")

        # extract arguments from the prototype and make a list from them
        if (len(fun_parts) > 1):
            fun_args = fun_parts[1]
        else:
            fun_args = ""

        fun_args = fun_args.split(")")[0]
        fun_args = fun_args.replace(";", "")
        fun_args = re.sub(r"volatile", "", fun_args)
        fun_args = fun_args.replace("\n", "")
        fun_args_list = fun_args.split(",")

        # generate argument list
        argument_list = []
        # function itself on the first position
        argument_list.append(parse_triple(fun_def))
        # append arguments
        for arg in fun_args_list:
            if (not (arg == "" or arg == " ")):
                arg = arg.replace("const", "")
                argument_list.append(parse_triple(arg))

        # add it only if there is no duplicity with previous one
        is_function_already_present = False
        fun_name = argument_list[0][2]
        for fun in function_list:
            if (fun_name == fun[0][2]):
                is_function_already_present = True

        if (not is_function_already_present):
            function_list.append(argument_list)

    return function_list


def iso_c_interface_type(arg, return_value):
    """Generate a declaration for a variable in the interface."""

    if (arg[1] == "*" or arg[1] == "**"):
        is_pointer = True
    else:
        is_pointer = False

    if (is_pointer):
        f_type = "type(c_ptr)"
    else:
        f_type = types_dict[arg[0]]

    if (not return_value and arg[1] != "**"):
        f_pointer = ", value"
    else:
        f_pointer = ""

    f_name = arg[2]

    f_line = f_type + f_pointer + " :: " + f_name

    return f_line


def iso_c_wrapper_type(arg):
    """Generate a declaration for a variable in the Fortran wrapper."""

    if (arg[1] == "*" or arg[1] == "**"):
        is_pointer = True
    else:
        is_pointer = False

    if (is_pointer and arg[0].strip() == "void"):
        f_type = "type(c_ptr)"
    else:
        f_type = types_dict[arg[0]]

    #if (is_pointer):
    #    f_intent = ", intent(inout)"
    #else:
    #    f_intent = ", intent(in)"

    if (is_pointer):
        if (arg[1] == "*"):
           f_target = ", target"
        else:
           f_target = ", pointer"
    else:
        f_target = ""

    f_name    = arg[2]

    # detect array argument
    if   (is_pointer and f_name in arrays_names_2D):
        f_array = "(*)"
    elif (is_pointer and f_name in arrays_names_1D):
        f_array = "(*)"
    else:
        f_array = ""

    #f_line = f_type + f_intent + f_target + " :: " + f_name + f_array
    f_line = f_type + f_target + " :: " + f_name + f_array

    return f_line

def fortran_interface_enum(enum):
    """Generate an interface for an enum.
       Translate it into constants."""

    # initialize a string with the fortran interface
    f_interface = ""

    # loop over the arguments of the enum
    for param in enum:
        name  = param[0]
        value = param[1]

        f_interface += indent + "integer, parameter :: " + name + " = " + value + "\n"

    return f_interface

def fortran_interface_struct(struct):
    """Generate an interface for a struct.
       Translate it into a derived type."""

    # initialize a string with the fortran interface
    f_interface = ""

    name = struct[0][2]
    f_interface += tab + "type, bind(c) :: " + name + "\n"
    # loop over the arguments of the enum
    for j in range(1,len(struct)):
        f_interface += indent + tab + iso_c_interface_type(struct[j], True)
        f_interface += "\n"

    f_interface += tab + "end type " + name + "\n"

    return f_interface

def fortran_interface_function(function):
    """Generate an interface for a function."""

    # is it a function or a subroutine
    if (function[0][0] == "void"):
        is_function = False
    else:
        is_function = True

    c_symbol = function[0][2]
    f_symbol = c_symbol + "_c"

    used_derived_types = set([])
    for arg in function:
        type_name = arg[0]
        if (type_name in derived_types):
            used_derived_types.add(type_name)

    # initialize a string with the fortran interface
    f_interface = ""
    f_interface += indent + "interface\n"

    if (is_function):
        f_interface += indent + tab + "function "
    else:
        f_interface += indent + tab + "subroutine "

    f_interface += f_symbol + "("

    if (is_function):
        initial_indent = len(indent + tab + "function " + f_symbol + "(") * " "
    else:
        initial_indent = len(indent + tab + "subroutine " + f_symbol + "(") * " "

    # loop over the arguments to compose the first line
    for j in range(1,len(function)):
        if (j != 1):
            f_interface += ", "
        if (j%9 == 0):
            f_interface += "&\n" + initial_indent

        f_interface += function[j][2]

    f_interface += ") &\n"
    f_interface += indent + tab + "  " + "bind(c, name='" + c_symbol +"')\n"

    # add common header
    f_interface += indent + 2*tab + "use iso_c_binding\n"
    # import derived types
    for derived_type in used_derived_types:
        f_interface += indent + 2*tab + "import " + derived_type +"\n"
    f_interface += indent + 2*tab + "implicit none\n"


    # add the return value of the function
    if (is_function):
        f_interface +=  indent + 2*tab + iso_c_interface_type(function[0], True) + "_c"
        f_interface += "\n"

    # loop over the arguments to describe them
    for j in range(1,len(function)):
        f_interface += indent + 2*tab + iso_c_interface_type(function[j], False)
        f_interface += "\n"

    if (is_function):
        f_interface += indent + tab + "end function\n"
    else:
        f_interface += indent + tab + "end subroutine\n"

    f_interface += indent + "end interface\n"

    return f_interface


def fortran_wrapper(function):
    """Generate a wrapper for a function.
       void functions in C will be called as subroutines,
       functions in C will be turned to subroutines by appending
       the return value as the last argument."""

    # is it a function or a subroutine
    if (function[0][0] == "void"):
        is_function = False
    else:
        is_function = True

    c_symbol = function[0][2]
    f_symbol = c_symbol + "_c"

    if (is_function):
        initial_indent_signature = len(indent + "subroutine " + c_symbol + "(") * " "
        initial_indent_call      = len(indent + tab + "info = " + f_symbol + "(") * " "
    else:
        initial_indent_signature = len(indent + "subroutine " + c_symbol + "(") * " "
        initial_indent_call      = len(indent + tab + "call " + f_symbol + "(") * " "

    # loop over the arguments to compose the first line and call line
    signature_line = ""
    call_line = ""
    double_pointers = []
    for j in range(1,len(function)):
        if (j != 1):
            signature_line += ", "
            call_line += ", "

        # do not make the argument list too long
        if (j%9 == 0):
            call_line      += "&\n" + initial_indent_call
            signature_line += "&\n" + initial_indent_signature

        # pointers
        arg_type    = function[j][0]
        arg_pointer = function[j][1]
        arg_name    = function[j][2]

        signature_line += arg_name
        if (arg_pointer == "**"):
            aux_name = arg_name + "_aux"
            call_line += aux_name
            double_pointers.append(arg_name)
        elif (arg_pointer == "*"):
            call_line += "c_loc(" + arg_name + ")"
        else:
            call_line += arg_name

    contains_derived_types = False
    for arg in function:
        if (arg[0] in derived_types):
            contains_derived_types = True

    # initialize a string with the fortran interface
    f_wrapper = ""
    f_wrapper += indent + "subroutine "
    f_wrapper += c_symbol + "("

    # add the info argument at the end
    f_wrapper += signature_line
    if (is_function):
        if (len(function) > 1):
            f_wrapper += ", "

        return_type = function[0][0]
        return_pointer = function[0][1]
        if (return_type == "int"):
            return_var = "info"
        else:
            return_var = return_variables_dict[return_type]

        f_wrapper += return_var

    f_wrapper += ")\n"

    # add common header
    f_wrapper += indent + tab + "use iso_c_binding\n"
    f_wrapper += indent + tab + "implicit none\n"

    # loop over the arguments to describe them
    for j in range(1,len(function)):
        f_wrapper += indent + tab + iso_c_wrapper_type(function[j]) + "\n"

    # add the return info value of the function
    if (is_function):
        if (function[0][1] == "*"):
            f_target = ", pointer"
        else:
            f_target = ""

        # do not export intents
        #f_wrapper += indent + tab + types_dict[return_type] + ", intent(out)" + f_target + " :: " + return_var + "\n"
        f_wrapper += indent + tab + types_dict[return_type] + f_target + " :: " + return_var + "\n"

    f_wrapper += "\n"

    # loop over potential double pointers and generate auxiliary variables for them
    for double_pointer in double_pointers:
        aux_name = double_pointer + "_aux"
        f_wrapper += indent + tab + "type(c_ptr) :: " + aux_name + "\n"
        f_wrapper += "\n"

    if (is_function):
        f_return = return_var
        f_return += " = "
    else:
        f_return = "call "

    # generate the call to the C function
    if (is_function and return_pointer == "*"):
        f_wrapper += indent + tab + "call c_f_pointer(" + f_symbol + "(" + call_line + "), " + return_var + ")\n"
    else:
        f_wrapper += indent + tab + f_return + f_symbol + "(" + call_line + ")\n"

    # loop over potential double pointers and translate them to Fortran pointers
    for double_pointer in double_pointers:
        aux_name = double_pointer + "_aux"
        f_wrapper += indent + tab + "call c_f_pointer(" + aux_name + ", " + double_pointer + ")\n"

    f_wrapper += indent + "end subroutine\n"

    return f_wrapper


def write_module(prefix, module_name, enum_list, struct_list, function_list):
    """Generate a single Fortran module. Its structure will be:
       enums converted to constants
       structs converted to derived types
       interfaces of all C functions
       Fortran wrappers of the C functions"""

    modulefilename = prefix + module_name + "_mod.f90"
    modulefile = open(modulefilename, "w")

    modulefile.write(
'''!>
!> @file
!>
!>  PLASMA is a software package provided by:
!>  University of Tennessee, US,
!>  University of Manchester, UK.
!>
!>  This file was automatically generated by the
!>  fortran_gen.py script.\n
''')

    modulefile.write("module " + module_name + "\n")

    # common header
    modulefile.write(indent + "use iso_c_binding\n")
    modulefile.write(indent + "implicit none\n\n")

    # enums
    if (len(enum_list) > 0):
        modulefile.write(indent + "! C enums converted to constants.\n")

        for enum in enum_list:
            f_interface = fortran_interface_enum(enum)
            modulefile.write(f_interface + "\n")

    # derived types
    if (len(struct_list) > 0):
        modulefile.write(indent + "! C structs converted to derived types.\n")

        for struct in struct_list:
            f_interface = fortran_interface_struct(struct)
            modulefile.write(f_interface + "\n")

    # functions
    if (len(function_list) > 0):
        modulefile.write(indent + "! Interfaces of the C functions.\n")

        for function in function_list:
            f_interface = fortran_interface_function(function)
            modulefile.write(f_interface + "\n")

        modulefile.write(indent + "contains\n\n")

        modulefile.write(indent + "! Wrappers of the C functions.\n")

        for function in function_list:
            f_wrapper = fortran_wrapper(function)
            modulefile.write(f_wrapper + "\n")

    modulefile.write("end module " + module_name + "\n")

    modulefile.close()

    return modulefilename


def main():

    # common cleaned header files
    preprocessed_list = []

    # source header files
    for filename in opts.args:

        # source a header file
        c_header_file = open(filename, 'r').read()

        # clean the string (remove comments, macros, etc.)
        clean_file = polish_file(c_header_file)

        # convert the string to a list of strings
        initial_list = clean_file.split("\n")

        # process the list so that each enum, struct or function
        # are just one item
        nice_list = preprocess_list(initial_list)

        # compose all files into one big list
        preprocessed_list += nice_list

    # register all enums
    enum_list = parse_enums(preprocessed_list)

    # register all structs
    struct_list = parse_structs(preprocessed_list)

    # register all individual functions and their signatures
    function_list = parse_prototypes(preprocessed_list)

    # export the module
    modulefilename = write_module(opts.prefix, module_name, enum_list, struct_list, function_list)
    print("Exported file: " + modulefilename)

# execute the program
main()
