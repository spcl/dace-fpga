# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import dace
from dace import registry, dtypes, symbolic
from dace.codegen import cppunparse
from dace.config import Config
from dace.codegen import exceptions as cgx
from dace.codegen.codeobject import CodeObject
from dace.codegen.dispatcher import DefinedType
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets import cpp
from dace_fpga.codegen import fpga
from dace.codegen.common import codeblock_to_cpp
from dace.codegen.tools.type_inference import infer_expr_type, infer_types
from dace.frontend.python.astutils import rname, unparse, evalnode
from dace.frontend import operations
from dace.sdfg import find_input_arraynode, find_output_arraynode
from dace.sdfg import nodes, utils as sdutils
from dace.codegen.common import sym2cpp
from dace.sdfg import SDFGState
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import ControlFlowRegion, StateSubgraphView
import dace.sdfg.utils as utils
from dace.symbolic import evaluate
from collections import defaultdict

# Lists allowed modules and maps them to OpenCL
_OPENCL_ALLOWED_MODULES = {"builtins": "", "dace": "", "math": ""}

class OpenCLUnparser(cppunparse.CPPUnparser):
    """Methods in this class recursively traverse an AST and
    output C++ source code for the abstract syntax; original formatting
    is disregarded. """

    def _Assign(self, t):
        self.fill()

        # Handle the case of a tuple output
        if len(t.targets) > 1:
            self.dispatch_lhs_tuple(t.targets)
        else:
            target = t.targets[0]
            if isinstance(target, ast.Tuple):
                if len(target.elts) > 1:
                    self.dispatch_lhs_tuple(target.elts)
                    target = None
                else:
                    target = target.elts[0]

            if target and not isinstance(target, (ast.Subscript, ast.Attribute)) and not self.locals.is_defined(
                    target.id, self._indent):

                # if the target is already defined, do not redefine it
                if self.defined_symbols is None or target.id not in self.defined_symbols:
                    # we should try to infer the type
                    if self.type_inference is True:
                        # Perform type inference
                        # Build dictionary with symbols
                        def_symbols = {}
                        def_symbols.update(self.locals.get_name_type_associations())
                        def_symbols.update(self.defined_symbols)
                        inferred_symbols = infer_types(t, def_symbols)
                        inferred_type = inferred_symbols[target.id]
                        if inferred_type is None:
                            raise RuntimeError(f"Failed to infer type of \"{target.id}\".")

                        self.locals.define(target.id, t.lineno, self._indent, inferred_type)
                        if self.language == dace.dtypes.Language.OpenCL and (inferred_type is not None
                                                                             and inferred_type.veclen > 1):
                            # if the veclen is greater than one, this should be defined with a vector data type
                            self.write("{}{} ".format(dace.dtypes._OCL_VECTOR_TYPES[inferred_type.type],
                                                      inferred_type.veclen))
                        elif self.language == dace.dtypes.Language.OpenCL:
                            self.write(dace.dtypes._OCL_TYPES[inferred_type.type] + " ")
                        else:
                            self.write(dace.dtypes._CTYPES[inferred_type.type] + " ")
                    else:
                        raise NotImplementedError("Type inference is required for OpenCL code generation")

            # dispatch target
            if target:
                self.dispatch(target)

        self.write(" = ")
        self.dispatch(t.value)
        #self.dtype = inferred_type
        self.write(';')


class OpenCLDaceKeywordRemover(cpp.DaCeKeywordRemover):
    """
    Removes Dace Keywords and enforces OpenCL compliance
    """

    nptypes_to_ctypes = {'float64': 'double', 'float32': 'float', 'int32': 'int', 'int64': 'long'}
    nptypes = ['float64', 'float32', 'int32', 'int64']
    ctypes = [
        'bool', 'char', 'cl_char', 'unsigned char', 'uchar', 'cl_uchar', 'short', 'cl_short', 'unsigned short',
        'ushort', 'int', 'unsigned int', 'uint', 'long', 'unsigned long', 'ulong', 'float', 'half', 'size_t',
        'ptrdiff_t', 'intptr_t', 'uintptr_t', 'void', 'double'
    ]

    def __init__(self, sdfg, defined_vars, memlets, codegen):
        self.sdfg = sdfg
        self.defined_vars = defined_vars
        # Keep track of the different streams used in a tasklet
        self.used_streams = []
        self.width_converters = set()  # Pack and unpack vectors
        self.dtypes = {k: v[3] for k, v in memlets.items() if k is not None}  # Type inference
        # consider also constants: add them to known dtypes
        for k, v in sdfg.constants.items():
            if k is not None:
                self.dtypes[k] = v.dtype

        super().__init__(sdfg, memlets, sdfg.constants, codegen)

    def visit_Assign(self, node):
        from dace_fpga.codegen.intel_fpga import REDUCTION_TYPE_TO_PYEXPR
        target = rname(node.targets[0])
        if target not in self.memlets:
            # If we don't have a memlet for this target, it could be the case
            # that on the right hand side we have a constant (a Name or a subscript)
            # If this is the case, we try to infer the type, otherwise we fallback to generic visit
            if ((isinstance(node.value, ast.Name) and node.value.id in self.constants)
                    or (isinstance(node.value, ast.Subscript) and node.value.value.id in self.constants)):
                dtype = infer_expr_type(unparse(node.value), self.dtypes)
                value = cppunparse.cppunparse(self.visit(node.value), expr_semicolon=False)
                code_str = "{} {} = {};".format(dtype, target, value)
                updated = ast.Name(id=code_str)
                return updated
            else:
                return self.generic_visit(node)

        memlet, nc, wcr, dtype = self.memlets[target]
        is_scalar = not isinstance(dtype, dtypes.pointer)

        value = cppunparse.cppunparse(self.visit(node.value), expr_semicolon=False)

        veclen_lhs = self.sdfg.data(memlet.data).veclen
        try:
            dtype_rhs = infer_expr_type(unparse(node.value), self.dtypes)
        except SyntaxError:
            # non-valid python
            dtype_rhs = None

        if dtype_rhs is None:
            # If we don't understand the vector length of the RHS, assume no
            # conversion is needed
            veclen_rhs = veclen_lhs
        else:
            veclen_rhs = dtype_rhs.veclen

        if ((veclen_lhs > veclen_rhs and veclen_rhs != 1) or (veclen_lhs < veclen_rhs and veclen_lhs != 1)):
            raise ValueError("Conflicting memory widths: {} and {}".format(veclen_lhs, veclen_rhs))

        if veclen_rhs > veclen_lhs:
            veclen = veclen_rhs
            ocltype = fpga.vector_element_type_of(dtype).ocltype
            self.width_converters.add((True, ocltype, veclen))
            unpack_str = "unpack_{}{}".format(ocltype, cpp.sym2cpp(veclen))

        if veclen_lhs > veclen_rhs and isinstance(dtype_rhs, dace.pointer):
            veclen = veclen_lhs
            ocltype = fpga.vector_element_type_of(dtype).ocltype
            self.width_converters.add((False, ocltype, veclen))
            pack_str = "pack_{}{}".format(ocltype, cpp.sym2cpp(veclen))
            # TODO: Horrible hack to not dereference pointers if we have to
            # unpack it
            if value[0] == "*":
                value = value[1:]
            value = "{}({})".format(pack_str, value)

        defined_type, _ = self.defined_vars.get(target)

        if defined_type == DefinedType.Pointer:
            # In case of wcr over an array, resolve access to pointer, replacing the code inside
            # the tasklet
            if isinstance(node.targets[0], ast.Subscript):

                if veclen_rhs > veclen_lhs:
                    code_str = unpack_str + "({src}, &{dst}[{idx}]);"
                else:
                    code_str = "{dst}[{idx}] = {src};"
                slice = self.visit(node.targets[0].slice)
                if (isinstance(slice, ast.Slice) and isinstance(slice.value, ast.Tuple)):
                    subscript = unparse(slice)[1:-1]
                else:
                    subscript = unparse(slice)
                if wcr is not None:
                    redtype = operations.detect_reduction_type(wcr)
                    red_str = REDUCTION_TYPE_TO_PYEXPR[redtype].format(a="{}[{}]".format(memlet.data, subscript),
                                                                       b=value)
                    code_str = code_str.format(dst=memlet.data, idx=subscript, src=red_str)
                else:
                    code_str = code_str.format(dst=target, idx=subscript, src=value)
            else:  # Target has no subscript
                if veclen_rhs > veclen_lhs:
                    code_str = unpack_str + "({}, {});".format(value, target)
                else:
                    if self.defined_vars.get(target)[0] == DefinedType.Pointer:
                        code_str = "*{} = {};".format(target, value)
                    else:
                        code_str = "{} = {};".format(target, value)
            updated = ast.Name(id=code_str)

        elif (defined_type == DefinedType.Stream or defined_type == DefinedType.StreamArray):
            if memlet.dynamic or memlet.num_accesses != 1:
                updated = ast.Name(id="write_channel_intel({}, {});".format(target, value))
                self.used_streams.append(target)
            else:
                # in this case for an output stream we have
                # previously defined an output local var: we use that one
                # instead of directly writing to channel
                updated = ast.Name(id="{} = {};".format(target, value))
        elif memlet is not None and (not is_scalar or memlet.dynamic):
            newnode = ast.Name(id="*{} = {}; ".format(target, value))
            return ast.copy_location(newnode, node)
        elif defined_type == DefinedType.Scalar:
            code_str = "{} = {};".format(target, value)
            updated = ast.Name(id=code_str)
        else:
            raise RuntimeError("Unhandled case: {}, type {}, veclen {}, "
                               "memory size {}, {} accesses".format(target, defined_type, veclen_lhs, veclen_lhs,
                                                                    memlet.num_accesses))

        return ast.copy_location(updated, node)

    def visit_BinOp(self, node):
        if node.op.__class__.__name__ == 'Pow':

            # Special case for integer power: do not generate dace namespaces (dace::math) but just call pow
            if not (isinstance(node.right,
                               (ast.Num, ast.Constant)) and int(node.right.n) == node.right.n and node.right.n >= 0):

                left_value = cppunparse.cppunparse(self.visit(node.left), expr_semicolon=False)

                try:
                    unparsed = symbolic.pystr_to_symbolic(evalnode(node.right, {
                        **self.constants,
                        'dace': dace,
                    }))
                    evaluated = symbolic.symstr(evaluate(unparsed, self.constants), cpp_mode=True)
                    infered_type = infer_expr_type(evaluated, self.dtypes)
                    right_value = evaluated

                    if infered_type == dtypes.int64 or infered_type == dtypes.int32:
                        updated = ast.Name(id="pown({},{})".format(left_value, right_value))
                    else:
                        updated = ast.Name(id="pow({},{})".format(left_value, right_value))

                except (TypeError, AttributeError, NameError, KeyError, ValueError, SyntaxError):
                    right_value = cppunparse.cppunparse(self.visit(node.right), expr_semicolon=False)
                    updated = ast.Name(id="pow({},{})".format(left_value, right_value))

                return ast.copy_location(updated, node)

        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id not in self.memlets:
            return self.generic_visit(node)

        memlet, nc, wcr, dtype = self.memlets[node.id]
        defined_type, _ = self.defined_vars.get(node.id)
        updated = node

        if ((defined_type == DefinedType.Stream or defined_type == DefinedType.StreamArray) and memlet.dynamic):
            # Input memlet, we read from channel
            # we should not need mangle here, since we are in a tasklet
            updated = ast.Call(func=ast.Name(id="read_channel_intel"), args=[ast.Name(id=node.id)], keywords=[])
            self.used_streams.append(node.id)
        elif defined_type == DefinedType.Pointer and memlet.dynamic:
            # if this has a variable number of access, it has been declared
            # as a pointer. We need to deference it
            if isinstance(node.id, ast.Subscript):
                slice = self.visit(node.id.slice)
                if isinstance(slice.value, ast.Tuple):
                    subscript = unparse(slice)[1:-1]
                else:
                    subscript = unparse(slice)
                updated = ast.Name(id="{}[{}]".format(node.id, subscript))
            else:  # no subscript
                updated = ast.Name(id="*{}".format(node.id))

        return ast.copy_location(updated, node)

    # Replace default modules (e.g., math) with OpenCL Compliant (e.g. "dace::math::"->"")
    def visit_Attribute(self, node):
        attrname = rname(node)
        module_name = attrname[:attrname.rfind(".")]
        func_name = attrname[attrname.rfind(".") + 1:]
        if module_name in _OPENCL_ALLOWED_MODULES:
            cppmodname = _OPENCL_ALLOWED_MODULES[module_name]
            return ast.copy_location(ast.Name(id=(cppmodname + func_name), ctx=ast.Load), node)
        return self.generic_visit(node)

    def visit_Call(self, node):
        # enforce compliance to OpenCL
        # Type casting:
        if isinstance(node.func, ast.Name):
            if node.func.id in self.ctypes:
                node.func.id = "({})".format(node.func.id)
            elif node.func.id in self.nptypes_to_ctypes:
                # if it as numpy type, convert to C type
                node.func.id = "({})".format(self.nptypes_to_ctypes[node.func.id])
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in self.ctypes:
                node.func.attr = "({})".format(node.func.attr)
            elif node.func.attr in self.nptypes_to_ctypes:
                # if it as numpy type, convert to C type
                node.func.attr = "({})".format(self.nptypes_to_ctypes[node.func.attr])
        elif (isinstance(node.func, (ast.Num, ast.Constant))
              and (node.func.n.to_string() in self.ctypes or node.func.n.to_string() in self.nptypes)):
            new_node = ast.Name(id="({})".format(node.func.n), ctx=ast.Load)
            new_node = ast.copy_location(new_node, node)
            node.func = new_node

        return self.generic_visit(node)
