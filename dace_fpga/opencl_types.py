from dace import dtypes
import numpy

# Translation of types to OpenCL types
_OCL_TYPES = {
    None: "void",
    int: "int",
    float: "float",
    bool: "bool",
    numpy.bool_: "bool",
    numpy.int8: "char",
    numpy.int16: "short",
    numpy.int32: "int",
    numpy.intc: "int",
    numpy.int64: "long",
    numpy.uint8: "uchar",
    numpy.uint16: "ushort",
    numpy.uint32: "uint",
    numpy.uint64: "ulong",
    numpy.uintc: "uint",
    numpy.float32: "float",
    numpy.float64: "double",
    numpy.complex64: "complex float",
    numpy.complex128: "complex double",
}

CTYPES_TO_OCLTYPES = {
    "void": "void",
    "int": "int",
    "float": "float",
    "double": "double",
    "dace::complex64": "complex float",
    "dace::complex128": "complex double",
    "bool": "bool",
    "char": "char",
    "short": "short",
    "int": "int",
    "int64_t": "long",
    "uint8_t": "uchar",
    "uint16_t": "ushort",
    "uint32_t": "uint",
    "dace::uint": "uint",
    "uint64_t": "ulong",
    "dace::float16": "half",
}

# Translation of types to OpenCL vector types
_OCL_VECTOR_TYPES = {
    numpy.int8: "char",
    numpy.uint8: "uchar",
    numpy.int16: "short",
    numpy.uint16: "ushort",
    numpy.int32: "int",
    numpy.intc: "int",
    numpy.uint32: "uint",
    numpy.uintc: "uint",
    numpy.int64: "long",
    numpy.uint64: "ulong",
    numpy.float16: "half",
    numpy.float32: "float",
    numpy.float64: "double",
    numpy.complex64: "complex float",
    numpy.complex128: "complex double",
}

# Lists allowed modules and maps them to OpenCL
OPENCL_ALLOWED_MODULES = {"builtins": "", "dace": "", "math": ""}


def dtype_to_ocl_str(dtype: dtypes.typeclass) -> str:
    """ Converts a DaCe dtype to its OpenCL string representation. """
    if isinstance(dtype, dtypes.vector):
        if dtype.veclen > 1:
            vectype = _OCL_VECTOR_TYPES[dtype.type]
            return f"{vectype}{dtype.veclen}"
        else:
            return dtype_to_ocl_str(dtype.base_type)
    elif isinstance(dtype, dtypes.pointer):
        base_ocl_type = dtype_to_ocl_str(dtype.base_type)
        return f"{base_ocl_type}*"
    elif isinstance(dtype, dtypes.typeclass):
        ocl_type = _OCL_TYPES.get(dtype.type, None)
        if ocl_type is None:
            raise TypeError(f"Type {dtype} not supported in OpenCL.")
        return ocl_type
    else:
        raise TypeError(f"Type {dtype} not supported in OpenCL.")
