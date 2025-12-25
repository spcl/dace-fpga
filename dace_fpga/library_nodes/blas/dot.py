# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import dace.library
from dace.transformation.transformation import ExpandTransformation
from dace import data as dt, dtypes
from dace.libraries.blas.nodes.dot import Dot


@dace.library.register_expansion(Dot, "FPGA_PartialSums")
class ExpandDotFpgaPartialSums(ExpandTransformation):
    """
    FPGA-expansion of DOT that does NOT assume that native accumulation of the
    data type is possible (e.g., floating point on Xilinx devices or float64
    on Stratix 10).

    To achieve II=1, accumulation is done into multiple partial sums, which are
    reduced at the end of the computation.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, partial_width=8):
        """
        :param node: The node to expand.
        :param parent_state: The state that the node is in.
        :param parent_sdfg: The SDFG that the node is in.
        :param n: Override the vector dimension. If this is not set, the value
                  specified in the node is used.
        :param partial_width: Width of the inner reduction buffer. Must be
                              larger than the latency of addition on the given
                              data type.
        """
        (desc_x, stride_x), (desc_y, stride_y), desc_res, sz = node.validate(parent_sdfg, parent_state)

        n = n or node.n or sz

        sdfg = dace.SDFG("dot")

        stream_state = sdfg.add_state("stream")

        dtype = desc_x.dtype.base_type
        veclen = desc_x.veclen
        vtype = dtypes.vector(dtype, veclen)

        desc_x = desc_x.clone()
        desc_x.transient = False
        desc_y = desc_y.clone()
        desc_y.transient = False
        desc_res = desc_res.clone()
        desc_res.transient = False
        sdfg.add_datadesc("_x", desc_x)
        sdfg.add_datadesc("_y", desc_y)
        sdfg.add_datadesc("_result", desc_res)

        x_read = stream_state.add_read("_x")
        y_read = stream_state.add_read("_y")
        res_write = stream_state.add_write("_result")

        input_x_name = "input_x"
        sdfg.add_array(input_x_name, (1, ), vtype, transient=True, storage=dtypes.StorageType.FPGA_Local)
        input_x_access = stream_state.add_access(input_x_name)

        input_y_name = "input_y"
        sdfg.add_array(input_y_name, (1, ), vtype, transient=True, storage=dtypes.StorageType.FPGA_Local)
        input_y_access = stream_state.add_access(input_y_name)

        entry, exit = stream_state.add_map("stream", {"_i_dot": f"0:{n}/{veclen}"},
                                           schedule=dtypes.ScheduleType.FPGA_Device)

        index_x = "0" if isinstance(desc_x, dt.Stream) else "_i_dot"
        index_y = "0" if isinstance(desc_y, dt.Stream) else "_i_dot"

        stream_state.add_memlet_path(x_read,
                                     entry,
                                     input_x_access,
                                     memlet=dace.Memlet(f"{x_read.data}[{index_x}]", other_subset="0", dynamic=False))
        stream_state.add_memlet_path(y_read,
                                     entry,
                                     input_y_access,
                                     memlet=dace.Memlet(f"{y_read.data}[{index_y}]", other_subset="0", dynamic=False))

        tasklet = stream_state.add_tasklet("multiply", {"__x", "__y"}, {f"_product": vtype}, f"_product = __x * __y")

        stream_state.add_memlet_path(input_x_access, tasklet, dst_conn="__x", memlet=dace.Memlet(f"{input_x_name}[0]"))
        stream_state.add_memlet_path(input_y_access, tasklet, dst_conn="__y", memlet=dace.Memlet(f"{input_y_name}[0]"))

        product_name = "product"
        sdfg.add_array(product_name, (veclen, ), dtype, transient=True, storage=dtypes.StorageType.FPGA_Local)
        product_access = stream_state.add_access(product_name)

        stream_state.add_memlet_path(tasklet,
                                     product_access,
                                     src_conn="_product",
                                     memlet=dace.Memlet(f"{product_name}[0:{veclen}]"))

        collapse_name = "reduce_vector"
        sdfg.add_array(collapse_name, (1, ), dtype, transient=True, storage=dtypes.StorageType.FPGA_Local)
        collapse_read = stream_state.add_read(collapse_name)
        collapse_access = stream_state.add_access(collapse_name)

        unroll_entry, unroll_exit = stream_state.add_map("unroll", {"_j_dot": f"0:{veclen}"},
                                                         unroll=True,
                                                         schedule=dtypes.ScheduleType.FPGA_Device)

        collapse_tasklet = stream_state.add_tasklet(
            "reduce_vector", {"val_in", "reduce_in"}, {"reduce_out"}, """\
prev = reduce_in if _j_dot > 0 else 0
reduce_out = prev + val_in""")

        stream_state.add_memlet_path(collapse_read,
                                     unroll_entry,
                                     collapse_tasklet,
                                     dst_conn="reduce_in",
                                     memlet=dace.Memlet(f"{collapse_name}[0]"))
        stream_state.add_memlet_path(entry, collapse_read, memlet=dace.Memlet())
        stream_state.add_memlet_path(collapse_tasklet,
                                     unroll_exit,
                                     collapse_access,
                                     src_conn="reduce_out",
                                     memlet=dace.Memlet(f"{collapse_name}[0]"))
        stream_state.add_memlet_path(product_access,
                                     unroll_entry,
                                     collapse_tasklet,
                                     dst_conn="val_in",
                                     memlet=dace.Memlet(f"{product_name}[_j_dot]"))

        buffer_name = "partial_sums"
        sdfg.add_array(buffer_name, (partial_width, ), dtype, transient=True, storage=dtypes.StorageType.FPGA_Local)

        # The partial result buffer must be initialized.
        init_tasklet = stream_state.add_tasklet("init_dummy_ps", {}, {"init_data"}, "init_data = 0")
        init_ps_entry, init_ps_exit = stream_state.add_map("init_unroll", {"_j_dot": f"0:{partial_width}"},
                                                           unroll=True,
                                                           schedule=dtypes.ScheduleType.FPGA_Device)
        buffer_read = stream_state.add_access(buffer_name)
        stream_state.add_memlet_path(init_ps_entry, init_tasklet, memlet=dace.Memlet())
        stream_state.add_memlet_path(init_tasklet,
                                     init_ps_exit,
                                     buffer_read,
                                     src_conn="init_data",
                                     memlet=dace.Memlet(f"{buffer_name}[_j_dot]"))

        buffer_write = stream_state.add_write(buffer_name)

        partial_sum_tasklet = stream_state.add_tasklet(
            "partial_sum", {"result_in", "buffer_in"}, {"buffer_out"}, f"""\
prev = buffer_in if _i_dot >= {partial_width} else 0
buffer_out = prev + result_in""")

        stream_state.add_memlet_path(collapse_access,
                                     partial_sum_tasklet,
                                     dst_conn="result_in",
                                     memlet=dace.Memlet(f"{collapse_access.data}[0]"))
        stream_state.add_memlet_path(buffer_read,
                                     entry,
                                     partial_sum_tasklet,
                                     dst_conn=f"buffer_in",
                                     memlet=dace.Memlet(f"{buffer_name}[_i_dot%{partial_width}]"))
        stream_state.add_memlet_path(partial_sum_tasklet,
                                     exit,
                                     buffer_write,
                                     src_conn=f"buffer_out",
                                     memlet=dace.Memlet(f"{buffer_name}[_i_dot%{partial_width}]"))

        reduce_entry, reduce_exit = stream_state.add_map("reduce", {"_i_dot": f"0:{partial_width}"},
                                                         schedule=dtypes.ScheduleType.FPGA_Device,
                                                         unroll=True)

        reduce_tasklet = stream_state.add_tasklet(
            "reduce", {"reduce_in", "result_in"}, {"reduce_out"}, """\
prev = reduce_in if _i_dot > 0 else 0
reduce_out = prev + result_in""")

        stream_state.add_memlet_path(buffer_write,
                                     reduce_entry,
                                     reduce_tasklet,
                                     dst_conn="result_in",
                                     memlet=dace.Memlet(f"{buffer_name}[_i_dot]"))

        reduce_name = "reduce"
        sdfg.add_array(reduce_name, (1, ), dtype, transient=True, storage=dtypes.StorageType.FPGA_Local)
        reduce_read = stream_state.add_read(reduce_name)
        reduce_access = stream_state.add_access(reduce_name)

        stream_state.add_memlet_path(reduce_read,
                                     reduce_entry,
                                     reduce_tasklet,
                                     dst_conn="reduce_in",
                                     memlet=dace.Memlet(f"{reduce_name}[0]"))
        stream_state.add_memlet_path(reduce_tasklet,
                                     reduce_exit,
                                     reduce_access,
                                     src_conn="reduce_out",
                                     memlet=dace.Memlet(f"{reduce_name}[0]"))

        stream_state.add_memlet_path(reduce_access,
                                     res_write,
                                     memlet=dace.Memlet(f"{reduce_name}[0]", other_subset="0"))

        return sdfg


@dace.library.register_expansion(Dot, "FPGA_Accumulate")
class ExpandDotFpgaAccumulate(ExpandTransformation):
    """
    Version of DOT that assumes that native II=1 accumulation of the data type
    is possible on the target architecture (e.g., 32-bit floating point on
    Stratix 10).
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        """
        :param node: The node to expand.
        :param parent_state: The state that the node is in.
        :param parent_sdfg: The SDFG that the node is in.
        :param n: Override the vector dimension. If this is not set, the value
                  specified in the node is used.
        """

        (desc_x, stride_x), (desc_y, stride_y), desc_res, sz = node.validate(parent_sdfg, parent_state)

        n = n or node.n or sz

        sdfg = dace.SDFG("dot")

        state = sdfg.add_state("dot")

        dtype = desc_x.dtype.base_type
        veclen = desc_x.veclen
        vtype = dtypes.vector(dtype, veclen)

        desc_x = desc_x.clone()
        desc_x.transient = False
        desc_y = desc_y.clone()
        desc_y.transient = False
        desc_res = desc_res.clone()
        desc_res.transient = False
        sdfg.add_datadesc("_x", desc_x)
        sdfg.add_datadesc("_y", desc_y)
        sdfg.add_datadesc("_result", desc_res)

        x_read = state.add_read("_x")
        y_read = state.add_read("_y")
        res_write = state.add_write("_result")

        input_x_name = "input_x"
        sdfg.add_array(input_x_name, (1, ), vtype, transient=True, storage=dtypes.StorageType.FPGA_Local)
        input_x_access = state.add_access(input_x_name)

        input_y_name = "input_y"
        sdfg.add_array(input_y_name, (1, ), vtype, transient=True, storage=dtypes.StorageType.FPGA_Local)
        input_y_access = state.add_access(input_y_name)

        entry, exit = state.add_map("stream", {"_i_dot": f"0:{n}/{veclen}"}, schedule=dtypes.ScheduleType.FPGA_Device)

        index_x = "0" if isinstance(desc_x, dt.Stream) else "_i_dot"
        index_y = "0" if isinstance(desc_y, dt.Stream) else "_i_dot"

        state.add_memlet_path(x_read,
                              entry,
                              input_x_access,
                              memlet=dace.Memlet(f"{x_read.data}[{index_x}]", other_subset="0", dynamic=False))
        state.add_memlet_path(y_read,
                              entry,
                              input_y_access,
                              memlet=dace.Memlet(f"{y_read.data}[{index_y}]", other_subset="0", dynamic=False))

        tasklet = state.add_tasklet("multiply", {"__x", "__y"}, {f"_product": vtype}, f"_product = __x * __y")

        state.add_memlet_path(input_x_access, tasklet, dst_conn="__x", memlet=dace.Memlet(f"{input_x_name}[0]"))
        state.add_memlet_path(input_y_access, tasklet, dst_conn="__y", memlet=dace.Memlet(f"{input_y_name}[0]"))

        product_name = "product"
        sdfg.add_array(product_name, (veclen, ), dtype, transient=True, storage=dtypes.StorageType.FPGA_Local)
        product_access = state.add_access(product_name)

        state.add_memlet_path(tasklet,
                              product_access,
                              src_conn="_product",
                              memlet=dace.Memlet(f"{product_name}[0:{veclen}]"))

        collapse_name = "reduce_vector"
        sdfg.add_array(collapse_name, (1, ), dtype, transient=True, storage=dtypes.StorageType.FPGA_Local)
        collapse_read = state.add_read(collapse_name)
        collapse_access = state.add_access(collapse_name)

        unroll_entry, unroll_exit = state.add_map("unroll", {"_j_dot": f"0:{veclen}"},
                                                  unroll=True,
                                                  schedule=dtypes.ScheduleType.FPGA_Device)

        collapse_tasklet = state.add_tasklet("reduce_vector", {"val_in", "reduce_in"}, {"reduce_out"}, """\
prev = reduce_in if _j_dot > 0 else 0
reduce_out = prev + val_in""")

        state.add_memlet_path(collapse_read,
                              unroll_entry,
                              collapse_tasklet,
                              dst_conn="reduce_in",
                              memlet=dace.Memlet(f"{collapse_name}[0]"))
        state.add_memlet_path(entry, collapse_read, memlet=dace.Memlet())
        state.add_memlet_path(collapse_tasklet,
                              unroll_exit,
                              collapse_access,
                              src_conn="reduce_out",
                              memlet=dace.Memlet(f"{collapse_name}[0]"))
        state.add_memlet_path(product_access,
                              unroll_entry,
                              collapse_tasklet,
                              dst_conn="val_in",
                              memlet=dace.Memlet(f"{product_name}[_j_dot]"))

        buffer_name = "reduce_buffer"
        sdfg.add_array(buffer_name, (1, ), dtype, transient=True, storage=dtypes.StorageType.FPGA_Local)
        buffer_read = state.add_read(buffer_name)
        buffer_write = state.add_access(buffer_name)

        zero_tasklet = state.add_tasklet("zero", {}, {"buffer"}, "buffer = 0")
        state.add_memlet_path(zero_tasklet, buffer_read, src_conn="buffer", memlet=dace.Memlet(f"{buffer_name}[0]"))

        reduce_tasklet = state.add_tasklet("sum", {"buffer_in", "result_in"}, {"buffer_out"}, """\
prev = buffer_in if _i_dot > 0 else 0
buffer_out = prev + result_in""")

        state.add_memlet_path(collapse_access,
                              reduce_tasklet,
                              dst_conn="result_in",
                              memlet=dace.Memlet(f"{collapse_access.data}[0]"))
        state.add_memlet_path(buffer_read,
                              entry,
                              reduce_tasklet,
                              dst_conn="buffer_in",
                              memlet=dace.Memlet(f"{buffer_name}[0]"))
        state.add_memlet_path(reduce_tasklet,
                              exit,
                              buffer_write,
                              src_conn=f"buffer_out",
                              memlet=dace.Memlet(f"{buffer_name}[0]"))

        state.add_memlet_path(buffer_write, res_write, memlet=dace.Memlet(f"{buffer_name}[0]", other_subset="0"))

        return sdfg
