# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
from dace import data as dt
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas.nodes.matmul import _get_matmul_operands

from dace.libraries.blas.nodes.gemv import Gemv


@dace.library.register_expansion(Gemv, "FPGA_Accumulate")
class ExpandGemvFpgaAccumulate(ExpandTransformation):
    """
    This FPGA-oriented expansion iterates over the input matrix A in simple
    row-major order, with optional tiling in both dimensions, where the tiles
    are also traversed in simple row-major order. This means that y is only
    written once, but x is read for every tile in the y-dimension.

    The implementation requires accumulation on the output, and does NOT assume
    native accumulation for the given data type. Instead it uses multiple
    partial sums to ensure that II=1, and only writes the final accumulated
    value once it has been combined from the partial sums.

    This works for both transposed and non-transposed A, but vectorization is
    only implemented for non-transposed A.
    """
    # The above corresponds to gemv_v1 in FBLAS

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, tile_size_x=None, tile_size_y=None, num_partial_sums=16):
        """
        :param node: Node to expand.
        :param parent_state: State that the node is in.
        :param parent_sdfg: SDFG that the node is in.
        :param tile_size_x: Tile size along the dimension of the vector x. If
                            set to None, no tiling is used, corresponding to
                            setting the tile size equal to the full size of x.
        :param tile_size_y: Tile size along the dimension of the vector y. If
                            set to None, no tiling is used, corresponding to
                            setting the tile size equal to the full size of y.
        :param num_partial_sums: The number of distinct registers to accumulate
                                 contributions to the final sum into. Should be
                                 a power of two, and should be higher than the
                                 latency of adding two numbers of the given
                                 data type.
        """

        node.validate(parent_sdfg, parent_state)

        sdfg = dace.SDFG("gemv")
        state = sdfg.add_state("gemv")

        alpha = node.alpha
        beta = node.beta

        # Get input/output data (the method considers also the presence of view nodes)
        ((edge_a, desc_a, _, _, shape_a, strides_a), (edge_x, desc_x, _, _, shape_x, strides_x),
         (edge_y, desc_y, _, _, shape_y, strides_y)) = _get_matmul_operands(node,
                                                                            parent_state,
                                                                            parent_sdfg,
                                                                            name_lhs="_A",
                                                                            name_rhs="_x",
                                                                            name_out="_y")

        # Create local versions of input/output data nodes
        _, desc_a = sdfg.add_array("_A",
                                   shape_a,
                                   desc_a.dtype,
                                   strides=strides_a,
                                   storage=desc_a.storage,
                                   transient=False)
        _, desc_x = sdfg.add_array("_x",
                                   shape_x,
                                   desc_x.dtype,
                                   strides=strides_x,
                                   storage=desc_x.storage,
                                   transient=False)
        _, desc_y_y = sdfg.add_array("_y",
                                     shape_y,
                                     desc_y.dtype,
                                     strides=strides_y,
                                     storage=desc_y.storage,
                                     transient=False)

        if node.transA and desc_a.dtype.veclen > 1:
            raise NotImplementedError("Vectorization not implemented for transposed A.")

        # Create accesses
        read_a = state.add_read("_A")
        read_x = state.add_read("_x")
        if beta != 0:
            read_y = state.add_read("_y")
        write_y = state.add_write("_y")

        size_x = desc_x.shape[0]
        size_y = desc_y.shape[0]
        if tile_size_x is None:
            tile_size_x = size_x
        if tile_size_y is None:
            tile_size_y = size_y
        num_tiles_y = f"{size_y}/{tile_size_y}"
        num_tiles_x = f"{size_x}/{tile_size_x}"

        veclen = desc_a.dtype.veclen

        # Create tile map
        y_tile_entry, y_tile_exit = state.add_map("y_tiles", {"ty": f"0:{num_tiles_y}"},
                                                  schedule=dace.ScheduleType.FPGA_Device)
        x_tile_entry, x_tile_exit = state.add_map("x_tiles", {"tx": f"0:{num_tiles_x}"},
                                                  schedule=dace.ScheduleType.FPGA_Device)

        # Create y map
        y_entry, y_exit = state.add_map("y", {"iy": f"0:{tile_size_y}"}, schedule=dace.ScheduleType.FPGA_Device)

        # Create x map
        x_entry, x_exit = state.add_map("x", {"ix": f"0:{tile_size_x}"}, schedule=dace.ScheduleType.FPGA_Device)

        # Local buffer of x
        sdfg.add_array("x_local", (tile_size_x, ), desc_x.dtype, storage=dace.StorageType.FPGA_Local, transient=True)
        x_local_access = state.add_read("x_local")

        if beta != 0:
            raise NotImplementedError("Not yet implemented.")

        multiply_tasklet = state.add_tasklet("multiply", {"A_in", "x_in"}, {f"product": desc_a.dtype},
                                             "product = A_in * x_in")

        if isinstance(desc_a, dt.Stream):
            subset = "0"
        elif node.transA:
            subset = f"tx * {tile_size_x} + ix, ty * {tile_size_y} + iy"
        else:
            subset = f"ty * {tile_size_y} + iy, tx * {tile_size_x} + ix"
        state.add_memlet_path(read_a,
                              y_tile_entry,
                              x_tile_entry,
                              y_entry,
                              x_entry,
                              multiply_tasklet,
                              dst_conn="A_in",
                              memlet=dace.Memlet(f"_A[{subset}]"))
        read_x_entry, read_x_exit = state.add_map("read_x", {"ix": f"0:{tile_size_x}"},
                                                  schedule=dace.ScheduleType.FPGA_Device)
        subset = ("0" if isinstance(desc_x, dt.Stream) else f"tx*{tile_size_x} + ix")
        read_x_tasklet = state.add_tasklet("read_x", {"x_memory"}, {"x_buffer"}, "x_buffer = x_memory")
        state.add_memlet_path(read_x,
                              y_tile_entry,
                              x_tile_entry,
                              read_x_entry,
                              read_x_tasklet,
                              dst_conn="x_memory",
                              memlet=dace.Memlet(f"_x[{subset}]"))
        state.add_memlet_path(read_x_tasklet,
                              read_x_exit,
                              x_local_access,
                              src_conn="x_buffer",
                              memlet=dace.Memlet(f"x_local[ix]"))
        state.add_memlet_path(x_local_access,
                              y_entry,
                              x_entry,
                              multiply_tasklet,
                              dst_conn="x_in",
                              memlet=dace.Memlet(f"x_local[ix]"))

        # Write to buffer
        sdfg.add_array("product_vector", (1, ), desc_a.dtype, transient=True, storage=dace.StorageType.FPGA_Local)
        product_vector = state.add_access("product_vector")
        state.add_memlet_path(multiply_tasklet,
                              product_vector,
                              src_conn="product",
                              memlet=dace.Memlet(f"product_vector[0]"))

        # Vector length conversion
        sdfg.add_array("product_scalar", (veclen, ),
                       desc_a.dtype.base_type,
                       transient=True,
                       storage=dace.StorageType.FPGA_Local)
        product_scalar = state.add_access("product_scalar")
        state.add_memlet_path(product_vector,
                              product_scalar,
                              memlet=dace.Memlet(f"product_vector[0]", other_subset=f"0:{veclen}"))

        # Now we need to collapse this
        reduce_vector_entry, reduce_vector_exit = state.add_map("reduce_vector", {"u": f"0:{veclen}"},
                                                                schedule=dace.ScheduleType.FPGA_Device,
                                                                unroll=True)

        reduce_vector_tasklet = state.add_tasklet("reduce_vector", {"product_in", "acc_in"}, {"acc_out"},
                                                  "acc_out = product_in + acc_in")
        state.add_memlet_path(product_scalar,
                              reduce_vector_entry,
                              reduce_vector_tasklet,
                              dst_conn="product_in",
                              memlet=dace.Memlet(f"{product_scalar}[u]"))

        # Add accumulation register
        sdfg.add_array("accumulate_product", (1, ),
                       desc_a.dtype.base_type,
                       transient=True,
                       storage=dace.StorageType.FPGA_Local)
        accumulate_product_read = state.add_access("accumulate_product")
        accumulate_product_write = state.add_access("accumulate_product")

        # Initialize it to zero
        init_reduce_vector_tasklet = state.add_tasklet("init_reduce_vector", {}, {"acc_out"}, "acc_out = 0")
        state.add_memlet_path(x_entry, init_reduce_vector_tasklet, memlet=dace.Memlet())
        state.add_memlet_path(init_reduce_vector_tasklet,
                              accumulate_product_read,
                              src_conn="acc_out",
                              memlet=dace.Memlet(f"accumulate_product[0]"))

        # Connect it to the tasklet
        state.add_memlet_path(accumulate_product_read,
                              reduce_vector_entry,
                              reduce_vector_tasklet,
                              dst_conn="acc_in",
                              memlet=dace.Memlet(f"accumulate_product[0]"))
        state.add_memlet_path(reduce_vector_tasklet,
                              reduce_vector_exit,
                              accumulate_product_write,
                              src_conn="acc_out",
                              memlet=dace.Memlet(f"accumulate_product[0]"))

        # Partial sums
        sdfg.add_array("partial_sums", (num_partial_sums, ),
                       desc_y.dtype,
                       storage=dace.StorageType.FPGA_Registers,
                       transient=True)
        partial_sum_read = state.add_read("partial_sums")
        partial_sum_write = state.add_access("partial_sums")

        # Output array
        sdfg.add_array("y_local", (tile_size_y, ), desc_y.dtype, storage=dace.StorageType.FPGA_Local, transient=True)

        # Now we need to actually accumulate into a local register of y
        y_local_read = state.add_read("y_local")
        y_local_write = state.add_read("y_local")
        update_y_tasklet = state.add_tasklet(
            "update_y", {"y_in", "acc_in"}, {"acc_out"}, f"""\
prev = acc_in if ix >= {num_partial_sums} else 0
acc_out = prev + y_in""")
        state.add_memlet_path(accumulate_product_write,
                              update_y_tasklet,
                              dst_conn="y_in",
                              memlet=dace.Memlet(f"accumulate_product[0]"))
        state.add_memlet_path(partial_sum_read,
                              x_entry,
                              update_y_tasklet,
                              dst_conn="acc_in",
                              memlet=dace.Memlet(f"partial_sums[ix%{num_partial_sums}]"))
        state.add_memlet_path(y_tile_entry, y_local_read, memlet=dace.Memlet())
        state.add_memlet_path(y_entry, partial_sum_read, memlet=dace.Memlet())
        state.add_memlet_path(update_y_tasklet,
                              x_exit,
                              partial_sum_write,
                              src_conn="acc_out",
                              memlet=dace.Memlet(f"partial_sums[ix%{num_partial_sums}]"))

        # Reduce the partial sums
        reduce_sums_entry, reduce_sums_exit = state.add_map("reduce_partial_sums", {"u": f"0:{num_partial_sums}"},
                                                            schedule=dace.ScheduleType.FPGA_Device,
                                                            unroll=True)
        reduce_sums_tasklet = state.add_tasklet("reduce_partial_sums", {"sum_in", "val_in"}, {"sum_out"}, """
prev = sum_in if u > 0 else 0
sum_out = prev + val_in""")
        sdfg.add_array("accumulate_sum", (1, ), desc_y.dtype, transient=True, storage=dace.StorageType.FPGA_Local)
        accumulate_sum_read = state.add_access("accumulate_sum")
        accumulate_sum_write = state.add_access("accumulate_sum")
        state.add_memlet_path(y_entry, accumulate_sum_read, memlet=dace.Memlet())
        state.add_memlet_path(accumulate_sum_read,
                              reduce_sums_entry,
                              reduce_sums_tasklet,
                              dst_conn="sum_in",
                              memlet=dace.Memlet("accumulate_sum[0]"))
        state.add_memlet_path(reduce_sums_tasklet,
                              reduce_sums_exit,
                              accumulate_sum_write,
                              src_conn="sum_out",
                              memlet=dace.Memlet("accumulate_sum[0]"))
        state.add_memlet_path(partial_sum_write,
                              reduce_sums_entry,
                              reduce_sums_tasklet,
                              dst_conn="val_in",
                              memlet=dace.Memlet("partial_sums[u]"))

        # Combine with y buffer
        combine_tasklet = state.add_tasklet("combine_y", {"val", "buffer_in"}, {"buffer_out"}, """\
prev = buffer_in if tx > 0 else 0
buffer_out = prev + val""")
        state.add_memlet_path(accumulate_sum_write,
                              combine_tasklet,
                              dst_conn="val",
                              memlet=dace.Memlet("accumulate_sum[0]"))
        state.add_memlet_path(y_local_read,
                              x_tile_entry,
                              y_entry,
                              combine_tasklet,
                              dst_conn="buffer_in",
                              memlet=dace.Memlet("y_local[iy]"))

        state.add_memlet_path(combine_tasklet,
                              y_exit,
                              x_tile_exit,
                              y_local_write,
                              src_conn="buffer_out",
                              memlet=dace.Memlet(f"y_local[iy]"))

        subset = ("0" if isinstance(desc_y, dt.Stream) else f"ty*{tile_size_y} + iy")
        write_y_entry, write_y_exit = state.add_map("write_y", {"iy": f"0:{tile_size_y}"},
                                                    schedule=dace.ScheduleType.FPGA_Device)
        write_y_tasklet = state.add_tasklet("write_y", {"y_buffer"}, {"y_memory"}, "y_memory = y_buffer")
        state.add_memlet_path(y_local_write,
                              write_y_entry,
                              write_y_tasklet,
                              dst_conn="y_buffer",
                              memlet=dace.Memlet(f"y_local[iy]"))
        state.add_memlet_path(write_y_tasklet,
                              write_y_exit,
                              y_tile_exit,
                              write_y,
                              src_conn="y_memory",
                              memlet=dace.Memlet(f"_y[{subset}]"))

        return sdfg


@dace.library.register_expansion(Gemv, "FPGA_TilesByColumn")
class ExpandGemvFpgaTilesByColumn(ExpandTransformation):
    """
    FPGA-oriented expansion that reads the input matrix A in column-major
    order, such that consecutive values are accumulated into different
    registers, avoiding a loop-carried dependency due to accumulation.

    The matrix can optionally be tiled, where the tiles will be traversed in
    row-major order in order to bound the size of the output buffer to the tile
    size. The tile size on y must be larger than the latency of addition for
    the given data type.

    This expansion supports both transposed A and non-transposed A, but
    vectorization is only implemented for transposed A.
    """
    # This corresponds to gemv_v2 in FBLAS

    environments = []

    @staticmethod
    def expansion(node, state, sdfg, tile_size_x=None, tile_size_y=None):
        """
        :param node: Node to expand.
        :param parent_state: State that the node is in.
        :param parent_sdfg: SDFG that the node is in.
        :param tile_size_x: Tile size along the dimension of the vector x. If
                            set to None, no tiling is used, corresponding to
                            setting the tile size equal to the full size of x.
        :param tile_size_y: Tile size along the dimension of the vector y. If
                            set to None, no tiling is used, corresponding to
                            setting the tile size equal to the full size of y.
        """

        node.validate(sdfg, state)

        for e in state.in_edges(node):
            if e.dst_conn == "_A":
                desc_a = sdfg.arrays[e.data.data]
            elif e.dst_conn == "_x":
                desc_x = sdfg.arrays[e.data.data]
        for e in state.out_edges(node):
            if e.src_conn == "_y":
                desc_y = sdfg.arrays[e.data.data]

        sdfg = dace.SDFG("gemv")
        state = sdfg.add_state("gemv")

        alpha = node.alpha
        beta = node.beta

        # Create local versions of input data nodes
        desc_a = desc_a.clone()
        desc_a.transient = False
        sdfg.add_datadesc("_A", desc_a)
        desc_x = desc_x.clone()
        desc_x.transient = False
        sdfg.add_datadesc("_x", desc_x)
        desc_y = desc_y.clone()
        desc_y.transient = False
        sdfg.add_datadesc("_y", desc_y)

        if not node.transA and desc_a.dtype.veclen > 1:
            raise NotImplementedError("Vectorization not implemented for non-transposed A.")

        # Create accesses
        read_a = state.add_read("_A")
        read_x = state.add_read("_x")
        if beta != 0:
            read_y = state.add_read("_y")
        write_y = state.add_write("_y")

        size_x = desc_x.shape[0]
        size_y = desc_y.shape[0]
        if tile_size_x is None:
            tile_size_x = size_x
        if tile_size_y is None:
            tile_size_y = size_y
        num_tiles_y = f"{size_y}/{tile_size_y}"
        num_tiles_x = f"{size_x}/{tile_size_x}"

        # Create y tile map
        y_tile_entry, y_tile_exit = state.add_map("y_tiles", {"ty": f"0:{num_tiles_y}"},
                                                  schedule=dace.ScheduleType.FPGA_Device)

        # Create buffer
        sdfg.add_array("y_local", (tile_size_y, ), desc_y.dtype, storage=dace.StorageType.FPGA_Local, transient=True)
        y_local = state.add_access("y_local")
        y_local_write = state.add_access("y_local")

        # Initialize buffer
        init_entry, init_exit = state.add_map("init", {"iy": f"0:{tile_size_y}"},
                                              schedule=dace.ScheduleType.FPGA_Device)
        if beta != 0:
            if isinstance(desc_y, dt.Stream):
                subset = "0"
            else:
                subset = f"ty*{tile_size_y}+iy"
            init_tasklet = state.add_tasklet("init", {"y_in"}, {"y_out"},
                                             f"y_out = {desc_y.dtype.base_type.ctype}({beta}) * y_in")
            state.add_memlet_path(read_y,
                                  y_tile_entry,
                                  init_entry,
                                  init_tasklet,
                                  dst_conn="y_in",
                                  memlet=dace.Memlet(f"_y[{subset}]"))
            state.add_memlet_path(init_tasklet,
                                  init_exit,
                                  y_local,
                                  src_conn="y_out",
                                  memlet=dace.Memlet(f"y_local[iy]"))
        else:
            state.add_memlet_path(y_tile_entry, init_entry, memlet=dace.Memlet())
            init_tasklet = state.add_tasklet("init", {}, {"y_out"}, "y_out = 0")
            state.add_memlet_path(init_entry, init_tasklet, memlet=dace.Memlet())
            state.add_memlet_path(init_tasklet, init_exit, y_local, src_conn="y_out", memlet=dace.Memlet("y_local[iy]"))

        # Create x tile map
        x_tile_entry, x_tile_exit = state.add_map("x_tiles", {"tx": f"0:{num_tiles_x}"},
                                                  schedule=dace.ScheduleType.FPGA_Device)

        # Create loop over tile size in x
        x_entry, x_exit = state.add_map("x", {"ix": f"0:{tile_size_x}"}, schedule=dace.ScheduleType.FPGA_Device)

        # Buffer a scalar value of x
        sdfg.add_array("x_local", (1, ), desc_x.dtype, transient=True, storage=dace.StorageType.FPGA_Local)
        x_local = state.add_access("x_local")
        subset = "0" if isinstance(desc_x, dt.Stream) else f"tx*{tile_size_x}+ix"
        state.add_memlet_path(read_x, y_tile_entry, x_tile_entry, x_entry, x_local, memlet=dace.Memlet(f"_x[{subset}]"))

        # Create loop over tile size in y
        y_entry, y_exit = state.add_map("y", {"iy": f"0:{tile_size_y}"}, schedule=dace.ScheduleType.FPGA_Device)

        # Do computation
        tasklet = state.add_tasklet("gemv", {"A_in", "x_in", "y_in"}, {"y_out"},
                                    f"y_out = y_in + {alpha} * A_in * x_in")
        state.add_memlet_path(y_local,
                              x_tile_entry,
                              x_entry,
                              y_entry,
                              tasklet,
                              dst_conn="y_in",
                              memlet=dace.Memlet("y_local[iy]"))
        state.add_memlet_path(x_local, y_entry, tasklet, dst_conn="x_in", memlet=dace.Memlet("x_local[0]"))
        state.add_memlet_path(tasklet,
                              y_exit,
                              x_exit,
                              x_tile_exit,
                              y_local_write,
                              src_conn="y_out",
                              memlet=dace.Memlet("y_local[iy]"))
        if isinstance(desc_a, dt.Stream):
            subset = "0"
        elif node.transA:
            subset = f"tx * {tile_size_x} + ix, ty * {tile_size_y} + iy"
        else:
            subset = f"ty * {tile_size_y} + iy, tx * {tile_size_x} + ix"
        state.add_memlet_path(read_a,
                              y_tile_entry,
                              x_tile_entry,
                              x_entry,
                              y_entry,
                              tasklet,
                              dst_conn="A_in",
                              memlet=dace.Memlet(f"_A[{subset}]"))

        # Write out tile of y
        write_y_entry, write_y_exit = state.add_map("write_y", {"iy": f"0:{tile_size_y}"},
                                                    schedule=dace.ScheduleType.FPGA_Device)
        write_y_tasklet = state.add_tasklet("write_y", {"y_in"}, {"y_out"}, "y_out = y_in")
        subset = ("0" if isinstance(desc_y, dt.Stream) else f"ty * {tile_size_y} + iy")
        state.add_memlet_path(y_local_write,
                              write_y_entry,
                              write_y_tasklet,
                              dst_conn="y_in",
                              memlet=dace.Memlet("y_local[iy]"))
        state.add_memlet_path(write_y_tasklet,
                              write_y_exit,
                              y_tile_exit,
                              write_y,
                              src_conn="y_out",
                              memlet=dace.Memlet(f"_y[{subset}]"))

        return sdfg
