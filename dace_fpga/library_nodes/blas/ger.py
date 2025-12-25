# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dace.transformation.transformation import ExpandTransformation
import dace

import dace.library

from dace.libraries.blas.nodes.ger import Ger


@dace.library.register_expansion(Ger, "FPGA")
class ExpandGerFpga(ExpandTransformation):
    """
    FPGA-specific expansion of GER with support for vectorization and tiling
    in both dimensions.
    """

    environments = []

    @staticmethod
    def expansion(node, state, sdfg, m=None, n=None, tile_size_x=None, tile_size_y=None):
        """
        :param node: Node to expand.
        :param state: State that the node is in.
        :param sdfg: SDFG that the node is in.
        :param m: Override the number of rows.
        :param n: Override the number of columns.
        :param tile_size_x: Tile size along the M-dimension (rows of A, size of
                            vector x).
        :param tile_size_x: Tile size along the N-dimension (columns of A,
                            size of vector y).
        """

        desc_a_in, desc_x, desc_y = node.validate(sdfg, state)
        desc_a_out = None
        for e in state.out_edges(node):
            if e.src_conn == "_res":
                desc_a_out = sdfg.arrays[e.data.data]

        sdfg = dace.SDFG("ger")
        state = sdfg.add_state("ger")

        desc_a_in = desc_a_in.clone()
        desc_x = desc_x.clone()
        desc_y = desc_y.clone()
        desc_a_out = desc_a_out.clone()
        desc_a_in.transient = False
        desc_a_out.transient = False
        desc_x.transient = False
        desc_y.transient = False
        sdfg.add_datadesc("_A", desc_a_in)
        sdfg.add_datadesc("_res", desc_a_out)
        sdfg.add_datadesc("_x", desc_x)
        sdfg.add_datadesc("_y", desc_y)

        m = m or node.m
        n = n or node.n
        alpha = node.alpha
        veclen = desc_y.dtype.veclen

        size_x = m
        size_y = n / veclen

        num_tiles_x = f"{size_x} / {tile_size_x}"
        num_tiles_y = f"{size_y} / {tile_size_y}"

        y_tile_entry, y_tile_exit = state.add_map("y_tiles", {"ty": f"0:{num_tiles_y}"},
                                                  schedule=dace.ScheduleType.FPGA_Device)

        sdfg.add_array("y_local", (tile_size_y, ), desc_y.dtype, transient=True, storage=dace.StorageType.FPGA_Local)
        y_local = state.add_access("y_local")

        # Load y buffer
        read_y = state.add_read("_y")
        subset = ("0" if isinstance(desc_y, dace.data.Stream) else f"ty*{tile_size_y}+iy")
        read_y_entry, read_y_exit = state.add_map("read_y", {"iy": f"0:{tile_size_y}"},
                                                  schedule=dace.ScheduleType.FPGA_Device)
        read_y_tasklet = state.add_tasklet("read_y", {"y_memory"}, {"y_buffer"}, "y_buffer = y_memory")
        state.add_memlet_path(read_y,
                              y_tile_entry,
                              read_y_entry,
                              read_y_tasklet,
                              dst_conn="y_memory",
                              memlet=dace.Memlet(f"_y[{subset}]"))
        state.add_memlet_path(read_y_tasklet,
                              read_y_exit,
                              y_local,
                              src_conn="y_buffer",
                              memlet=dace.Memlet(f"y_local[iy]"))

        x_tile_entry, x_tile_exit = state.add_map("x_tiles", {"tx": f"0:{num_tiles_x}"},
                                                  schedule=dace.ScheduleType.FPGA_Device)

        x_entry, x_exit = state.add_map("x", {"ix": f"0:{tile_size_x}"}, schedule=dace.ScheduleType.FPGA_Device)

        # Load x
        read_x = state.add_read("_x")
        sdfg.add_array("x_local", (1, ), desc_x.dtype, transient=True, storage=dace.StorageType.FPGA_Local)
        x_local = state.add_access("x_local")
        subset = ("0" if isinstance(desc_x, dace.data.Stream) else f"tx*{tile_size_x} + ix")
        state.add_memlet_path(read_x,
                              y_tile_entry,
                              x_tile_entry,
                              x_entry,
                              x_local,
                              memlet=dace.Memlet(f"_x[{subset}]", other_subset="0"))

        y_entry, y_exit = state.add_map("y", {"iy": f"0:{tile_size_y}"}, schedule=dace.ScheduleType.FPGA_Device)

        # Actual computation
        compute_tasklet = state.add_tasklet("ger", {"a_in", "x_in", "y_in"}, {"a_out"},
                                            f"a_out = {alpha} * x_in * y_in + a_in")

        # Stream in A
        read_a = state.add_read("_A")
        subset_a = ("0" if isinstance(desc_a_in, dace.data.Stream) else f"tx*{tile_size_x} + ix, ty*{tile_size_y} + iy")
        state.add_memlet_path(read_a,
                              y_tile_entry,
                              x_tile_entry,
                              x_entry,
                              y_entry,
                              compute_tasklet,
                              dst_conn="a_in",
                              memlet=dace.Memlet(f"_A[{subset_a}]"))

        # Load buffered x and y
        state.add_memlet_path(x_local, y_entry, compute_tasklet, dst_conn="x_in", memlet=dace.Memlet("x_local[0]"))
        state.add_memlet_path(y_local,
                              x_tile_entry,
                              x_entry,
                              y_entry,
                              compute_tasklet,
                              dst_conn="y_in",
                              memlet=dace.Memlet(f"y_local[iy]"))

        # Store result
        write_a = state.add_write("_res")
        state.add_memlet_path(compute_tasklet,
                              y_exit,
                              x_exit,
                              x_tile_exit,
                              y_tile_exit,
                              write_a,
                              src_conn="a_out",
                              memlet=dace.Memlet(f"_res[{subset_a}]"))

        return sdfg
