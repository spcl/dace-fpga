# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dace.symbolic import equal_valued
import dace.library
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas.nodes.matmul import _get_matmul_operands
import numpy as np

from dace.libraries.blas.nodes.gemm import Gemm
from dace_fpga import api


@dace.library.register_expansion(Gemm, "FPGA1DSystolic")
class ExpandGemmFPGA1DSystolic(ExpandTransformation):
    """
    FPGA based implementation of GEMM, using a 1D systolic array.

    Currently it supports non-transposed input matrices, and non-vectorized input array A.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, num_pes=32, tile_size_m=None):
        """
        GEMM node expansion.

        :param node: Node to expand.
        :param parent_state: State that the node is in.
        :param parent_sdfg: SDFG that the node is in.
        :param num_pes: Number of Processing Elements of the systolic array. By default it is set to 32.

        :param tile_size_m: tiling size considering columns of the input matrix B and resulting matrix C.
                            If B/C are vectorized, the tile size refers to the vectorized container.
                            If set to None, no tiling is used, corresponding to setting the tile size
                            equal to the number of columns of B/C.
        :return:
        """

        ((edge_a, outer_array_a, shape_a, strides_a, _, _), (edge_b, outer_array_b, shape_b, strides_b, _, _),
         (edge_c, outer_array_c, shape_c, strides_c, _, _)) = _get_matmul_operands(node, parent_state, parent_sdfg)

        dtype_a = outer_array_a.dtype.type
        dtype_b = outer_array_b.dtype.type
        dtype_c = dace.dtype_to_typeclass(np.result_type(dtype_a, dtype_b).type)
        shape_c = (shape_a[0], shape_b[1])
        if node.transA:
            raise NotImplementedError("GEMM FPGA expansion not implemented for transposed A.")
        if node.transB:
            raise NotImplementedError("GEMM FPGA expansion not implemented for transposed B.")

        if outer_array_a.veclen > 1:
            raise NotImplementedError("Vectorization not support for input array A.")

        if len(shape_a) != 2 or len(shape_b) != 2 or shape_a[1] != shape_b[0]:
            raise SyntaxError("Matrix sizes must match")

        if outer_array_b.dtype.veclen != outer_array_c.dtype.veclen:
            raise SyntaxError("Vectorization lengths of B and C must match")

        ######################################################################
        # GEMM Parameters and checks

        # Note: the following sizes consider also vectorization
        vec_width = outer_array_b.dtype.veclen
        vec_type = dace.vector(dtype_c, vec_width)
        N, K, M = shape_a[0], shape_a[1], shape_b[1]

        P = num_pes
        T = tile_size_m
        if T is None:
            T = M

        # we will perform sanity check using T and M. But at this stage, we still
        # don't know to what outer symbol they will map.
        # We try to resolve them to constant if they are symbolic, otherwise we skip the checks
        T_constant = dace.symbolic.resolve_symbol_to_constant(T, parent_sdfg)
        K_constant = dace.symbolic.resolve_symbol_to_constant(K, parent_sdfg)

        # Safe delay: this will be used in the compute state, pipeline scope, to insert
        # a delay between accumulation on the same result if needed.
        # Further explanations are provided in the compute state.

        # Note: this is a platform and type dependent parameter.
        if T_constant is not None:
            L = max(16 - T_constant, 0)
        else:
            L = 0

        # This implementation uses a flattened nested loop, that overlaps feeding,
        # computing and draining phases. Each PE is responsible for computing one
        # tile of one row of the final result C. With the current implementation,
        # A PE needs K*T cycles to compute the results and then P*T clock cycles
        # to fully drain them (draining is distributed across PEs).
        # Therefore, in order to guarantee correctness and deadlock free we have
        # to ensure that the number of cycles needed to drain the results is less
        # or equal to the number of cycles needed to compute them.
        # That is PT <= KT.

        if K_constant is not None and P > K_constant:
            raise ValueError(f"GEMM-FPGA: Number of processing elements {P} must be smaller than the K-dimension {K}.")

        ######################################################################
        # Build the SDFG

        new_sdfg = dace.SDFG(node.label + "_sdfg")
        new_state = new_sdfg.add_state("compute")

        # Add data descriptors
        new_sdfg.add_array("_a", shape_a, dtype_a, strides=strides_a, storage=outer_array_a.storage)
        new_sdfg.add_array("_b", shape_b, dtype_b, strides=strides_b, storage=outer_array_b.storage)
        new_sdfg.add_array("_c", shape_c, dtype_c, strides=strides_c, storage=outer_array_c.storage)

        def make_read_A(state):

            # A given row of A must be repeated according to B number of tiles
            # Both N and M can be not a multiple of P and T respectively
            entry, exit = state.add_map("read_A", {
                "n0": f"0:ceiling({N}/{P})",
                "tm": f"0:ceiling({M}/{T})",
                "k": f"0:{K}",
                "n1": f"0:{P}"
            },
                                        schedule=dace.ScheduleType.FPGA_Device)

            # The reader of A reads one element per clock cycle.
            # Note that if P > T+L, then this will be the bottleneck

            mem = state.add_read("_a")
            pipe = state.add_write("A_pipe")

            # Read data from memory: if we are out-of-bound do not read from memory
            # but inject dummy data
            tasklet = state.add_tasklet("read_A", {"from_memory"}, {"to_kernel"}, f"""\
data = from_memory if n0 * {P} + n1 < {N} else 0
to_kernel = data""")

            state.add_memlet_path(mem,
                                  entry,
                                  tasklet,
                                  dst_conn="from_memory",
                                  memlet=dace.Memlet(f"_a[n0 * {P} + n1, k]", dynamic=True, allow_oob=True))
            state.add_memlet_path(tasklet,
                                  exit,
                                  pipe,
                                  src_conn="to_kernel",
                                  memlet=dace.Memlet(f"A_pipe[{P} - n1 - 1]"))

        def make_read_B(state):

            # Also while reading B, we have to consider that T and P could not divide
            # M and N

            entry, exit = state.add_map("read_B", {
                "n": f"0:ceiling({N}/{P})",
                "tm": f"0:ceiling({M}/{T})",
                "k": f"0:{K}",
                "m": f"0:{T}"
            },
                                        schedule=dace.ScheduleType.FPGA_Device)

            # If we are out-of bound, use a dummy value
            new_sdfg.add_array("B_dummy",
                               dtype=vec_type,
                               shape=[1],
                               transient=True,
                               storage=dace.dtypes.StorageType.FPGA_Registers)
            b_dummy = state.add_access("B_dummy")
            init_tasklet = state.add_tasklet("init_dummy_B", {}, {"init_data"}, "init_data = 0")

            state.add_memlet_path(init_tasklet, b_dummy, src_conn="init_data", memlet=dace.Memlet("B_dummy[0]"))

            mem = state.add_read("_b")
            pipe = state.add_write("B_pipe")
            tasklet = state.add_tasklet(
                "read_B", {"from_memory", "dummy_data"}, {"to_kernel"}, f"""\
data = from_memory if tm*{T} + m < {M} else dummy_data
to_kernel = data""")

            state.add_memlet_path(b_dummy, entry, tasklet, dst_conn="dummy_data", memlet=dace.Memlet("B_dummy[0]"))

            state.add_memlet_path(mem,
                                  entry,
                                  tasklet,
                                  dst_conn="from_memory",
                                  memlet=dace.Memlet(f"_b[k, tm*{T} + m]", dynamic=True, allow_oob=True))

            state.add_memlet_path(tasklet, exit, pipe, src_conn="to_kernel", memlet=dace.Memlet("B_pipe[0]"))

        def make_write_C(state):

            # Receives the results and adds it to C

            pipe = state.add_read("C_pipe")
            if not equal_valued(0, node.beta):
                mem_read = state.add_read("_c")
            mem = state.add_write("_c")

            entry_map, exit_map = state.add_map("write_C", {
                "n0": f"0:ceiling({N}/{P})",
                "tm": f"0:ceiling({M}/{T})",
                "n1": f"0:{P}",
                "m": f"0:{T}"
            },
                                                schedule=dace.ScheduleType.FPGA_Device)

            # write in memory by adding C when we copy that to memory

            # deal with out-of-bound accesses

            mul_accumulated = f"{node.alpha} * from_kernel" if not equal_valued(1, node.alpha) else "from_kernel"
            if not equal_valued(0, node.beta):
                if not equal_valued(1, node.beta):
                    add_prev_c = f" + {node.beta} * prev_c"
                else:
                    add_prev_c = " + prev_c"
            else:
                add_prev_c = ""
            tasklet_inputs = {"from_kernel", "prev_c"} if not equal_valued(0, node.beta) else {"from_kernel"}
            tasklet = state.add_tasklet(
                "write_C", tasklet_inputs, {"to_memory"}, f"""\
if tm * {T} + m  < {M}  and  n0 * {P} + n1 < {N} :
    to_memory = {mul_accumulated}{add_prev_c}
""")
            state.add_memlet_path(pipe,
                                  entry_map,
                                  tasklet,
                                  dst_conn="from_kernel",
                                  memlet=dace.Memlet(f"C_pipe[{P}-1]"))
            if not equal_valued(0, node.beta):
                state.add_memlet_path(mem_read,
                                      entry_map,
                                      tasklet,
                                      dst_conn="prev_c",
                                      memlet=dace.Memlet(f"_c[n0 * {P} + n1, tm * {T} + m]",
                                                         dynamic=True,
                                                         allow_oob=True))

            state.add_memlet_path(tasklet,
                                  exit_map,
                                  mem,
                                  src_conn="to_memory",
                                  memlet=dace.Memlet(f"_c[n0 * {P} + n1, tm * {T} + m]", dynamic=True, allow_oob=True))

        def make_compute(sdfg, state):

            A_pipe_in = state.add_read("A_pipe")
            B_pipe_in = state.add_read("B_pipe")
            B_pipe_out = state.add_write("B_pipe")
            C_pipe_in = state.add_read("C_pipe")
            C_pipe_out = state.add_write("C_pipe")

            # The computation is expressed a single, flattened loop, which is generated by the following
            # pipeline scope. Each PE accumulates over T partial results. The drain phase last P*T clock cycles.
            # Draining and compute are overlapped.
            # We are generating the loop by explicitly ignoring loop carried dependencies. Therefore, we have
            # to guarantee that the PE will accumulate on the same partial result only when its value is consolidated.
            # The + L is a safe delay between accumulation between the same partial result.
            # It must be computed by considering T and the latency needed to consolidate a partial result
            # (which is the latency of the add + latency for reading and writing to BRAM).

            entry_pipeline, exit_pipeline = api.add_pipeline(state, "compute_and_drain", {
                "n0": f"0:ceiling({N}/{P})",
                "tm": f"0:ceiling({M}/{T})",
                "k": f"0:{K}",
                "m": f"0:{T} + {L}"
            },
                                                               drain_size=P * T,
                                                               drain_overlap=False,
                                                               additional_iterators={
                                                                   'm_drain': 0,
                                                                   'k_drain': 0
                                                               },
                                                               schedule=dace.ScheduleType.FPGA_Device)

            # Instantiate buffers
            sdfg.add_scalar("A_reg", dtype=dtype_a, transient=True, storage=dace.dtypes.StorageType.FPGA_Registers)
            A_reg = state.add_write("A_reg")
            A_reg_init = state.add_access("A_reg")

            # For C result we are going to use vectorized data type

            # Note: for some of the Sacred Mysteries of Intel OpenCL Compiler (TM), if this buffer is smaller
            # than 24 floats, the II of the pipeline will be 5. Therefore we check this and in case we enlarge it
            buffer_size = T if T_constant is None else max(T_constant, 24)
            sdfg.add_array("C_buffer", [buffer_size],
                           dtype=vec_type,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Local)
            C_buffer_in = state.add_read("C_buffer")
            C_buffer_out = state.add_write("C_buffer")

            # Init data to reset partial results
            new_sdfg.add_array("C_init",
                               dtype=vec_type,
                               shape=[1],
                               transient=True,
                               storage=dace.dtypes.StorageType.FPGA_Registers)
            C_init = state.add_access("C_init")
            C_init_tasklet = state.add_tasklet("C_data_init", {}, {"init_data"}, "init_data = 0")

            state.add_memlet_path(C_init_tasklet, C_init, src_conn="init_data", memlet=dace.Memlet("C_init[0]"))
            state.add_memlet_path(entry_pipeline, C_init_tasklet, memlet=dace.Memlet())

            # Feed A
            # every PE: reads input data, buffer the data assigned to it
            buffer_a_tasklet = state.add_tasklet(
                "buffer_a", {"a_in"}, {
                    "a_reg",
                }, f"""\
if m == 0 and not {entry_pipeline.pipeline.drain_condition()}:
    a_reg = a_in""")

            state.add_memlet_path(A_pipe_in,
                                  entry_pipeline,
                                  buffer_a_tasklet,
                                  memlet=dace.Memlet("A_pipe[p]", dynamic=True),
                                  dst_conn="a_in")
            state.add_memlet_path(buffer_a_tasklet,
                                  A_reg,
                                  memlet=dace.Memlet("A_reg[0]", dynamic=True),
                                  src_conn="a_reg")

            # Feed B
            sdfg.add_array("B_reg",
                           shape=[1],
                           dtype=vec_type,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Local)
            B_reg = state.add_access("B_reg")
            buffer_b_tasklet = state.add_tasklet(
                "buffer_b", {"b_in"}, {"b_reg_out"}, f"""\
if  m>={L} and not {entry_pipeline.pipeline.drain_condition()}:
    b_reg_out = b_in""")

            state.add_memlet_path(B_pipe_in,
                                  entry_pipeline,
                                  buffer_b_tasklet,
                                  memlet=dace.Memlet("B_pipe[p]", dynamic=True),
                                  dst_conn="b_in")
            state.add_memlet_path(buffer_b_tasklet,
                                  B_reg,
                                  memlet=dace.Memlet("B_reg[0]", dynamic=True),
                                  src_conn="b_reg_out")

            # Compute, Forward B, and Drain
            compute_tasklet = state.add_tasklet(
                "compute_and_drain", {"a_in", "b_in", "c_in", "forward_in", "c_init_data"},
                {"b_out", "c_out", "c_pipe_out"}, f"""\
result = c_in
if m >= {L} and not {entry_pipeline.pipeline.drain_condition()}:
    c_prev = c_init_data if k == 0 else c_in
    result =  c_prev + a_in * b_in
    c_out = result
    if p < {P} - 1:
        b_out = b_in
# Drain
# when we have to drain:
# - if we are working on second assigned row or second tile and we have something to drain
# - if k = K-1 and m>=L: each PE has just finished to compute something
# - if we are in the draining phase
# How:
# - if k = K-1 and m>=L: then the PE drains its own result
#-  otherwise, if k_drain<p forward data coming from previous PEs (this could happens also in the drain phase)
if((n0 > 0 or tm > 0)  and k_drain <p and m_drain <{T}) or  (k=={K}-1 and m>= {L}) or ({entry_pipeline.pipeline.drain_condition()} and k_drain < p):
    c_pipe_out = result if (p==0 or (k_drain=={K}-1 and not {entry_pipeline.pipeline.drain_condition()})) else forward_in

# adjust draining iterators
if not {entry_pipeline.pipeline.drain_condition()}:
    if m_drain >= {L} +  {T} -1:
        m_drain = 0
        if k_drain >= {K} - 1:
            k_drain = 0
        else:
            k_drain = k_drain +1
    else:
        m_drain = m_drain + 1
else:
    if m_drain >=  {T} -1:
        m_drain = 0
        if k_drain >= {K} - 1:
            k_drain = 0
        else:
            k_drain = k_drain +1
    else:
        m_drain = m_drain + 1
    """)

            state.add_memlet_path(A_reg, compute_tasklet, dst_conn="a_in", memlet=dace.Memlet("A_reg[0]"))
            state.add_memlet_path(B_reg,
                                  compute_tasklet,
                                  memlet=dace.Memlet("B_reg[0]", dynamic=False),
                                  dst_conn="b_in")
            state.add_memlet_path(C_init, compute_tasklet, memlet=dace.Memlet("C_init[0]"), dst_conn="c_init_data")

            state.add_memlet_path(compute_tasklet,
                                  exit_pipeline,
                                  B_pipe_out,
                                  memlet=dace.Memlet("B_pipe[p + 1]", dynamic=True),
                                  src_conn="b_out")
            state.add_memlet_path(C_buffer_in,
                                  entry_pipeline,
                                  compute_tasklet,
                                  dst_conn="c_in",
                                  memlet=dace.Memlet(f"C_buffer[m-{L}]", allow_oob=True))

            state.add_memlet_path(compute_tasklet,
                                  exit_pipeline,
                                  C_buffer_out,
                                  memlet=dace.Memlet(f"C_buffer[m-{L}]", allow_oob=True, dynamic=True),
                                  src_conn="c_out")

            state.add_memlet_path(C_pipe_in,
                                  entry_pipeline,
                                  compute_tasklet,
                                  memlet=dace.Memlet("C_pipe[p-1]", dynamic=True),
                                  dst_conn="forward_in")
            state.add_memlet_path(compute_tasklet,
                                  exit_pipeline,
                                  C_pipe_out,
                                  memlet=dace.Memlet("C_pipe[p]", dynamic=True),
                                  src_conn="c_pipe_out")

            # Unroll processing elements
            compute_entry, compute_exit = state.add_map("unroll_compute", {"p": "0:{}".format(P)},
                                                        schedule=dace.ScheduleType.FPGA_Device,
                                                        unroll=True)

            # Bring data nodes into scope
            state.add_memlet_path(compute_entry, A_pipe_in, memlet=dace.memlet.Memlet())
            state.add_memlet_path(compute_entry, B_pipe_in, memlet=dace.memlet.Memlet())
            state.add_memlet_path(compute_entry, C_pipe_in, memlet=dace.memlet.Memlet())

            state.add_memlet_path(B_pipe_out, compute_exit, memlet=dace.memlet.Memlet())

            state.add_memlet_path(C_pipe_out, compute_exit, memlet=dace.memlet.Memlet())

            state.add_memlet_path(compute_entry, A_reg_init, memlet=dace.memlet.Memlet())
            state.add_memlet_path(A_reg_init, entry_pipeline, memlet=dace.memlet.Memlet())
            b_init = state.add_access("B_reg")
            state.add_memlet_path(compute_entry, b_init, memlet=dace.Memlet())
            state.add_memlet_path(b_init, entry_pipeline, memlet=dace.Memlet())
            state.add_memlet_path(compute_entry, C_buffer_in, memlet=dace.Memlet())
            state.add_memlet_path(C_buffer_out, compute_exit, memlet=dace.Memlet())

        # build the compute State

        new_sdfg.add_stream("A_pipe",
                            dtype_a,
                            transient=True,
                            shape=(P, ),
                            storage=dace.dtypes.StorageType.FPGA_Local,
                            buffer_size=str(P))
        new_sdfg.add_stream("B_pipe",
                            vec_type,
                            transient=True,
                            shape=(P + 1, ),
                            buffer_size=1,
                            storage=dace.dtypes.StorageType.FPGA_Local)
        new_sdfg.add_stream("C_pipe",
                            vec_type,
                            transient=True,
                            shape=(P + 1, ),
                            buffer_size=T,
                            storage=dace.dtypes.StorageType.FPGA_Local)

        make_read_A(new_state)
        make_read_B(new_state)
        make_compute(new_sdfg, new_state)
        make_write_C(new_state)
        return new_sdfg
