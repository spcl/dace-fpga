# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" File defining FPGA expansions for reduction library node. """
import dace
import dace.library
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import graph
from dace.frontend.operations import detect_reduction_type
from dace import dtypes
from dace.transformation import transformation as pm
from dace.symbolic import symstr

from dace.libraries.standard.nodes.reduce import Reduce


@dace.library.register_expansion(Reduce, 'FPGAPartialReduction')
class ExpandReduceFPGAPartialReduction(pm.ExpandTransformation):
    """
        FPGA SDFG Reduce expansion.  This does not assume single-cycle accumulation of the given data type.

        To achieve II=1, reduction is done into multiple partial reduction, which are then
        combined at the end.
    """
    environments = []

    # Reduction type expressions dictionary
    _REDUCTION_TYPE_EXPR = {
        dtypes.ReductionType.Max: 'max(prev, data_in)',
        dtypes.ReductionType.Min: 'min(prev, data_in)',
        dtypes.ReductionType.Sum: 'prev + data_in',
        dtypes.ReductionType.Product: 'prev * data_in',
        dtypes.ReductionType.Sub: 'prev - data_in',
        dtypes.ReductionType.Div: 'prev / data_in'
    }

    @staticmethod
    def expansion(node: 'Reduce', state: SDFGState, sdfg: SDFG, partial_width=16):
        """

        :param node: the node to expand
        :param state: the state in which the node is in
        :param sdfg: the SDFG in which the node is in
        :param partial_width: Width of the inner reduction buffer. Must be
                              larger than the latency of the reduction operation on the given
                              data type
        """
        node.validate(sdfg, state)
        inedge: graph.MultiConnectorEdge = state.in_edges(node)[0]
        outedge: graph.MultiConnectorEdge = state.out_edges(node)[0]
        input_dims = len(inedge.data.subset)
        output_dims = len(outedge.data.subset)
        input_data = sdfg.arrays[inedge.data.data]
        output_data = sdfg.arrays[outedge.data.data]

        # Standardize axes
        axes = node.axes if node.axes else [i for i in range(input_dims)]

        # Create nested SDFG
        nsdfg = SDFG('reduce')

        nsdfg.add_array('_in',
                        inedge.data.subset.size(),
                        input_data.dtype,
                        strides=input_data.strides,
                        storage=input_data.storage)

        nsdfg.add_array('_out',
                        outedge.data.subset.size(),
                        output_data.dtype,
                        strides=output_data.strides,
                        storage=output_data.storage)
        if input_data.dtype.veclen > 1:
            raise NotImplementedError('Vectorization currently not implemented for FPGA expansion of Reduce.')

        nstate = nsdfg.add_state()

        # (If axes != all) Add outer map, which corresponds to the output range
        if len(axes) != input_dims:
            all_axis = False
            # Interleave input and output axes to match input memlet
            ictr, octr = 0, 0
            input_subset = []
            for i in range(input_dims):
                if i in axes:
                    input_subset.append(f'_i{ictr}')
                    ictr += 1
                else:
                    input_subset.append(f'_o{octr}')
                    octr += 1

            output_size = outedge.data.subset.size()

            ome, omx = nstate.add_map('reduce_output', {
                f'_o{i}': f'0:{symstr(sz)}'
                for i, sz in enumerate(outedge.data.subset.size())
            })
            outm_idx = ','.join([f'_o{i}' for i in range(output_dims)])
            outm = dace.Memlet(f'_out[{outm_idx}]')
            inm_idx = ','.join(input_subset)
            inmm = dace.Memlet(f'_in[{inm_idx}]')
        else:
            all_axis = True
            ome, omx = None, None
            outm = dace.Memlet('_out[0]')
            inm_idx = ','.join([f'_i{i}' for i in range(len(axes))])
            inmm = dace.Memlet(f'_in[{inm_idx}]')

        # Add inner map, which corresponds to the range to reduce
        r = nstate.add_read('_in')
        w = nstate.add_read('_out')

        # TODO support vectorization
        buffer_name = 'partial_results'
        nsdfg.add_array(buffer_name, (partial_width, ),
                        input_data.dtype,
                        transient=True,
                        storage=dtypes.StorageType.FPGA_Local)
        buffer = nstate.add_access(buffer_name)
        buffer_write = nstate.add_write(buffer_name)

        # Initialize explicitly partial results, as the inner map could run for a number of iteration < partial_width
        init_me, init_mx = nstate.add_map('partial_results_init', {'i': f'0:{partial_width}'},
                                          schedule=dtypes.ScheduleType.FPGA_Device,
                                          unroll=True)
        init_tasklet = nstate.add_tasklet('init_pr', {}, {'pr_out'}, f'pr_out = {node.identity}')
        nstate.add_memlet_path(init_me, init_tasklet, memlet=dace.Memlet())
        nstate.add_memlet_path(init_tasklet,
                               init_mx,
                               buffer,
                               src_conn='pr_out',
                               memlet=dace.Memlet(f'{buffer_name}[i]'))

        if not all_axis:
            nstate.add_memlet_path(ome, init_me, memlet=dace.Memlet())

        ime, imx = nstate.add_map('reduce_values', {
            f'_i{i}': f'0:{symstr(inedge.data.subset.size()[axis])}'
            for i, axis in enumerate(sorted(axes))
        })

        # Accumulate over partial results
        redtype = detect_reduction_type(node.wcr)
        if redtype not in ExpandReduceFPGAPartialReduction._REDUCTION_TYPE_EXPR:
            raise ValueError('Reduction type not supported for "%s"' % node.wcr)
        else:
            reduction_expr = ExpandReduceFPGAPartialReduction._REDUCTION_TYPE_EXPR[redtype]

        # generate flatten index considering inner map: will be used for indexing into partial results
        ranges_size = ime.range.size()
        inner_index = '+'.join([f'_i{i} * {ranges_size[i + 1]}' for i in range(len(axes) - 1)])
        inner_op = ' + ' if len(axes) > 1 else ''
        inner_index = inner_index + f'{inner_op}_i{(len(axes) - 1)}'
        partial_reduce_tasklet = nstate.add_tasklet('partial_reduce', {'data_in', 'buffer_in'}, {'buffer_out'}, f'''\
prev = buffer_in
buffer_out = {reduction_expr}''')

        if not all_axis:
            # Connect input and partial sums
            nstate.add_memlet_path(r, ome, ime, partial_reduce_tasklet, dst_conn='data_in', memlet=inmm)
        else:
            nstate.add_memlet_path(r, ime, partial_reduce_tasklet, dst_conn='data_in', memlet=inmm)
        nstate.add_memlet_path(buffer,
                               ime,
                               partial_reduce_tasklet,
                               dst_conn='buffer_in',
                               memlet=dace.Memlet(f'{buffer_name}[({inner_index})%{partial_width}]'))
        nstate.add_memlet_path(partial_reduce_tasklet,
                               imx,
                               buffer_write,
                               src_conn='buffer_out',
                               memlet=dace.Memlet(f'{buffer_name}[({inner_index})%{partial_width}]'))

        # Then perform reduction on partial results
        reduce_entry, reduce_exit = nstate.add_map('reduce', {'i': f'0:{partial_width}'},
                                                   schedule=dtypes.ScheduleType.FPGA_Device,
                                                   unroll=True)

        reduce_tasklet = nstate.add_tasklet(
            'reduce', {'reduce_in', 'data_in'}, {'reduce_out'}, f'''\
prev = reduce_in if i > 0 else {node.identity}
reduce_out = {reduction_expr}''')
        nstate.add_memlet_path(buffer_write,
                               reduce_entry,
                               reduce_tasklet,
                               dst_conn='data_in',
                               memlet=dace.Memlet(f'{buffer_name}[i]'))

        reduce_name = 'reduce_result'
        nsdfg.add_array(reduce_name, (1, ), output_data.dtype, transient=True, storage=dtypes.StorageType.FPGA_Local)
        reduce_read = nstate.add_access(reduce_name)
        reduce_access = nstate.add_access(reduce_name)

        if not all_axis:
            nstate.add_memlet_path(ome, reduce_read, memlet=dace.Memlet())

        nstate.add_memlet_path(reduce_read,
                               reduce_entry,
                               reduce_tasklet,
                               dst_conn='reduce_in',
                               memlet=dace.Memlet(f'{reduce_name}[0]'))
        nstate.add_memlet_path(reduce_tasklet,
                               reduce_exit,
                               reduce_access,
                               src_conn='reduce_out',
                               memlet=dace.Memlet(f'{reduce_name}[0]'))

        if not all_axis:
            # Write out the result
            nstate.add_memlet_path(reduce_access, omx, w, memlet=outm)
        else:
            nstate.add_memlet_path(reduce_access, w, memlet=outm)

        # Rename outer connectors and add to node
        inedge._dst_conn = '_in'
        outedge._src_conn = '_out'
        node.add_in_connector('_in')
        node.add_out_connector('_out')
        nsdfg.validate()

        return nsdfg
