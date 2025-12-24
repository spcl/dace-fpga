import dace
from dace import dtypes, data, symbolic
from dace import SDFGState
from dace.sdfg import nodes
from dace.sdfg.state import _getdebuginfo, _make_iterators
from dace.transformation.auto.auto_optimize import set_fast_implementations
from dace_fpga.transformations import fpga_auto_opt
from dace_fpga.transformations.fpga_transform_sdfg import FPGATransformSDFG
from dace_fpga import nodes as nd

from typing import Tuple


def apply_fpga_transformations(sdfg, states=None, validate=True, validate_all=False, permissive=False):
    """ Applies a series of transformations on the SDFG for it to
        generate FPGA code.

        :note: This is an in-place operation on the SDFG.
    """
    # Avoiding import loops

    sdfg.apply_transformations(FPGATransformSDFG,
                               validate=validate,
                               validate_all=validate_all,
                               permissive=permissive,
                               states=states)


def auto_optimize_fpga(sdfg: dace.SDFG, device: dtypes.DeviceType) -> dace.SDFG:
    # Apply FPGA Transformations
    apply_fpga_transformations(sdfg)
    fpga_auto_opt.fpga_global_to_local(sdfg)
    fpga_auto_opt.fpga_rr_interleave_containers_to_banks(sdfg)

    # Set all library nodes to expand to fast library calls
    set_fast_implementations(sdfg, device, find_fast_library_fn=fpga_auto_opt.find_fast_library_fpga_override)
    return sdfg


def can_run_state_on_fpga(state: SDFGState):
    """
    Checks if state can be executed on FPGA. Used by FPGATransformState
    and HbmTransform.
    """
    for node, graph in state.all_nodes_recursive():
        # Consume scopes are currently unsupported
        if isinstance(node, (nodes.ConsumeEntry, nodes.ConsumeExit)):
            return False

        # Streams have strict conditions due to code generator limitations
        if (isinstance(node, nodes.AccessNode) and isinstance(graph.sdfg.arrays[node.data], data.Stream)):
            nodedesc = graph.sdfg.arrays[node.data]
            sdict = graph.scope_dict()
            if nodedesc.storage in [
                    dtypes.StorageType.CPU_Heap, dtypes.StorageType.CPU_Pinned, dtypes.StorageType.CPU_ThreadLocal
            ]:
                return False

            # Cannot allocate FIFO from CPU code
            if sdict[node] is None:
                return False

            # Arrays of streams cannot have symbolic size on FPGA
            if symbolic.issymbolic(nodedesc.total_size, graph.sdfg.constants):
                return False

            # Streams cannot be unbounded on FPGA
            if nodedesc.buffer_size < 1:
                return False

    return True


def add_pipeline(state: SDFGState,
                 name,
                 ndrange,
                 init_size=0,
                 init_overlap=False,
                 drain_size=0,
                 drain_overlap=False,
                 additional_iterators=None,
                 schedule=dtypes.ScheduleType.FPGA_Device,
                 debuginfo=None,
                 **kwargs) -> Tuple[nd.PipelineEntry, nd.PipelineExit]:
    """ Adds a pipeline entry and pipeline exit. These are used for FPGA
        kernels to induce distinct behavior between an "initialization"
        phase, a main streaming phase, and a "draining" phase, which require
        a additive number of extra loop iterations (i.e., N*M + I + D),
        where I and D are the number of initialization/drain iterations.
        The code can detect which phase it is in by querying the
        init_condition() and drain_condition() boolean variable.

        :param state:         The SDFG state to which to add the pipeline.
        :param name:          Pipeline label
        :param ndrange:       Mapping between range variable names and
                                their subsets (parsed from strings)
        :param init_size:     Number of iterations of initialization phase.
        :param init_overlap:  Whether the initialization phase overlaps
                                with the "main" streaming phase of the loop.
        :param drain_size:    Number of iterations of draining phase.
        :param drain_overlap: Whether the draining phase overlaps with
                                the "main" streaming phase of the loop.
        :param additional_iterators: A dictionary containing additional
                                iterators that will be created for this scope and that are not
                                automatically managed by the scope code.
                                The dictionary takes the form 'variable_name' -> init_value
        :return: (map_entry, map_exit) node 2-tuple
    """
    debuginfo = _getdebuginfo(debuginfo or state._default_lineinfo)
    pipeline = nd.PipelineScope(name,
                                *_make_iterators(ndrange),
                                init_size=init_size,
                                init_overlap=init_overlap,
                                drain_size=drain_size,
                                drain_overlap=drain_overlap,
                                additional_iterators=additional_iterators or {},
                                schedule=schedule,
                                debuginfo=debuginfo,
                                **kwargs)
    pipeline_entry = nd.PipelineEntry(pipeline)
    pipeline_exit = nd.PipelineExit(pipeline)
    state.add_nodes_from([pipeline_entry, pipeline_exit])
    return pipeline_entry, pipeline_exit
