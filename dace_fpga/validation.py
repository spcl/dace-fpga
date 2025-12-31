"""
Additional validation checks for FPGA SDFGs.
"""
from dace import SDFG, subsets, nodes as nd
from dace_fpga.codegen import fpga
from dace.sdfg.validation import InvalidSDFGError, InvalidSDFGNodeError


def validate_fpga_sdfg(sdfg: SDFG):
    """
    Validates an FPGA SDFG for correctness.

    :param sdfg: The SDFG to validate.
    :raises InvalidSDFGError: If the SDFG is invalid.
    :raises InvalidSDFGNodeError: If a node in the SDFG is invalid.
    """
    # Validate data descriptors
    for name, desc in sdfg._arrays.items():
        # Check for valid bank assignments
        try:
            bank_assignment = fpga.parse_location_bank(desc)
        except ValueError as e:
            raise InvalidSDFGError(str(e), sdfg, None)
        if bank_assignment is not None:
            if bank_assignment[0] == "DDR" or bank_assignment[0] == "HBM":
                try:
                    tmp = subsets.Range.from_string(bank_assignment[1])
                except SyntaxError:
                    raise InvalidSDFGError(
                        "Memory bank specifier must be convertible to subsets.Range"
                        f" for array {name}", sdfg, None)
                try:
                    low, high = fpga.get_multibank_ranges_from_subset(bank_assignment[1], sdfg)
                except ValueError as e:
                    raise InvalidSDFGError(str(e), sdfg, None)
                if (high - low < 1):
                    raise InvalidSDFGError(
                        "Memory bank specifier must at least define one bank to be used"
                        f" for array {name}", sdfg, None)
                if (high - low > 1 and (high - low != desc.shape[0] or len(desc.shape) < 2)):
                    raise InvalidSDFGError(
                        "Arrays that use a multibank access pattern must have the size of the first dimension equal"
                        f" the number of banks and have at least 2 dimensions for array {name}", sdfg, None)

    # Validate nodes
    for node, state in sdfg.all_nodes_recursive():
        # Tasklets may only access 1 HBM bank at a time
        if isinstance(node, nd.Tasklet):
            for attached in state.all_edges(node):
                if attached.data.data in sdfg.arrays:
                    if fpga.is_multibank_array_with_distributed_index(sdfg.arrays[attached.data.data]):
                        low, high, _ = attached.data.subset[0]
                        if (low != high):
                            state_id = state.parent.node_id(state)
                            nid = state.node_id(node)
                            raise InvalidSDFGNodeError(
                                "Tasklets may only be directly connected"
                                " to HBM-memlets accessing only one bank", sdfg, state_id, nid)
