"""
FPGA extensions to the SDFG IR.
"""
import dace.serialize
from dace.sdfg.nodes import MapEntry, MapExit, Map
from dace.properties import make_properties, Property, SymbolicProperty, indirect_properties
from typing import Dict
from dace import dtypes

# ------------------------------------------------------------------------------


@dace.serialize.serializable
class PipelineEntry(MapEntry):

    @staticmethod
    def map_type():
        return PipelineScope

    @property
    def pipeline(self):
        return self._map

    @pipeline.setter
    def pipeline(self, val):
        self._map = val

    def new_symbols(self, sdfg, state, symbols) -> Dict[str, dtypes.typeclass]:
        result = super().new_symbols(sdfg, state, symbols)
        for param in self.map.params:
            result[param] = dtypes.int64  # Overwrite params from Map
        for param in self.pipeline.additional_iterators:
            result[param] = dtypes.int64
        result[self.pipeline.iterator_str()] = dtypes.int64
        try:
            result[self.pipeline.init_condition()] = dtypes.bool
        except ValueError:
            pass  # Overlaps
        try:
            result[self.pipeline.drain_condition()] = dtypes.bool
        except ValueError:
            pass  # Overlaps
        return result


@dace.serialize.serializable
class PipelineExit(MapExit):

    @staticmethod
    def map_type():
        return PipelineScope

    @property
    def pipeline(self):
        return self._map

    @pipeline.setter
    def pipeline(self, val):
        self._map = val


@make_properties
class PipelineScope(Map):
    """ This a convenience-subclass of Map that allows easier implementation of
        loop nests (using regular Map indices) that need a constant-sized
        initialization and drain phase (e.g., N*M + c iterations), which would
        otherwise need a flattened one-dimensional map.
    """
    init_size = SymbolicProperty(default=0, desc="Number of initialization iterations.")
    init_overlap = Property(dtype=bool,
                            default=True,
                            desc="Whether to increment regular map indices during initialization.")
    drain_size = SymbolicProperty(default=1, desc="Number of drain iterations.")
    drain_overlap = Property(dtype=bool,
                             default=True,
                             desc="Whether to increment regular map indices during pipeline drain.")
    additional_iterators = Property(dtype=dict, desc="Additional iterators, managed by the user inside the scope.")

    def __init__(self,
                 *args,
                 init_size=0,
                 init_overlap=False,
                 drain_size=0,
                 drain_overlap=False,
                 additional_iterators={},
                 **kwargs):
        super(PipelineScope, self).__init__(*args, **kwargs)
        self.init_size = init_size
        self.init_overlap = init_overlap
        self.drain_size = drain_size
        self.drain_overlap = drain_overlap
        self.additional_iterators = additional_iterators

    def iterator_str(self):
        return "__" + "".join(self.params)

    def loop_bound_str(self):
        from dace.codegen.common import sym2cpp
        bound = 1
        for begin, end, step in self.range:
            bound *= (step + end - begin) // step
        # Add init and drain phases when relevant
        add_str = (" + " + sym2cpp(self.init_size) if self.init_size != 0 and not self.init_overlap else "")
        add_str += (" + " + sym2cpp(self.drain_size) if self.drain_size != 0 and not self.drain_overlap else "")
        return sym2cpp(bound) + add_str

    def init_condition(self):
        """Variable that can be checked to see if pipeline is currently in
           initialization phase."""
        if self.init_size == 0:
            raise ValueError("No init condition exists for " + self.label)
        return self.iterator_str() + "_init"

    def drain_condition(self):
        """Variable that can be checked to see if pipeline is currently in
           draining phase."""
        if self.drain_size == 0:
            raise ValueError("No drain condition exists for " + self.label)
        return self.iterator_str() + "_drain"


PipelineEntry = indirect_properties(PipelineScope, lambda obj: obj.map)(PipelineEntry)
