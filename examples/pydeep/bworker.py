"""\ 
Process a stream of images into their bottleneck projection for a given
architecture and a chosen bottleneck tensor.
"""

import pydoop.mapreduce.api as api
import pydoop.mapreduce.pipes as pp

from .ioformats import SamplesReader as Reader
from .ioformats import BottleneckProjectionsWriter as Writer
from .tflow import BottleneckProjector
from .models import model
from .keys import (GRAPH_ARCH_KEY, GRAPH_PATH_KEY)


class Mapper(api.Mapper):

    def __init__(self, context):
        super(Mapper, self).__init__(context)
        jc = context.job_conf
        m = model[jc[GRAPH_ARCH_KEY]]
        m['path'] = jc[GRAPH_PATH_KEY]
        # recover model_root, model
        self.projector = BottleneckProjector(m)

    def map(self, context):
        # we expect key to be, respectively, the image label and value the file
        # path. Here, if needed, we could also generate many derived
        # (distorted) variants of the image
        context.emit(context.key, self.projector(context.value))


factory = pp.Factory(mapper_class=Mapper, record_reader_class=Reader,
                     record_writer_class=Writer)


def __main__():
    pp.run_task(factory, auto_serialize=False)
