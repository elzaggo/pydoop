"""
Specialized i/o support for pydpoets.

The current implementation is somewhat minimalistic.
An avro implementation would be more elegant, but probably slower.
"""

import logging
import pydoop.mapreduce.api as api
import pydoop.hdfs as hdfs
from pydoop.utils.serialize import OpaqueInputSplit
import json
import struct
import numpy as np
from .keys import (GRAPH_ARCH_KEY, GRAPH_PATH_KEY)
from .models import model


logging.basicConfig()
LOGGER = logging.getLogger("pydeep.ioformats")
LOGGER.setLevel(logging.LOG)


class SamplesReader(api.RecordReader):
    """FIXME"""
    def __init__(self, context):
        super(SamplesReader, self).__init__(context)
        self.logger = LOGGER.getChild("SamplesReader")
        self.logger.debug('started')
        self.isplit = OpaqueInputSplit().read_buffer(context.input_split)
        self.paths = self.isplit.payload
        self.n_paths = len(self.paths)

    def close(self):
        pass

    def next(self):
        while len(self.paths) > 0:
            yield (1, self.paths.pop())

    def get_progress(self):
        return float(len(self.paths) / self.n_paths)


class BottleneckProjectionsWriter(api.RecordWriter):
    """Write out the bottleneck projection of images as binary records.
    The records have the structure <4 bytes integer><n floats>, where
    the first is the image label code, while the second is the result of
    the bottleneck projection.
    """
    def __init__(self, context):
        super(BottleneckProjectionsWriter, self).__init__(context)
        self.logger = LOGGER.getChild("BottleneckProjectionsWriter")
        jc = context.job_conf
        hdfs_user = jc.get("pydoop.hdfs.user", None)
        self.file = hdfs.open(self.get_output_filename(jc), "wb",
                              user=hdfs_user)

    # FIXME - can we push this method to the parent class?
    def get_output_filename(self, jc):
        part = jc.get_int("mapred.task.partition")
        out_dir = jc["mapred.work.output.dir"]
        self.logger.debug("part: %d", part)
        self.logger.debug("outdir: %s", out_dir)
        return "%s/part-%05d" % (out_dir, part)

    def close(self):
        self.logger.debug("closing open handles")
        self.file.close()
        self.file.fs.close()

    def emit(self, _, value):
        self.file.write(value.tobytes())


class BottleneckProjectionsReader(api.RecordReader):
    """Read binary records representing bottleneck projection of images.
    BROKEN
    """
    def __init__(self, context):
        super(BottleneckProjectionsReader, self).__init__(context)
        self.logger = LOGGER.getChild("BottleneckProjectionsReader")
        self.logger.debug('started')
        self.isplit = OpaqueInputSplit().read_buffer(context.input_split)
        jc = context.job_conf
        m = model[jc[GRAPH_ARCH_KEY]]
        self.bottleneck_tensor_size = m['bottleneck_tensor_size']
        self.record_length = 4 * self.bottleneck_tensor_size
        # now we need to manage creating batches from multiple files
        # remainder = self.isplit.offset % self.record_length
        # self.bytes_read = 0 if remainder == 0 else (self.record_length -
        #                                             remainder)
        # self.file = hdfs.open(self.isplit.filename)
        # self.file.seek(self.isplit.offset + self.bytes_read)

    def close(self):
        self.logger.debug("closing open handles")
        self.file.close()
        self.file.fs.close()

    def next(self):
        if self.bytes_read > self.isplit.length:
            raise StopIteration
        record = self.file.read(self.record_length)
        if not record:
            self.logger.debug("StopIteration on eof")
            raise StopIteration
        if len(record) < self.record_length:  # broken file?
            self.logger.warn("StopIteration on bad rec len %d", len(record))
            raise StopIteration
        self.bytes_read += self.record_length
        key = struct.unpack('>I', record[:4])
        value = np.fromstring(record[4:], dtype=np.float32,
                              count=self.bottleneck_tensor_size)
        return (key, value)

    def get_progress(self):
        return min(float(self.bytes_read) / self.isplit.length, 1.0)
