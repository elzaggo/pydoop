from pydoop.app.submit import (
    add_parser_common_arguments,
    add_parser_arguments)
from pydoop.app.submit import PydoopSubmitter
from pydoop import hdfs

from .ioformat import tuple_writer
from .tflow import BottleneckProjector, save_graph
from .models import model
from .bworker import GRAPH_PATH_KEY, GRAPH_ARCH_KEY

import sys
import argparse
import random
import os

# Pre-assembled options
DEFAULT_NUM_MAPS = 10
DEFAULT_ARCHITECTURE = 'inception_v3'
NUM_MAPS_KEY = 'mapreduce.job.maps'


def get_categories_data(path):
    categories = {}
    for x in os.scandir(path):
        if x.is_dir():
            categories[x.name] = [os.path.join(x.path, f)
                                  for f in os.listdir(x.path)
                                  if (f.split('.')[-1].lower()
                                      in ['jpg', 'jpeg'])]
    # FIXME Balance between categories
    return categories


def random_shuffle(categories):
    data = [(l, f)
            for l in categories for f in categories[l]]
    random.shuffle(data)
    return data


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-maps', metavar='INT', type=int,
        default=DEFAULT_NUM_MAPS,
        help=''
    )
    parser.add_argument(
        '--architecture', metavar='STR', type=str,
        default=DEFAULT_ARCHITECTURE,
        help=''
    )
    add_parser_common_arguments(parser)
    add_parser_arguments(parser)
    return parser


def add_D_arg(args, arg_name, arg_key):
    val = str(getattr(args, arg_name))
    if args.D is None:
        args.D = [[arg_key, val]]
    elif not any(map(lambda _: _[0] == arg_key,
                 args.D)):
        args.D.append([arg_key, val])


def create_directory_and_write_data_file(input_dir, data):
    # input_dir should not pre-exist
    hdfs.mkdir(input_dir)
    # use a random tmp name
    tuple_writer(data, os.path.join(input_dir, 'samples'))


def prepare_and_save_graph(model, graph_path):
    graph = BottleneckProjector.create_graph(model)
    save_graph(graph, graph_path)


def main(argv=None):
    parser = make_parser()
    args, unknown_args = parser.parse_known_args(argv)
    args.job_name = 'bworker'
    args.module = 'bworker'
    args.upload_file_to_cache = [
        'models.py', 'bworker.py', 'tflow.py', 'ioformats.py']
    args.python_zip = ['']
    args.do_not_use_java_record_reader = True
    args.do_not_use_java_record_writer = True

    # FIXME change to an unique file name
    graph_path = os.path.join(args.input, '__graph.pb')

    data = random_shuffle(get_categories_data(args.data_dir))
    create_directory_and_write_data_file(args.input, data)
    prepare_and_save_graph(model[args.architecture], graph_path)

    add_D_arg(args, 'num_maps', NUM_MAPS_KEY)
    add_D_arg(args, 'architecture', GRAPH_ARCH_KEY)
    args.D.append([GRAPH_PATH_KEY, graph_path])

    args.num_reducers = 0
    submitter = PydoopSubmitter()
    submitter.set_args(args, [] if unknown_args is None else unknown_args)

    submitter.run()


if __name__ == "__main__":
    main(sys.argv)
