import argparse

class LoadFromFile(argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            # parse arguments in the file and store them in the target namespace
            parser.parse_args(f.read().split(), namespace)


parser = argparse.ArgumentParser()

parser.add_argument("--param_file",       type=open, action=LoadFromFile,        help="Path to parameter file.")
parser.add_argument("--runlist",          type=str,  required=False,             help="Path to runlist.")
