import argparse
from TransferFunction import TransferFunction
import warnings

# Ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import sys
import git
from copy import deepcopy

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)

sys.path.append(parent_directory)


class COmap2TF():
    def __init__(self):
        self.read_params()


    def read_params(self):
        from l2gen_argparser import parser, LoadFromFile
        
        params, _ = parser.parse_known_args()

        parser_sim = deepcopy(parser)

        parser_sim.add_argument(
        "-P",
        "--params_sim",
        type=open,
        action=LoadFromFile,
        help="Path to parameter file. File should have argparse syntax, and overwrites any value listed here.",
        )

        params_sim = parser_sim.parse_args()
        
        if not params.map_name or not params_sim.map_name:
            raise ValueError(
                "A map file name must be specified in parameter file or terminal."
            )

        self.params = params
        self.params_sim = params_sim
    
    def run(self):
        with open(self.params.runlist) as runlist:
            fieldname = runlist.readlines()[1].split()[0]

        with open(self.params_sim.runlist) as runlist:
            fieldname_sim = runlist.readlines()[1].split()[0]

        if fieldname != fieldname_sim:
            raise ValueError("Cannot compute transfer function from two maps of different fields.")

        simpath = os.path.join(self.params_sim.map_dir, f"{fieldname}_{self.params_sim.map_name}_simcube.h5")
        mappath = os.path.join(self.params_sim.map_dir, f"{fieldname}_{self.params_sim.map_name}.h5")
        noisepath = os.path.join(self.params.map_dir, f"{fieldname}_{self.params.map_name}.h5")

        mappaths = [simpath, mappath, noisepath]

        tf_dir = self.params_sim.transfer_function_dir        
        tf_name = f"{fieldname_sim}_{self.params_sim.transfer_function_name}_tf.h5"
        
        tf_outname = os.path.join(tf_dir, tf_name)

        if self.params_sim.transfer_function_dir is None or self.params_sim.transfer_function_name is None:
            raise ValueError("Please specify a directory and name for the transfer function to be saved.")

        tf = TransferFunction(mappaths=mappaths)
        
        print("\nComputing transfer functions from:")
        print(f"1) {simpath}")
        print(f"2) {mappath}")
        print(f"3) {noisepath}")

        tf.compute_transfer_function()

        tf.write(tf_outname)

        NFEED = 19
        for feed in range(NFEED):
            tf.compute_transfer_function(feed=feed)
            tf_outname = os.path.join(
                tf_dir,
                f"{tf_name}_feed{feed + 1}.h5",
            )
            tf.write(tf_outname)


        
def main():
    comap2tf = COmap2TF()
    comap2tf.run()

if __name__ == "__main__":
    main()