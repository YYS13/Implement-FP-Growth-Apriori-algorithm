"""
The entry point of the program

First of all, you don't have to follow the structure of this file,
but please make sure that: (1)we can run your code by running this file,
(2)it can generate the output files, and (3)it can accept the command line arguments.

Please implement the `apriori` and `fp_growth` functions in
the `my_cool_algorithms.py` file (or any module name you prefer).

The `input_data` is a list of lists of integers. Each inner list
is in the form of [transaction_id, transaction_id, item_id].
For example, the following input data contains 2 transactions,
transaction 1 contains 2 items 9192, 31651;
transaction 2 contains 2 items 26134, 57515.

[
    [1, 1, 9192],
    [1, 1, 31651],
    [2, 2, 26134],
    [2, 2, 57515],
]


The `a` is a `Namespace` object that contains the following attributes:
    - dataset: the name of the dataset
    - min_sup: the minimum support
    - min_conf: the minimum confidence
you can access them by `a.dataset`, `a.min_sup`, `a.min_conf`.
"""
from pathlib import Path
from typing import List
import os
import args
import config
from utils import *

# TODO: you have to implement this module by yourself
# import my_cool_algorithms

def main():
    # Parse command line arguments
    a = args.parse_args()
    l.info(f"Arguments: {a}")

    # Load dataset, the below io handles ibm dataset
    data_path = config.IN_DIR / a.dataset
    data_path = str(data_path)
    filename = Path(a.dataset).stem
    # # TODO: you have to implement this function by yourself
    start = time.time()
    fp_growth_out = FPG(data_path, a.min_sup, a.min_conf)
    end = time.time()
    print(f"fpgrowth runtime : {round(end-start, 4)}")
    # Write output to file
    write_file(
        data=fp_growth_out,
        filename=config.OUT_DIR / f"{filename}-fp_growth-{a.min_sup}-{a.min_conf}.csv"
    )
    # # TODO: you have to implement this function by yourself
    start = time.time()
    apriori_out = Apriori(data_path, a.min_sup, a.min_conf)
    end = time.time()
    print(f"apriori runtime : {round(end-start, 4)}")
    # Write output to file
    write_file(
        data=apriori_out,
        filename=config.OUT_DIR / f"{filename}-apriori-{a.min_sup}-{a.min_conf}.csv"
    )


if __name__ == "__main__":
    main()