__author__ = 'Aubrey'


import sys
from operator import add
from pyspark import SparkContext
from main import run_main

if __name__ == "__main__":
    try:
        sc
    except NameError:
        sc = SparkContext(appName="CASP-ML")
    else:
        print "Weird error"
    # slices = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    num_children = 10
    map_args = [[x, num_children]
                for x in range(num_children)]
    count = sc.parallelize(map_args, num_children).map(run_main).reduce(add)
    print 'Val:', count