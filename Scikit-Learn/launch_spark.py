__author__ = 'Aubrey'


import sys
from operator import add
from pyspark import SparkContext
from main import run_main

if __name__ == "__main__":
    try:
        sc
    except NameError:
        try:
            sc = SparkContext(appName="CASP-ML")
        except:
            print "sc doesn't exist AND cannot create new SparkContext!"
            print "Did you pass globals() to execfile()?"
    else:
        sc
    num_jobs = 10
    run_main(num_jobs, sc)