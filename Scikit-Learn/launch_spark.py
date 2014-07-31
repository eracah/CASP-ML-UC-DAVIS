__author__ = 'Aubrey'


import sys
from operator import add
from pyspark import SparkContext
from pyspark import SparkConf
from main import run_main

if __name__ == "__main__":
    try:
        sc
    except NameError:
        try:
            spark_conf = SparkConf()
            spark_conf.set("spark.executor.memory", "8g")
            spark_conf.set("spark.eventLog.enable", "false")
            spark_conf.set("spark.logConf", "true")
            spark_conf.set("spark.default.parallelism", "2")
            sc = SparkContext(conf=spark_conf, appName="CASP-ML")
        except:
            print "sc doesn't exist AND cannot create new SparkContext!"
            print "Did you pass globals() to execfile()?"
    else:
        sc
    num_jobs = 10
    run_main(num_jobs, sc)