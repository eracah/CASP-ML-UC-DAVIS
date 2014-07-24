__author__ = 'Aubrey'
import mpi4py
from mpi4py import MPI
import numpy as np
import sys
import subprocess

subprocess.call([
    "mpiexec",
    "-localonly",
    "4",
    "python",
    "main.py"
])

# max_procs = 5
# print "Spawning Processes"
# comm = MPI.COMM_SELF.Spawn(sys.executable,
#                            args=['main.py'],
#                            maxprocs=max_procs)
# print "Done Spawning Processes"
# N = np.array(100, 'i')
# comm.Bcast([N, MPI.INT], root=MPI.ROOT)
# PI = np.array(0.0, 'd')
# comm.Reduce(None, [PI, MPI.DOUBLE],
#             op=MPI.SUM, root=MPI.ROOT)
# print(PI)
# comm.Disconnect()
