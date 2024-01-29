import segyio
import numpy as np
from mpi4py import MPI
import time
import sys
import pandas as pd

def mpiHeaders(filename, bite=189):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    # Open the SEGY file using a context manager
    with segyio.open(filename, ignore_geometry=True) as f:
        n_traces = f.tracecount
        header_keys = segyio.tracefield.keys
        header_name = [k for k, v in header_keys.items() if v == bite]
        segy_header = getattr(segyio.TraceField, header_name[0])
        # Calculate the number of traces each process should read
        ave, res = divmod(n_traces, nprocs)
        counts = [ave + 1 if p < res else ave for p in range(nprocs)]
        displs = [sum(counts[:p]) for p in range(nprocs)]
        # Scatter the displacement indices to each process
        recvbuf = np.zeros(counts[rank], dtype=int)
        comm.Scatterv([np.arange(n_traces, dtype=int), counts, displs, MPI.INT], recvbuf, root=0)
        # Each process reads its portion of the file
        headers = f.attributes(segy_header)[recvbuf[0]:recvbuf[-1] + 1]
        # Gather the header arrays at the root process
        if rank == 0:
            gathered_headers = np.empty(n_traces, dtype=headers.dtype)
        else:
            gathered_headers = None
        comm.Gatherv(sendbuf=headers, recvbuf=(gathered_headers, counts, displs, MPI.INT), root=0)
        # At the root process, save the gathered headers to a binary file
        if rank == 0:
            print (gathered_headers.shape,gathered_headers.dtype)
            gathered_headers.astype('int32').tofile(header_name[0]+'.bin')
            print (f"Reading header {header_name[0],bite} is finished")

def listHeaders(filename,bites=[189,193]):
    t1 = time.time()
    for bite in bites:
        mpiHeaders(filename,bite)
    header_keys = segyio.tracefield.keys
    header_name = [k for k, v in header_keys.items() if v in bites]
    print (f"Reading {header_name,bites} headers is finished")
    t2 = time.time()
    # Measure and print the total time taken
    t2 = time.time()
    total_time = t2 - t1
    print(f"Total time: {total_time / 60} minutes")
    
def csvHeaders(bites=[189,193]):
    header_keys = segyio.tracefield.keys
    header_names = [k for k, v in header_keys.items() if v in bites]
    file_names = [f+'.bin' for f in header_names]
    print (header_names)
    df = pd.DataFrame(columns=header_names)
    for i in range(len(file_names)):
        df.loc[:,header_names[i]] = np.fromfile(file_names[i], dtype=np.int32)
    df.to_csv('headers.csv')

if __name__ == "__main__":
    filename = sys.argv[1]
    listHeaders(filename)
    csvHeaders()
