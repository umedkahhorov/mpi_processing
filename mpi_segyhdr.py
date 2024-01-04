import segyio
import numpy as np
from mpi4py import MPI
import time
import sys
t1 = time.time()

def mpiHeaders(filename=None,segy_header=None,out=None):
    """
    Function to process a SEGY file using MPI.
    Args:
    - filename (str): Path to the SEGY file
    - segy_header (str): Trace header keys --> class TraceField
    - out (str): Name of the output csv file
    Returns:
    - total_time (float): Total execution time
    # Example usage:
        mpiexec -np 12 python .\mpi_segyhdr.py "C:\\Work\\..\\ST0703MR13_Final_Migrated_FULL_STACK_in_Time_0to35_DEG.segy" CDP_X 'header_cdpX.csv'

    """

    t1 = time.time()
    if segy_header=='CDP_X':
        segy_header=segyio.TraceField.CDP_X
    if segy_header=='CDP_Y':
        segy_header=segyio.TraceField.CDP_Y
    if segy_header=='INLINE_3D':
        segy_header=segyio.TraceField.INLINE_3D
    if segy_header=='CROSSLINE_3D':
        segy_header=segyio.TraceField.CROSSLINE_3D

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # Open the SEGY file
    f = segyio.open(filename, ignore_geometry=True)
    n_traces = f.tracecount

    # Read scale factor of coordinates
    scale = f.attributes(segyio.TraceField.SourceGroupScalar)[0]
    if scale < 0:
        scale = abs(scale)

    # Calculate the number of traces each process should read
    sendbuf = np.linspace(0, n_traces, n_traces, endpoint=False, dtype=int)
    ave, res = divmod(n_traces, nprocs)
    count = np.array([ave + 1 if p < res else ave for p in range(nprocs)])
    displ = np.array([sum(count[:p]) for p in range(nprocs)])

    # Create buffers for scatter and gather
    recvbuf = np.zeros(count[rank], dtype=int)

    # Scatter the displacement indices to each process
    comm.Scatterv([sendbuf, count, displ, MPI.INT], recvbuf, root=0)

    # Each process reads its portion of the file
    start_idx = recvbuf[0]
    end_idx = recvbuf[-1] + 1  # +1 because the end index is exclusive
    header = f.attributes(segy_header)[start_idx:end_idx]
    # Gather the header arrays at the root process
    recvbuf = None
    if rank == 0:
        recvbuf = np.empty((n_traces), dtype=header.dtype)
    comm.Gatherv(sendbuf=header, recvbuf=(recvbuf, count), root=0)
    # At the root process, save the gathered headers to a CSV file
    if rank == 0:
        np.savetxt(out, recvbuf, delimiter=',', header='CDP_X', comments='')
    # Close the SEGY file
    f.close()
    t2 = time.time()
    total_time = t2 - t1
    if rank == 0:
        print(total_time / 60)
    return None

if __name__ == "__main__":
    filename=sys.argv[1]
    out=sys.argv[3]
    segy_header = sys.argv[2]
    mpiHeaders(filename,segy_header,out)
