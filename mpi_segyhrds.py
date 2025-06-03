"""
MPI-based SEGY header extraction tool.

This script extracts multiple header fields from a SEGY file in parallel using MPI.
Each MPI process handles a portion of the file, improving performance for large files.

Usage:
    mpiexec -np N python mpi_v5.py <segy_file> [header_bites...]

Example:
    mpiexec -np 12 python mpi_v5.py data.segy 181 185 189 193

Default header bites (if none specified):
    181: CDP_X
    185: CDP_Y
    189: INLINE_3D
    193: CROSSLINE_3D
"""

import segyio
import numpy as np
from mpi4py import MPI
import time
import sys
import pandas as pd

def read_chunk(filename, bite, start_idx, end_idx):
    """Read a chunk of SEGY header data.
    Args:
        filename: Path to SEGY file
        bite: Header field byte position
        start_idx: Start trace index
        end_idx: End trace index
    Returns:
        tuple: (header_values, header_name)
    """
    with segyio.open(filename, ignore_geometry=True) as f:
        header_name = [k for k, v in segyio.tracefield.keys.items() if v == bite][0]
        headers = f.attributes(getattr(segyio.TraceField, header_name))[start_idx:end_idx]
        return headers, header_name

def master(filename, bites):
    """Master process: coordinates work distribution and result collection.
    
    Args:
        filename: Path to SEGY file
        bites: List of header byte positions to extract
    
    Returns:
        dict: Extracted header values by field name
    """
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    
    with segyio.open(filename, ignore_geometry=True) as f:
        n_traces = f.tracecount
    
    results = {}
    
    for bite in bites:
        # Send work to all slaves
        for rank in range(1, nprocs):
            comm.send((bite, n_traces), dest=rank, tag=1)
        
        # Collect results
        gathered = np.empty(n_traces, dtype=np.int32)
        ave, res = divmod(n_traces, nprocs - 1)
        
        for rank in range(1, nprocs):
            chunk_data, header_name, start_idx = comm.recv(source=rank, tag=2)
            chunk_size = ave + 1 if rank - 1 < res else ave
            gathered[start_idx:start_idx + chunk_size] = chunk_data
        
        results[header_name] = gathered
    
    # Terminate slaves
    for rank in range(1, nprocs):
        comm.send(None, dest=rank, tag=3)
    
    return results

def slave(filename):
    """Slave process: processes assigned chunks of the SEGY file.
    
    Args:
        filename: Path to SEGY file
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    
    while True:
        message = comm.recv(source=0, tag=MPI.ANY_TAG)
        if message is None:
            break
        
        bite, n_traces = message
        
        # Calculate chunk boundaries
        ave, res = divmod(n_traces, nprocs - 1)
        chunk_size = ave + 1 if rank - 1 < res else ave
        start_idx = (rank - 1) * ave + min(rank - 1, res)
        end_idx = start_idx + chunk_size
        
        # Process chunk and send result
        chunk_data, header_name = read_chunk(filename, bite, start_idx, end_idx)
        comm.send((chunk_data, header_name, start_idx), dest=0, tag=2)

def mpi_headers(filename, bites=[181, 185, 189, 193]):
    """Main function to extract SEGY headers using MPI.
    
    Args:
        filename: Path to SEGY file
        bites: List of header byte positions to extract
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    t1 = time.time()
    
    if rank == 0:
        results = master(filename, bites)
        df = pd.DataFrame(results)
        df.to_csv('headers.csv', index=False)
        print(f"Processed {len(results)} headers in {(time.time()-t1)/60:.2f} minutes -> headers.csv")
    else:
        slave(filename)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
        
    filename = sys.argv[1]
    bites = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else [181, 185, 189, 193]
    mpi_headers(filename, bites)
