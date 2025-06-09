"""
MPI-based SEGY File Processor for Parallel Data Normalization

This script processes SEGY files in parallel using MPI (Message Passing Interface).
The master process coordinates work distribution while slave processes handle 
chunk-by-chunk data processing and normalization.

Usage: mpiexec -np <num_processes> python mpi_NormalizeSegy.py
"""

import segyio
import numpy as np
import os
from mpi4py import MPI
import time

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def normalize(data, between=(-1000, 1000)):
    """
    Normalize a NumPy array to the range [a, b].
    """
    a, b = between
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max == data_min:
        return np.full_like(data, a)  # Avoid div by zero
    return (b - a) * ((data - data_min) / (data_max - data_min)) + a


def get_chunks(n_traces: int, chunk_shape: int) -> list[int]:
    """
    Estimate chunk sizes given number of traces of a segyfile and desired chunk shape
    """
    full_chunks = [chunk_shape] * (n_traces // chunk_shape)
    remainder = n_traces % chunk_shape
    if remainder > 0:
        full_chunks.append(remainder)
    return full_chunks


def get_chunk_boundry_indices(chunks: list[int]) -> list[list[int, int]]:
    """
    Calculate inclusive start/end indices for each chunk in each dimension.
    """
    chunk_indices = []
    start = 0
    for chunk_size in chunks:
        end = start + chunk_size - 1
        chunk_indices.append([start, end])
        start = end + 1
    return chunk_indices

# ============================================================================
# MPI MASTER PROCESS
# ============================================================================

def master(segyfilename, chunk_shape):
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    
    print(f"Using {nprocs} processes ({nprocs-1} workers)")
    
    with segyio.open(segyfilename, 'r+', ignore_geometry=True) as f:
        n_traces = f.tracecount
        data_dtype = f.trace.raw[0].dtype
        
        print(f"Total traces: {n_traces}, Data type: {data_dtype}")

        chunks = get_chunks(n_traces, chunk_shape)
        chunk_boundary_indices = get_chunk_boundry_indices(chunks)
        
        # Pre-assign chunks to workers
        worker_assignments = [[] for _ in range(nprocs-1)]
        for i, (start_idx, end_idx) in enumerate(chunk_boundary_indices):
            worker_idx = i % (nprocs - 1)
            worker_assignments[worker_idx].append((start_idx, end_idx))
        
        # Send batches to workers
        for rank in range(1, nprocs):
            if worker_assignments[rank-1]:  # Only send if there's work
                comm.send({
                    "filename": segyfilename,
                    "chunks": worker_assignments[rank-1],
                    "dtype": data_dtype
                }, dest=rank, tag=1)
        
        # Wait for completions
        confirmation_received = 0
        total_chunks = len(chunks)
        
        while confirmation_received < total_chunks:
            _ = comm.recv(source=MPI.ANY_SOURCE, tag=2)
            confirmation_received += 1
            print(f"\rProgress: {confirmation_received}/{total_chunks} chunks", end="", flush=True)
        
        # Terminate workers
        for rank in range(1, nprocs):
            comm.send(None, dest=rank, tag=3)
        
        print("\nMPI job finished successfully!")


# ============================================================================
# MPI SLAVE PROCESSES
# ============================================================================
def slave():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    current_filename = None
    f = None
    
    try:
        while True:
            message = comm.recv(source=0, tag=MPI.ANY_TAG)

            if message is None:
                break
                
            # Check if we need to open a new file
            if f is None or message['filename'] != current_filename:
                if f is not None:
                    f.close()
                current_filename = message['filename']
                f = segyio.open(current_filename, "r+", ignore_geometry=True)

            # Process all chunks in this batch
            for start_idx, end_idx in message['chunks']:
                chunk_data = f.trace.raw[start_idx:end_idx + 1]
                chunk_data = normalize(chunk_data)
                f.trace.raw[start_idx:end_idx + 1] = chunk_data
                comm.send("chunk completed", dest=0, tag=2)
                
    finally:
        if f is not None:
            f.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main execution block - handles MPI process role assignment.
    
    Process roles:
    - Rank 0: Master process (coordinates work)
    - Rank 1+: Slave processes (perform actual processing)
    """
    t1 = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        # Master process configuration
        segyfilename = r"D:\Projects\PCA_workflow\espfam3d_QC2.sgy"
        chunk_shape = 50000
        
        # Coordinate processing
        master(segyfilename, chunk_shape)
    else:
        # Slave process execution
        slave()
    
    t2 = time.time()
    if rank == 0:
        print(f"Total processing time: {t2 - t1:.2f} seconds")