# Create a temporary connection to get cube metadata
conc_temp = conc
select cube
seismic cube dimen
data = np.zeros((seismic cube dimen))
# Clean up temporary connection

# Define chunking parameters
chunk_size = 50
total_inlines = 550
num_chunks = (total_inlines + chunk_size - 1) // chunk_size

@delayed
def read_chunk(i):
    """Read a chunk using a dedicated connection per task"""
    # Create a new connection for each task
    conc_inst = 
    selected cube
    
    i_start = i * chunk_size
    i_end = min(i_start + chunk_size - 1, total_inlines - 1)
    
    # Read the chunk and explicitly get the array
    chunk = selected cube.chunk(
        irange=(i_start, i_end),
        jrange=(0, jmax-1),
        krange=(0, kmax - 1)
    )
    return data from chunking, i_start, i_end

# Create parallel processing tasks
chunk_tasks = [read_chunk(i) for i in range(num_chunks)]

# Compute in parallel and get results
results = dask.compute(*chunk_tasks)

# Assemble results into final array
for arr, start_idx, end_idx in results:
    # Calculate slice length to handle last chunk correctly
    slice_length = end_idx - start_idx + 1
    data[start_idx:start_idx+slice_length, :, :] = arr

# Verify we have non-zero data
print("Non-zero values:", np.count_nonzero(data))
data.shape
