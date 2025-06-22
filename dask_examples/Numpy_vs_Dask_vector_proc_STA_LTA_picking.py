# =============================================================================
# STA/LTA Processing: NumPy vs Dask Performance Comparison
# Input: refdata with shape (2001, 27030) - 2001 samples x 27030 traces
# Goal: Apply STA/LTA to each trace and find peak indices
# =============================================================================

# -----------------------------------------------------------------------------
# Method 1: NumPy Vectorization (Sequential Processing)
# -----------------------------------------------------------------------------
print("Starting NumPy vectorized processing...")
t1 = time.time()

# Define STA/LTA function with fixed parameters (STA=10, LTA=600)
fpicking = lambda trace: classic_sta_lta_py(trace, 10, 600)

# Apply STA/LTA to each column (trace) in refdata
# Result: refpicks shape = (2001, 27030) - STA/LTA values for each trace
refpicks = np.apply_along_axis(fpicking, axis=0, arr=refdata)

# Find index of maximum STA/LTA value in each trace (pick time)
# Alternative: refpicks_idx = np.argmax(refpicks, axis=0)  # More efficient
refpicks_idx = np.apply_along_axis(
    lambda trace: np.where(trace == trace.max())[0][0], 
    axis=0, 
    arr=refpicks
)

t2 = time.time()
print(f"NumPy result - Shape: {refpicks_idx.shape}, Time: {t2-t1:.2f} sec")

# -----------------------------------------------------------------------------
# Method 2: Dask Parallel Processing (Chunked Computation)
# -----------------------------------------------------------------------------
def process_data(refdata):
    """
    Process seismic data using Dask for parallel computation
    
    Args:
        refdata: Input array (samples x traces)
    
    Returns:
        Array of pick indices for each trace
    """
    # Convert to Dask array with column-wise chunking (1000 traces per chunk)
    # This enables parallel processing across chunks
    da_refdata = da.from_array(refdata, chunks=(refdata.shape[0], 1000))
    
    # Apply STA/LTA function to each trace (column)
    # Creates computation graph - no actual computation yet
    refpicks = da.apply_along_axis(
        lambda trace: classic_sta_lta_py(trace, 10, 600),
        axis=0, 
        arr=da_refdata,
        dtype=np.float16,  # Reduced precision to save memory
        shape=(refdata.shape[0],)
    )
    
    # Find argmax indices for each trace
    # Still part of computation graph - builds on previous step  
    refpicks_idx = da.argmax(refpicks, axis=0)
    
    # Execute computation graph and return results
    # Only now does actual computation happen (lazy evaluation)
    return refpicks_idx.compute()

print("\nStarting Dask parallel processing...")
t1 = time.time()
da_refpicks_idx = process_data(refdata)
t2 = time.time()
print(f"Dask result - Shape: {da_refpicks_idx.shape}, Time: {t2-t1:.2f} sec")

# -----------------------------------------------------------------------------
# Performance Summary
# -----------------------------------------------------------------------------
print(f"\nPerformance Comparison:")
print(f"- NumPy (sequential): {refpicks_idx.shape} picks")  
print(f"- Dask (parallel):    {da_refpicks_idx.shape} picks")
print(f"- Results identical:  {np.array_equal(refpicks_idx, da_refpicks_idx)}")
