# Seismic Parallel Computation

## Description

We develop parallel computation examples for seismic data processing. Many seismic operations are naturally data-parallel, allowing independent processing of different traces, shot gathers, frequency slices, and more. This enables scalable, high-performance workflows using distributed computing tools.

Our implementation leverages MPI (Message Passing Interface), the industry-standard framework for parallel computing, along with Dask, a modern Python-native library for parallel and distributed processing.

## Dataset

The examples in this repository use the following open seismic datasets:

- **KAHU-3D Dataset (New Zealand)**  
  Source: [SEG Wiki – Kahu-3D](https://wiki.seg.org/wiki/Kahu-3D)  
  Download using:  
  ```bash
  wget http://s3.amazonaws.com/open.source.geoscience/open_data/newzealand/Taranaiki_Basin/KAHU-3D/KAHU-3D-PR3177-FM.3D.Final_Migration.sgy
