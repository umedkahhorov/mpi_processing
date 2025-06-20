# Seismic Parallel Computation

## Description

We develop parallel computation examples for seismic data processing. Many seismic operations are naturally data-parallel, allowing independent processing of different traces, shot gathers, frequency slices, and more. This enables scalable, high-performance workflows using distributed computing tools.

As we primarily use Python, our approach focuses on Python-compatible parallel processing tools. The two main frameworks used are MPI (Message Passing Interface), a widely adopted standard for distributed computing, and Dask, a modern library for parallel and scalable workflows in Python.

## Dataset

The examples in this repository use the following open seismic datasets:

- **KAHU-3D Dataset (New Zealand)**  
  Source: [SEG Wiki – Kahu-3D](https://wiki.seg.org/wiki/Kahu-3D)  
  Download using:  
  ```bash
  wget http://s3.amazonaws.com/open.source.geoscience/open_data/newzealand/Taranaiki_Basin/KAHU-3D/KAHU-3D-PR3177-FM.3D.Final_Migration.sgy

- **Volve field data set**
  Source: [https://www.equinor.com/energy/volve-data-sharing]

## Use

