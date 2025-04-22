file = "picked_image_5Hz_4ms.sgy"
def segy_change_header(file,bites=[181,185,189,193],scaler_eval=10,scaler_coor=100):
    if scaler_eval:
        bites.append(69)
    if scaler_coor:
        bites.append(71)
    print(bites)
    with segyio.open(file, "r+", ignore_geometry=True) as f:
        f.mmap()
        n_traces = f.tracecount  
        header_keys = segyio.tracefield.keys
        header_names = [k for k, v in header_keys.items() if v in bites]
        segy_header = getattr(segyio.TraceField, header_names[0])
        headers = f.attributes(segy_header)[:]
        cdpx = np.round(np.arange(0,headers.shape[0]*50,50),2)
        cdpy = 0
        line_num = 1001
        traces = np.arange(1,headers.shape[0]+1,1)
        for i in range(headers.shape[0]):
            f.header[i] = {bites[0]: cdpx[i]*scaler_coor}
            f.header[i] = {bites[1]: cdpy}
            f.header[i] = {bites[2]: line_num}
            f.header[i] = {bites[3]: traces[i]}
            f.header[i] = {bites[4]: 10}
            f.header[i] = {bites[5]: 100}

        print(headers.shape)
        #cdp_x = np.arange(0,n_traces*scaler_coor,scaler_coor)
        #segy_header = getattr(segyio.TraceField, header_name[0])
        #headers = f.attributes(segy_header)[:]
        #x = np.arange(0,headers.shape[0]*50,50)
        ##for i in range(headers.shape[0]):
        #    f.header[i] = {segy_header: x[i]}
segy_change_header(file)
