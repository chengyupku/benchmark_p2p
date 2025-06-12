## Update environment variables

``` bash
# change to your nvshmem install path
export NVSHMEM_ROOT=/home/aiscuser/cy/tilelang/3rdparty/nvshmem
export PATH=$NVSHMEM_ROOT/scripts/build/bin/:$PATH
export LD_LIBRARY_PATH=$NVSHMEM_ROOT/build/src/lib:$LD_LIBRARY_PATH
```

## Compile
``` bash
make -j
```

## Run benchmark
``` bash
nvshmrun -n 2 ./build/benchmark_nvshmem_signal
```