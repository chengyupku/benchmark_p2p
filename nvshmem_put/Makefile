NVCC        := nvcc
TARGET      := benchmark_nvshmem_put
SRC         := $(TARGET).cu
BUILD_DIR   := build
BIN         := $(BUILD_DIR)/$(TARGET)

CUDA_ARCH   := compute_90
CUDA_CODE   := sm_90

NVSHMEM_ROOT := $(NVSHMEM_ROOT)/build/src

INCLUDES    := -I$(NVSHMEM_ROOT)/include
LIBDIRS     := -L$(NVSHMEM_ROOT)/lib
LIBS        := -lnvshmem_host -lnvshmem_device

NVCC_FLAGS  := -rdc=true -Xcompiler -fPIC \
               -gencode arch=$(CUDA_ARCH),code=$(CUDA_CODE) \
               $(INCLUDES) $(LIBDIRS) $(LIBS)

.PHONY: all clean

all: $(BIN)

$(BUILD_DIR):
	mkdir -p $@

$(BIN): $(SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -rf $(BUILD_DIR)
