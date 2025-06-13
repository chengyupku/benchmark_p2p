# === Makefile ===

TARGET_NAME := benchmark_p2p_latency
SRC := benchmark_p2p_latency.cu
BUILD_DIR := build
EXEC := $(BUILD_DIR)/$(TARGET_NAME)

NVCC := nvcc
NVCC_FLAGS := -std=c++11 -O3

all: $(EXEC)

$(EXEC): $(SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
