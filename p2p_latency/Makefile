# === Makefile ===

TARGET_NAME := benchmark_p2p_latency
SRC := benchmark_p2p_latency.cu
BUILD_DIR := build
EXEC := $(BUILD_DIR)/$(TARGET_NAME)

NVCC := nvcc
NVCC_FLAGS := -std=c++11 -O3

all: $(EXEC) init_git

$(EXEC): $(SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

init_git:
	@if [ ! -f .gitignore ]; then echo "# Ignore build output" > .gitignore && echo "build/" >> .gitignore; fi
	@if [ ! -f README.md ]; then echo "# Benchmark P2P Latency" > README.md && echo "CUDA benchmark for GPU P2P latency." >> README.md; fi
	@if [ ! -d .git ]; then git init && echo "Initialized Git repository."; fi

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean init_git
