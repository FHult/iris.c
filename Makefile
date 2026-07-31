# FLUX.2 klein 4B / iris.c - Pure C Inference Engine
# Makefile

CC = gcc
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math
LDFLAGS = -lm

# Platform detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Source files
SRCS = iris.c iris_kernels.c iris_vae.c iris_transformer_flux.c iris_transformer_zimage.c iris_sample.c iris_image.c jpeg.c iris_safetensors.c iris_qwen3.c iris_qwen3_tokenizer.c terminals.c embcache.c iris_lora.c iris_ip_adapter.c iris_csdmod.c
OBJS = $(SRCS:.c=.o)
CLI_SRCS = iris_cli.c linenoise.c
CLI_OBJS = $(CLI_SRCS:.c=.o)
MAIN = main.c
TARGET = iris
LIB = libiris.a

# Debug build flags
DEBUG_CFLAGS = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address

.PHONY: all clean debug lib install info test test-unit test-parity test-quick test-ci check-untested web-tests pngtest help generic blas mps
.NOTPARALLEL: mps

# Default: show available targets
all: help

help:
	@echo "iris.c / FLUX.2 klein 4B - Build Targets"
	@echo ""
	@echo "Choose a backend:"
	@echo "  make generic  - Pure C, no dependencies (slow)"
	@echo "  make blas     - With BLAS acceleration (~30x faster)"
ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),arm64)
	@echo "  make mps      - Apple Silicon with Metal GPU (fastest)"
endif
endif
	@echo ""
	@echo "Other targets:"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make test     - Run inference test"
	@echo "  make test-parity - Real-weights parity scaffolds (skip w/o weights)"
	@echo "  make pngtest  - Compare PNG load on compressed image"
	@echo "  make info     - Show build configuration"
	@echo "  make lib      - Build static library"
	@echo ""
	@echo "Example: make mps && ./iris -d flux-klein-4b -p \"a cat\" -o cat.png"

# =============================================================================
# Backend: generic (pure C, no BLAS)
# =============================================================================
generic: CFLAGS = $(CFLAGS_BASE) -DGENERIC_BUILD
generic: clean $(TARGET)
	@echo ""
	@echo "Built with GENERIC backend (pure C, no BLAS)"
	@echo "This will be slow but has zero dependencies."

# =============================================================================
# Backend: blas (Accelerate on macOS, OpenBLAS on Linux)
# =============================================================================
ifeq ($(UNAME_S),Darwin)
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK
blas: LDFLAGS += -framework Accelerate
else
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
blas: LDFLAGS += -lopenblas
endif
blas: clean $(TARGET)
	@echo ""
	@echo "Built with BLAS backend (~30x faster than generic)"

# =============================================================================
# Backend: mps (Apple Silicon Metal GPU)
# =============================================================================
ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),arm64)
MPS_CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_METAL -DACCELERATE_NEW_LAPACK -flto=thin
MPS_OBJCFLAGS = $(MPS_CFLAGS) -fobjc-arc
MPS_LDFLAGS = $(LDFLAGS) -framework Accelerate -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Foundation -Wl,-dead_strip

mps: clean mps-build
	@echo ""
	@echo "Built with MPS backend (Metal GPU acceleration)"

mps-build: $(SRCS:.c=.mps.o) $(CLI_SRCS:.c=.mps.o) iris_metal.o main.mps.o
	$(CC) $(MPS_CFLAGS) -o $(TARGET) $^ $(MPS_LDFLAGS)

# Pre-compile Metal shaders for faster startup (requires full Xcode, not just Command Line Tools)
shaders: iris_shaders.metallib
	@echo "Metal shaders pre-compiled"

iris_shaders.metallib: iris_shaders.metal
	@if xcrun --find metal >/dev/null 2>&1; then \
		echo "Compiling Metal shaders..."; \
		xcrun -sdk macosx metal -O2 -ffast-math -c iris_shaders.metal -o iris_shaders.air && \
		xcrun -sdk macosx metallib iris_shaders.air -o iris_shaders.metallib && \
		rm -f iris_shaders.air && \
		echo "Created iris_shaders.metallib"; \
	else \
		echo "Error: Metal compiler requires full Xcode (not just Command Line Tools)"; \
		echo "Install Xcode from the App Store, then run: sudo xcode-select -s /Applications/Xcode.app"; \
		echo "Shaders will be compiled at runtime instead."; \
		exit 1; \
	fi

%.mps.o: %.c iris.h iris_kernels.h
	$(CC) $(MPS_CFLAGS) -c -o $@ $<

# Embed Metal shader source as C array (runtime compilation, no Metal toolchain needed)
iris_shaders_source.h: iris_shaders.metal
	xxd -i $< > $@

iris_metal.o: iris_metal.m iris_metal.h iris_shaders_source.h
	$(CC) $(MPS_OBJCFLAGS) -c -o $@ $<

else
mps:
	@echo "Error: MPS backend requires Apple Silicon (arm64)"
	@exit 1
endif
else
mps:
	@echo "Error: MPS backend requires macOS"
	@exit 1
endif

# =============================================================================
# Build rules
# =============================================================================
$(TARGET): $(OBJS) $(CLI_OBJS) main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

lib: $(LIB)

$(LIB): $(OBJS)
	ar rcs $@ $^

%.o: %.c iris.h iris_kernels.h
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug build
debug: CFLAGS = $(DEBUG_CFLAGS)
debug: LDFLAGS += -fsanitize=address
debug: clean $(TARGET)

# =============================================================================
# Test and utilities
# =============================================================================
test: test-unit web-tests
	@web/venv/bin/python3 run_test.py --iris-binary ./$(TARGET)

test-quick:
	@web/venv/bin/python3 run_test.py --iris-binary ./$(TARGET) --quick

# Fast, dependency-free CI subset of the training tests. Excludes tests that need
# the MLX/Metal GPU runtime, real shard/dataset files, or golden-set quality eval
# (markers registered in train/tests/conftest.py). See BACKLOG GROK-TEST-8.
test-ci:
	@echo "=== Training tests (CI subset: not slow/requires_shards/requires_mps/quality) ==="
	@cd train && .venv/bin/python -m pytest tests/ -x -q \
		-m "not slow and not requires_shards and not requires_mps and not quality"
	@$(MAKE) check-untested --no-print-directory

# Warn (do not fail) on source modules above the line threshold that have no test.
check-untested:
	@train/.venv/bin/python train/scripts/check_untested_modules.py

# Web server API tests (no model or binary required)
web-tests:
	@echo "=== Web server API tests ==="
	@web/venv/bin/python3 -m pytest web/tests/ -v --tb=short
	@echo ""

# Unit tests that run without a model (LoRA math, kernel ops, embcache, JPEG, PNG)
test-unit:
	@echo "=== LoRA unit tests ==="
	@$(CC) -O2 -I. -o /tmp/flux_test_lora debug/test_lora.c iris_lora.c iris_safetensors.c -lm
	@/tmp/flux_test_lora
	@rm -f /tmp/flux_test_lora
	@echo "=== Kernel unit tests (incl. flash attention parity) ==="
	@$(CC) -O2 -I. -o /tmp/flux_test_kernels debug/test_kernels.c iris_kernels.c -lm
	@/tmp/flux_test_kernels
	@rm -f /tmp/flux_test_kernels
	@echo "=== Tokenizer unit tests ==="
	@$(CC) -O2 -I. -o /tmp/flux_test_tokenizer debug/test_tokenizer.c iris_qwen3_tokenizer.c iris_kernels.c -lm
	@/tmp/flux_test_tokenizer || true
	@rm -f /tmp/flux_test_tokenizer
	@echo "=== Embcache unit tests ==="
	@$(CC) -O2 -I. -o /tmp/flux_test_embcache debug/test_embcache.c embcache.c -lm
	@/tmp/flux_test_embcache
	@rm -f /tmp/flux_test_embcache
	@echo "=== VAE white-box tests (encode/decode parity, synthetic weights) ==="
	@$(CC) -O2 -I. -o /tmp/flux_test_vae debug/test_vae.c iris_kernels.c iris_safetensors.c -lm
	@/tmp/flux_test_vae
	@rm -f /tmp/flux_test_vae
	@echo "=== Config-parse golden tests ==="
	@$(CC) -O2 -I. -o /tmp/flux_test_config debug/test_config_parse.c -lm
	@/tmp/flux_test_config
	@rm -f /tmp/flux_test_config
	@echo "=== Z-Image scheduler golden tests (hermetic; links real iris_sample.c) ==="
	@$(CC) -O2 -I. -Wl,-dead_strip -o /tmp/flux_test_zimage_scheduler debug/test_zimage_scheduler.c iris_sample.c -lm
	@/tmp/flux_test_zimage_scheduler
	@rm -f /tmp/flux_test_zimage_scheduler
	@echo "=== Safetensors parser tests ==="
	@$(CC) -O2 -I. -o /tmp/flux_test_st debug/test_safetensors.c iris_safetensors.c -lm
	@/tmp/flux_test_st
	@rm -f /tmp/flux_test_st
	@echo "=== IP-Adapter parity tests (C port vs Python goldens) ==="
	@$(CC) -O2 -I. -o /tmp/flux_test_ip_adapter debug/test_ip_adapter.c iris_ip_adapter.c iris_kernels.c iris_safetensors.c -lm
	@/tmp/flux_test_ip_adapter
	@rm -f /tmp/flux_test_ip_adapter
	@echo "=== CSDModulation parity tests (Learned Style adapter, C vs Python golden) ==="
	@$(CC) -O2 -I. -o /tmp/flux_test_csdmod debug/test_csdmod.c iris_csdmod.c iris_safetensors.c iris_kernels.c -lm
	@/tmp/flux_test_csdmod debug/fixtures/csdmod
	@rm -f /tmp/flux_test_csdmod
	@echo "=== JPEG unit tests ==="
	@$(MAKE) -C jpg_test test --no-print-directory
	@echo "=== PNG tests ==="
	@$(CC) $(CFLAGS_BASE) -I. png_compare.c iris_image.c jpeg.c -lm -o /tmp/flux_png_compare
	@/tmp/flux_png_compare images/cat_uncompressed.png images/cat_compressed.png
	@rm -f /tmp/flux_png_compare
	@echo "PNG TEST PASSED"
	@echo ""
	@echo "All unit tests passed."

# Real-weights / GPU parity SCAFFOLDS for the train<->infer & reference boundaries.
# These CANNOT run in a clean checkout (no Flux/Z-Image weights, no flux_env/mflux).
# Each scaffold SKIPS gracefully (exit 0) with a clear message and the fixture
# contract needed to enable it; NONE of them fake a pass. See each debug/*.c/.py
# header. Wired as a separate on-demand target (NOT part of `make test`), because
# a green run here means "skipped cleanly", not "parity verified".
test-parity:
	@echo "=== Flux base-forward parity (C vs mflux) — scaffold ==="
	@$(CC) -O2 -I. -o /tmp/iris_test_flux_forward_parity debug/test_flux_forward_parity.c -lm
	@/tmp/iris_test_flux_forward_parity; rm -f /tmp/iris_test_flux_forward_parity
	@echo "=== Qwen3 layer-extraction (8/17/26) parity — scaffold ==="
	@$(CC) -O2 -I. -o /tmp/iris_test_qwen3_extract_parity debug/test_qwen3_extract_parity.c -lm
	@/tmp/iris_test_qwen3_extract_parity; rm -f /tmp/iris_test_qwen3_extract_parity
	@echo "=== SigLIP/CSD producer parity (inference producer vs training cache) — scaffold ==="
	@if [ -x train/.venv/bin/python ]; then \
		train/.venv/bin/python debug/test_siglip_csd_parity.py; \
	else \
		python3 debug/test_siglip_csd_parity.py || \
		echo "SKIP siglip/csd-producer-parity: no train/.venv and system python3 lacks deps."; \
	fi
	@echo "=== Z-Image transformer/RoPE parity — scaffold (BLOCKED: no Z-Image model) ==="
	@$(CC) -O2 -I. -o /tmp/iris_test_zimage_transformer_parity debug/test_zimage_transformer_parity.c -lm
	@/tmp/iris_test_zimage_transformer_parity; rm -f /tmp/iris_test_zimage_transformer_parity
	@echo ""
	@echo "Parity scaffolds ran. 'SKIP' = prerequisite absent (expected in a clean"
	@echo "checkout); only 'FAIL' indicates a real parity regression."

pngtest:
	@echo "Running PNG compression compare test..."
	@$(CC) $(CFLAGS_BASE) -I. png_compare.c iris_image.c jpeg.c -lm -o /tmp/flux_png_compare
	@/tmp/flux_png_compare images/cat_uncompressed.png images/cat_compressed.png
	@rm -f /tmp/flux_png_compare
	@echo "PNG TEST PASSED"

install: $(TARGET) $(LIB)
	install -d /usr/local/bin
	install -d /usr/local/lib
	install -d /usr/local/include
	install -m 755 $(TARGET) /usr/local/bin/
	install -m 644 $(LIB) /usr/local/lib/
	install -m 644 iris.h /usr/local/include/
	install -m 644 iris_kernels.h /usr/local/include/

clean:
	rm -f $(OBJS) $(CLI_OBJS) *.mps.o iris_metal.o main.o $(TARGET) $(LIB) convert_bf16_to_f16
	rm -f iris_shaders_source.h

clean-shaders:
	rm -f iris_shaders.metallib iris_shaders.air

info:
	@echo "Platform: $(UNAME_S) $(UNAME_M)"
	@echo "Compiler: $(CC)"
	@echo ""
	@echo "Available backends for this platform:"
	@echo "  generic - Pure C (always available)"
ifeq ($(UNAME_S),Darwin)
	@echo "  blas    - Apple Accelerate"
ifeq ($(UNAME_M),arm64)
	@echo "  mps     - Metal GPU (recommended)"
endif
else
	@echo "  blas    - OpenBLAS (requires libopenblas-dev)"
endif

# =============================================================================
# Dependencies
# =============================================================================
iris.o: iris.c iris.h iris_kernels.h iris_safetensors.h iris_qwen3.h embcache.h iris_lora.h
iris_kernels.o: iris_kernels.c iris_kernels.h
iris_vae.o: iris_vae.c iris.h iris_kernels.h
iris_transformer_flux.o: iris_transformer_flux.c iris.h iris_kernels.h iris_lora.h
iris_lora.o: iris_lora.c iris_lora.h iris_safetensors.h
iris_transformer_zimage.o: iris_transformer_zimage.c iris.h iris_kernels.h iris_safetensors.h
iris_sample.o: iris_sample.c iris.h iris_kernels.h
iris_image.o: iris_image.c iris.h
iris_safetensors.o: iris_safetensors.c iris_safetensors.h
iris_qwen3.o: iris_qwen3.c iris_qwen3.h iris_safetensors.h
iris_qwen3_tokenizer.o: iris_qwen3_tokenizer.c iris_qwen3.h
terminals.o: terminals.c terminals.h iris.h
iris_cli.o: iris_cli.c iris_cli.h iris.h iris_qwen3.h embcache.h linenoise.h terminals.h
linenoise.o: linenoise.c linenoise.h
embcache.o: embcache.c embcache.h
main.o: main.c iris.h iris_kernels.h iris_cli.h terminals.h embcache.h
