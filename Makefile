# FLUX.2 klein 4B - Pure C Inference Engine
# Makefile

CC = gcc
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math
LDFLAGS = -lm

# Platform detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Source files
SRCS = flux.c flux_kernels.c flux_tokenizer.c flux_vae.c flux_transformer.c flux_zimage_transformer.c flux_sample.c flux_image.c jpeg.c flux_safetensors.c flux_qwen3.c flux_qwen3_tokenizer.c terminals.c embcache.c flux_lora.c
OBJS = $(SRCS:.c=.o)
CLI_SRCS = flux_cli.c linenoise.c
CLI_OBJS = $(CLI_SRCS:.c=.o)
MAIN = main.c
TARGET = flux
LIB = libflux.a

# Debug build flags
DEBUG_CFLAGS = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address

.PHONY: all clean debug lib install info test test-unit test-quick web-tests pngtest help generic blas mps
.NOTPARALLEL: mps

# Default: show available targets
all: help

help:
	@echo "FLUX.2 klein 4B - Build Targets"
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
	@echo "  make pngtest  - Compare PNG load on compressed image"
	@echo "  make info     - Show build configuration"
	@echo "  make lib      - Build static library"
	@echo ""
	@echo "Example: make mps && ./flux -d flux-klein-4b -p \"a cat\" -o cat.png"

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

mps-build: $(SRCS:.c=.mps.o) $(CLI_SRCS:.c=.mps.o) flux_metal.o main.mps.o
	$(CC) $(MPS_CFLAGS) -o $(TARGET) $^ $(MPS_LDFLAGS)

# Pre-compile Metal shaders for faster startup (requires full Xcode, not just Command Line Tools)
shaders: flux_shaders.metallib
	@echo "Metal shaders pre-compiled"

flux_shaders.metallib: flux_shaders.metal
	@if xcrun --find metal >/dev/null 2>&1; then \
		echo "Compiling Metal shaders..."; \
		xcrun -sdk macosx metal -O2 -ffast-math -c flux_shaders.metal -o flux_shaders.air && \
		xcrun -sdk macosx metallib flux_shaders.air -o flux_shaders.metallib && \
		rm -f flux_shaders.air && \
		echo "Created flux_shaders.metallib"; \
	else \
		echo "Error: Metal compiler requires full Xcode (not just Command Line Tools)"; \
		echo "Install Xcode from the App Store, then run: sudo xcode-select -s /Applications/Xcode.app"; \
		echo "Shaders will be compiled at runtime instead."; \
		exit 1; \
	fi

%.mps.o: %.c flux.h flux_kernels.h
	$(CC) $(MPS_CFLAGS) -c -o $@ $<

# Embed Metal shader source as C array (runtime compilation, no Metal toolchain needed)
flux_shaders_source.h: flux_shaders.metal
	xxd -i $< > $@

flux_metal.o: flux_metal.m flux_metal.h flux_shaders_source.h
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

%.o: %.c flux.h flux_kernels.h
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug build
debug: CFLAGS = $(DEBUG_CFLAGS)
debug: LDFLAGS += -fsanitize=address
debug: clean $(TARGET)

# =============================================================================
# Test and utilities
# =============================================================================
test: test-unit web-tests
	@python3 run_test.py --flux-binary ./$(TARGET)

test-quick:
	@python3 run_test.py --flux-binary ./$(TARGET) --quick

# Web server API tests (no model or binary required)
web-tests:
	@echo "=== Web server API tests ==="
	@python3 -m pytest web/tests/ -v --tb=short
	@echo ""

# Unit tests that run without a model (LoRA math, JPEG decoder, PNG roundtrip)
test-unit:
	@echo "=== LoRA unit tests ==="
	@$(CC) -O2 -I. -o /tmp/flux_test_lora debug/test_lora.c flux_lora.c flux_safetensors.c -lm
	@/tmp/flux_test_lora
	@rm -f /tmp/flux_test_lora
	@echo "=== JPEG unit tests ==="
	@$(MAKE) -C jpg_test test --no-print-directory
	@echo "=== PNG tests ==="
	@$(CC) $(CFLAGS_BASE) -I. png_compare.c flux_image.c jpeg.c -lm -o /tmp/flux_png_compare
	@/tmp/flux_png_compare images/cat_uncompressed.png images/cat_compressed.png
	@rm -f /tmp/flux_png_compare
	@echo "PNG TEST PASSED"
	@echo ""
	@echo "All unit tests passed."

pngtest:
	@echo "Running PNG compression compare test..."
	@$(CC) $(CFLAGS_BASE) -I. png_compare.c flux_image.c jpeg.c -lm -o /tmp/flux_png_compare
	@/tmp/flux_png_compare images/cat_uncompressed.png images/cat_compressed.png
	@rm -f /tmp/flux_png_compare
	@echo "PNG TEST PASSED"

install: $(TARGET) $(LIB)
	install -d /usr/local/bin
	install -d /usr/local/lib
	install -d /usr/local/include
	install -m 755 $(TARGET) /usr/local/bin/
	install -m 644 $(LIB) /usr/local/lib/
	install -m 644 flux.h /usr/local/include/
	install -m 644 flux_kernels.h /usr/local/include/

clean:
	rm -f $(OBJS) $(CLI_OBJS) *.mps.o flux_metal.o main.o $(TARGET) $(LIB) convert_bf16_to_f16
	rm -f flux_shaders_source.h

clean-shaders:
	rm -f flux_shaders.metallib flux_shaders.air

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
flux.o: flux.c flux.h flux_kernels.h flux_safetensors.h flux_qwen3.h embcache.h flux_lora.h
flux_kernels.o: flux_kernels.c flux_kernels.h
flux_tokenizer.o: flux_tokenizer.c flux.h
flux_vae.o: flux_vae.c flux.h flux_kernels.h
flux_transformer.o: flux_transformer.c flux.h flux_kernels.h flux_lora.h
flux_lora.o: flux_lora.c flux_lora.h flux_safetensors.h
flux_zimage_transformer.o: flux_zimage_transformer.c flux.h flux_kernels.h flux_safetensors.h
flux_sample.o: flux_sample.c flux.h flux_kernels.h
flux_image.o: flux_image.c flux.h
flux_safetensors.o: flux_safetensors.c flux_safetensors.h
flux_qwen3.o: flux_qwen3.c flux_qwen3.h flux_safetensors.h
flux_qwen3_tokenizer.o: flux_qwen3_tokenizer.c flux_qwen3.h
terminals.o: terminals.c terminals.h flux.h
flux_cli.o: flux_cli.c flux_cli.h flux.h flux_qwen3.h embcache.h linenoise.h terminals.h
linenoise.o: linenoise.c linenoise.h
embcache.o: embcache.c embcache.h
main.o: main.c flux.h flux_kernels.h flux_cli.h terminals.h embcache.h
