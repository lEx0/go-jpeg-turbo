.PHONY: test test-purego race vet vet-ports cover bench bench-compare fuzz levels

GO ?= go

# Extra flags for go test in the test, test-purego, race and levels targets,
# e.g. make levels TESTFLAGS=-short.
TESTFLAGS ?=

# The exhaustive kernel tests make the packages slower than go test's
# default 10m timeout allows on a loaded machine, and the race detector
# makes them several times slower still. Under it internal/ycc converts a
# sample of its exhaustive inputs, and the jpeg tests run their -short
# input sweeps unless -short is given explicitly (about 4 minutes instead
# of 27 to 39 on a shared 32-thread EPYC; see jpeg/race_short_test.go):
# make race TESTFLAGS=-short=false runs the jpeg sweeps in full.
TIMEOUT ?= 30m
RACETIMEOUT ?= 120m

test:
	$(GO) test -timeout $(TIMEOUT) $(TESTFLAGS) ./...

# The portable Go kernels (no assembly) on every architecture.
test-purego:
	$(GO) test -tags purego -timeout $(TIMEOUT) $(TESTFLAGS) ./...

race:
	$(GO) test -race -timeout $(RACETIMEOUT) $(TESTFLAGS) ./...

# go vet (asmdecl included) of the default and purego builds, and of both
# the amd64 and the arm64 assembly whatever the host is. Also the example
# programs and _tools/gencorpus (which regenerates testdata/corpus), which
# ./... skips because their directories start with "_"; the
# ignore-tagged internal/cpuinfo/levels.go, which make levels runs (a
# file, so in a go vet of its own); and the benchmarks module, which
# ./... does not reach either. All of them use the package's API, and
# the benchmarks and levels.go reach into internal/.
EXAMPLES ?= ./_examples/decode ./_examples/encode ./_examples/batch
TOOLS ?= ./_tools/gencorpus
vet:
	$(GO) vet ./... $(EXAMPLES) $(TOOLS)
	$(GO) vet internal/cpuinfo/levels.go
	$(GO) vet -tags purego ./...
	GOARCH=amd64 $(GO) vet ./...
	GOARCH=arm64 $(GO) vet ./...
	cd benchmarks && $(GO) vet ./... && $(GO) vet -tags purego ./...

# go vet of other targets: the portable code (no assembly) on 32-bit,
# big-endian (s390x), other 64-bit and wasm targets, and the amd64
# assembly on an OS without the guard-page tests (freebsd). CI runs it on
# Linux; of these targets it tests only linux/386 (GOARCH=386 on amd64).
PORTS ?= linux/386 linux/arm linux/riscv64 linux/ppc64le linux/s390x linux/loong64 js/wasm wasip1/wasm freebsd/amd64
vet-ports:
	@set -e; for t in $(PORTS); do \
		echo "== go vet $$t"; \
		CGO_ENABLED=0 GOOS=$${t%/*} GOARCH=$${t#*/} $(GO) vet ./...; \
	done

# Coverage of the whole module: code that any package's tests run counts.
# Browse it with go tool cover -html=coverage.txt.
cover:
	$(GO) test -timeout $(TIMEOUT) -coverpkg=./... -coverprofile=coverage.txt ./...
	$(GO) tool cover -func=coverage.txt | tail -1

# This package against image/jpeg (impl=std|turbo, MB/s = megapixels/s);
# compare with benchstat -col /impl. JPEG_TURBO_LEVEL selects the kernels.
BENCH ?= Compat
bench:
	$(GO) test -run '^$$' -bench '$(BENCH)' -benchmem ./jpeg

# This package against image/jpeg on the workloads of benchmarks/suite
# (decode, DecodeConfig, encode, 12 MP photos, saturation, peak memory),
# with the -tags purego build as a third column. Needs benchstat (go
# install golang.org/x/perf/cmd/benchstat@latest); about 20 minutes. The
# knobs are environment variables, e.g. make bench-compare ROUNDS=3 SAT=0;
# see the top of benchmarks/run.sh.
bench-compare:
	./benchmarks/run.sh

# Each fuzz target for FUZZTIME, e.g. make fuzz FUZZTIME=10m
# FUZZFLAGS=-parallel=8. JPEG_TURBO_LEVEL selects the kernels. The
# fuzzer sees the coverage of Go code only, not of the assembly kernels:
# FuzzLevelKernels shapes its inputs to reach their paths (level_test.go).
# FuzzDecProgGen turns the fuzzer's bytes into a progressive stream (scan
# script, tables, entropy-coded data; decode_proggen_test.go), which gets
# further into the progressive decoder than mutations of a file do.
FUZZ ?= FuzzCompatDecode FuzzDecCompatFast FuzzCompatEncode FuzzLevelKernels FuzzDecProgGen
FUZZTIME ?= 60s
FUZZFLAGS ?=
fuzz:
	@set -e; for f in $(FUZZ); do \
		echo "== $$f"; \
		$(GO) test -run '^$$' -fuzz "^$${f}\$$" -fuzztime $(FUZZTIME) $(FUZZFLAGS) ./jpeg; \
	done

# go test ./jpeg once per JPEG_TURBO_LEVEL this machine supports (go only on
# arm64; go, sse41, avx2, avx512 on an x86-64-v4 machine). The list is
# taken first and must not be empty: the shell ignores a failing command
# in a for loop's word list, so a levels.go that does not build would
# otherwise run no test and succeed.
levels:
	@set -e; lv=$$($(GO) run internal/cpuinfo/levels.go); [ -n "$$lv" ]; \
	for l in $$lv; do \
		echo "== JPEG_TURBO_LEVEL=$$l"; \
		JPEG_TURBO_LEVEL=$$l $(GO) test -timeout $(TIMEOUT) $(TESTFLAGS) ./jpeg; \
	done
