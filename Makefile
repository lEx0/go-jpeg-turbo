.PHONY: test test-purego race vet cover bench fuzz levels

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
# the amd64 and the arm64 assembly whatever the host is.
vet:
	$(GO) vet ./...
	$(GO) vet -tags purego ./...
	GOARCH=amd64 $(GO) vet ./...
	GOARCH=arm64 $(GO) vet ./...

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

# Each fuzz target for FUZZTIME, e.g. make fuzz FUZZTIME=10m
# FUZZFLAGS=-parallel=8. JPEG_TURBO_LEVEL selects the kernels. The
# fuzzer sees the coverage of Go code only, not of the assembly kernels:
# FuzzLevelKernels shapes its inputs to reach their paths (level_test.go).
FUZZ ?= FuzzCompatDecode FuzzDecCompatFast FuzzCompatEncode FuzzLevelKernels
FUZZTIME ?= 60s
FUZZFLAGS ?=
fuzz:
	@set -e; for f in $(FUZZ); do \
		echo "== $$f"; \
		$(GO) test -run '^$$' -fuzz "^$${f}\$$" -fuzztime $(FUZZTIME) $(FUZZFLAGS) ./jpeg; \
	done

# go test ./jpeg once per JPEG_TURBO_LEVEL this machine supports (go only on
# arm64; go, sse41, avx2, avx512 on an x86-64-v4 machine).
levels:
	@set -e; for l in $$($(GO) run internal/cpuinfo/levels.go); do \
		echo "== JPEG_TURBO_LEVEL=$$l"; \
		JPEG_TURBO_LEVEL=$$l $(GO) test -timeout $(TIMEOUT) $(TESTFLAGS) ./jpeg; \
	done
