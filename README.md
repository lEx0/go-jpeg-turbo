# go-jpeg-turbo

[![CI](https://github.com/lEx0/go-jpeg-turbo/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/lEx0/go-jpeg-turbo/actions/workflows/ci.yml?query=branch%3Amain)
[![Go Reference](https://pkg.go.dev/badge/github.com/lEx0/go-jpeg-turbo/jpeg.svg)](https://pkg.go.dev/github.com/lEx0/go-jpeg-turbo/jpeg)
[![Latest tag](https://img.shields.io/github/v/tag/lEx0/go-jpeg-turbo?sort=semver)](https://github.com/lEx0/go-jpeg-turbo/tags)
[![Go version](https://img.shields.io/github/go-mod/go-version/lEx0/go-jpeg-turbo)](go.mod)
[![License: MIT](https://img.shields.io/github/license/lEx0/go-jpeg-turbo)](LICENSE)

A drop-in replacement for Go's `image/jpeg`: the same API, the same decoded
images and the same encoded bytes as `image/jpeg` of Go 1.27, several times
faster. It is pure Go with Go assembly: **no cgo**, no libjpeg. The only
dependency outside the standard library is `golang.org/x/sys`, which detects
the CPU's features. On arm64 and amd64 the hot loops run as SIMD assembly
(NEON on arm64; SSE4.1, AVX2 or AVX-512 on amd64, picked at run time); other
architectures and `-tags purego` builds run portable Go code, written to give
the same results (tested with `-tags purego` on amd64 and arm64 and on
linux/386, vetted for the others; see [Assembly and the `purego`
tag](#assembly-and-the-purego-tag)).

- **Faster than `image/jpeg`**, one call at a time on the same files (see
  [Benchmarks](#benchmarks)):

  | | Apple M5 Pro (NEON) | AMD EPYC 8124P, one CPU (AVX-512) |
  |---|---:|---:|
  | Decode, baseline photos | 4.3–5.0x | 4.3–4.7x |
  | Decode, progressive photo | 2.9x | 2.7x |
  | Decode, 1750×697 logo | 8.0x | 8.4x |
  | Encode | 8.0–13.1x | 6.9–17.6x |
  | Decode and encode with `-tags purego` | 2.2–4.6x | 1.9–3.3x |

- **Same API.** `Decode`, `DecodeConfig`, `Encode`, `Options`,
  `DefaultQuality`, `FormatError`, `UnsupportedError` and `Reader`, with the
  same signatures, and nothing else: no options, no goroutines of its own.
- **Same results as `image/jpeg` of Go 1.27,** checked by the tests against
  the standard library: decoded images are `reflect.DeepEqual` to its own,
  including the plane padding past the image edge; `Encode` writes the same
  bytes; errors and panics have the same values in the same cases;
  `DecodeConfig` returns the same configs; the reader sees the same `Read`
  calls (see [Compatibility](#compatibility)).
- **The same inputs:** baseline and progressive JPEG, restart intervals,
  every subsampling `image/jpeg` accepts, grayscale, RGB, CMYK and YCCK.
  Files that `image/jpeg` rejects are rejected with the same error.
- **For servers:** safe for concurrent use, one goroutine per call, no
  allocation per `Encode` call of the standard image types once its pooled
  state exists, and no cgo: it builds with `CGO_ENABLED=0` and runs in
  scratch or distroless containers (see [Running on a
  server](#running-on-a-server)).

## Install

```bash
go get github.com/lEx0/go-jpeg-turbo
```

Then change the import path, and nothing else:

```diff
-import "image/jpeg"
+import "github.com/lEx0/go-jpeg-turbo/jpeg"
```

Requires Go 1.27 or newer, because Go 1.27's `image/jpeg` is the one this
package reproduces: it is the first release that decodes files with
non-standard chroma subsampling, such as Cb and Cr sampled differently (Go
1.26 rejects them with `luma/chroma subsampling ratio`). An older `go`
command reports that the module requires go >= 1.27 or, with the default
`GOTOOLCHAIN=auto`, switches to a Go 1.27 toolchain. `go get` raises your
module's `go` line to 1.27 (for example `go: upgraded go 1.24 => 1.27`) and
marks the requirement `// indirect`; `go mod tidy`, run after changing the
import, drops the mark.

A later Go release builds and runs the same code. The tests compare the
package with the `image/jpeg` of the toolchain that runs them, and CI also
runs them with the latest stable Go, which moves to each new release
([CI](#ci)): if a release changes what `image/jpeg` produces, they report
it, and this package keeps producing Go 1.27's results until it is updated.

## Examples

#### Decode

```go
package main

import (
	"bytes"
	"fmt"
	"log"
	"os"

	"github.com/lEx0/go-jpeg-turbo/jpeg" // was: "image/jpeg"
)

func main() {
	data, err := os.ReadFile("testdata/photo-960x1280.jpg")
	if err != nil {
		log.Fatalln(err)
	}

	// DecodeConfig reads only the header: check the size of untrusted
	// input before Decode allocates the pixels.
	cfg, err := jpeg.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		log.Fatalln(err)
	}
	if int64(cfg.Width)*int64(cfg.Height) > 50_000_000 {
		log.Fatalln("image too large")
	}

	img, err := jpeg.Decode(bytes.NewReader(data))
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Printf("%dx%d %T\n", cfg.Width, cfg.Height, img) // 960x1280 *image.YCbCr
}
```

As with `image/jpeg`, `Decode` returns an `*image.YCbCr` (with the file's
chroma subsampling, or 4:4:4 for the other layouts Go 1.27 accepts), an
`*image.Gray` (one component), an `*image.RGBA` (RGB components, no color
transform) or an `*image.CMYK` (Adobe CMYK or YCCK). The size check matters
because `Decode`, like `image/jpeg`, allocates the image the header declares
before it reads the compressed data, and a small file can declare
65535×65535 pixels.

#### Encode

```go
package main

import (
	"bytes"
	"image"
	_ "image/png" // lets image.Decode read PNG
	"log"
	"os"
	"path/filepath"

	"github.com/lEx0/go-jpeg-turbo/jpeg" // was: "image/jpeg"
)

func main() {
	in, err := os.Open("testdata/video-001.png")
	if err != nil {
		log.Fatalln(err)
	}
	defer in.Close()

	img, _, err := image.Decode(in)
	if err != nil {
		log.Fatalln(err)
	}

	var out bytes.Buffer
	if err := jpeg.Encode(&out, img, &jpeg.Options{Quality: 90}); err != nil {
		log.Fatalln(err)
	}
	if err := os.WriteFile(filepath.Join(os.TempDir(), "video-001.jpg"), out.Bytes(), 0o644); err != nil {
		log.Fatalln(err)
	}
}
```

A nil `*Options` means `jpeg.DefaultQuality` (75); qualities outside 1–100
are clamped. Like `image/jpeg`, `Encode` writes baseline JPEG with the
standard Huffman tables: one component for an `*image.Gray`, 4:2:0 YCbCr for
every other image. Images 65536 pixels wide or high fail with `jpeg: image
is too large to encode`. Any `image.Image` works; `*image.RGBA`,
`*image.NRGBA`, `*image.YCbCr`, `*image.Gray`, `*image.RGBA64`,
`*image.NRGBA64`, `*image.CMYK`, `*image.Gray16`, `*image.Alpha`,
`*image.Alpha16`, `*image.Paletted` (when its palette holds only standard
`image/color` types) and `*image.NYCbCrA` are read from their pixel slices
instead of through `At`. An image whose slices are too short for its
rectangle is read the way `image/jpeg` reads it, so that it panics at the
same pixel.

#### `image.Decode`

Importing the package registers the "jpeg" format with package `image`, as
`image/jpeg` does, so a program that only uses `image.Decode` and
`image.DecodeConfig` needs just the blank import (if `image/jpeg` is linked
too, see [What differs](#what-differs)):

```go
package main

import (
	"fmt"
	"image"
	"log"
	"os"

	_ "github.com/lEx0/go-jpeg-turbo/jpeg" // was: _ "image/jpeg"
)

func main() {
	f, err := os.Open("testdata/video-005.gray.jpeg")
	if err != nil {
		log.Fatalln(err)
	}
	defer f.Close()

	img, format, err := image.Decode(f)
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Println(format, img.Bounds()) // jpeg (0,0)-(150,103)
}
```

#### Errors

`Decode` and `DecodeConfig` fail in the same cases and with the same
messages as `image/jpeg`. The error types are this package's own (see
[What differs](#what-differs)):

```go
package main

import (
	"bytes"
	"errors"
	"fmt"
	"io"

	"github.com/lEx0/go-jpeg-turbo/jpeg"
)

func main() {
	_, err := jpeg.Decode(bytes.NewReader([]byte("GIF89a")))

	var formatErr jpeg.FormatError
	var unsupportedErr jpeg.UnsupportedError
	switch {
	case errors.As(err, &formatErr):
		// Not JPEG, corrupt, or cut short inside a scan's compressed data.
		fmt.Println("invalid:", string(formatErr)) // invalid: missing SOI marker
	case errors.As(err, &unsupportedErr): // arithmetic coding, 12-bit samples, ...
		fmt.Println("unsupported:", string(unsupportedErr))
	case errors.Is(err, io.ErrUnexpectedEOF):
		// Cut short in a header, at a restart marker, between scans or
		// before the final EOI marker.
		fmt.Println("truncated")
	}
}
```

As with `image/jpeg`, a file whose pixel data is complete but that lacks its
final EOI marker also fails with `io.ErrUnexpectedEOF`.

#### Example programs

[`_examples`](_examples) holds three programs. Run them from the repository
root; they write under the temporary directory unless given `-out`:

```bash
go run ./_examples/decode                      # a few files from testdata/ to PNG
go run ./_examples/decode -out /tmp/png a.jpg  # your own files
go run ./_examples/encode                      # a JPEG and a PNG from testdata/ at -q 50,75,90
go run ./_examples/encode -q 60,85 a.png       # JPEG, PNG or GIF input
go run ./_examples/batch                       # every JPEG under testdata/, on GOMAXPROCS goroutines
go run ./_examples/batch -dir ~/Pictures -q 80 -workers 4
go run -race ./_examples/batch -out ''         # race detector, nothing written
```

[`_examples/batch`](_examples/batch/main.go) re-encodes every JPEG under a
directory and groups the files `Decode` rejects by error message. On
`testdata` (M5 Pro, `-out ''`; the time includes reading the files and
varied from 19 to 41 ms between runs):

```
1233 JPEG files under testdata, quality 80, workers 15
re-encoded 1045: 24.9 megapixels, 5.2 MiB -> 4.0 MiB, in 30ms (823 megapixels/s)
skipped 86: invalid JPEG format: missing 0xff00 sequence (testdata/corpus/photo16x16_141211_prog.jpeg, ...)
skipped 1: invalid JPEG format: total sampling factors too large (testdata/corpus/photo33x31_441111_seqscans7.jpeg, ...)
skipped 66: unsupported JPEG feature: luma/chroma subsampling ratio (testdata/corpus/photo16x16_311111_base.jpeg, ...)
skipped 11: unsupported JPEG feature: precision (testdata/corpus/photo16x16_p12.jpeg, ...)
skipped 24: unsupported JPEG feature: unknown marker (testdata/corpus/photo16x16_arith.jpeg, ...)
```

The 188 skipped files are in the test corpus on purpose, and `image/jpeg`
rejects each of them with the same error. 102 use features it does not
support (arithmetic coding, lossless JPEG, 12-bit samples, a sampling factor
of 3, a 4×4 luma component in a three-component file). The other 86 are
valid files that it fails to decode (`missing 0xff00 sequence`, see
[Supported inputs](#supported-inputs)).

The examples in [`jpeg/example_test.go`](jpeg/example_test.go), which
pkg.go.dev shows, run as tests with checked output (`go test ./jpeg -run
Example`); `Example_compatibility` decodes and encodes a file with both
packages and prints `same image: true` and `same bytes: true`.

## Compatibility

The target is `image/jpeg` of Go 1.27, and the [tests](#tests) compare with
it.

#### What is the same

- **Decoded images:** the same concrete type, bounds, strides, subsampling
  ratio and every byte of the pixel buffers, including the padding of the
  planes past the image edge (`reflect.DeepEqual`).
- **`DecodeConfig`:** the same `image.Config`.
- **Errors:** for every input the same text and kind: `FormatError`,
  `UnsupportedError`, `io.ErrUnexpectedEOF` or the reader's own error.
- **The reader:** the same `Read` calls with the same buffer lengths, in the
  same order, so `Decode` leaves a reader at the same position as
  `image/jpeg` does. Like `image/jpeg`, it reads up to 4096 bytes at a time
  and can consume bytes past the end of the image; when it has (from a
  `bytes.Reader` that holds two files, for instance), a second `Decode` on
  the same reader fails in both packages: with `missing SOI marker`, or
  with `unexpected EOF` when the first one read to the end of the
  stream. To
  decode JPEG after JPEG from one stream, split the stream first, as a
  multipart MJPEG reader does.
- **Encoded bytes,** for every image type, size, origin and quality, as long
  as `At` returns the same color each time it is called for a pixel (see
  [What differs](#what-differs)).
- **The writer:** `Encode` wraps a plain `io.Writer` in a 4096-byte buffer,
  as `image/jpeg` does, and makes exactly the same `Write` calls; it leaves a
  `*bufio.Writer` in the same state and reports a failing writer the same
  way.
- **Panics:** a malformed image (a `Pix` too short for its rectangle, a
  palette index past the palette) or an `At` that panics at a given pixel
  makes `Encode` panic at the same pixel with the same value.

#### What differs

None of these changes the decoded pixels or the encoded bytes, as long as
`At` returns the same color for a pixel every time it is called.

- **Memory** ([Allocations](#allocations)). `Decode` allocates 0.1% to 2.8%
  more than `image/jpeg` on the benchmark files (the tests allow at most 10%
  plus 64 KiB). For the image types it reads from their pixel slices,
  `Encode` allocates nothing per call once its pooled state exists, where
  `image/jpeg` allocates about 4.5 KB per call (4.46–4.55 KB in the M5 Pro
  benchmark, its counting writer included), and 4.9 MB for a 1.2-megapixel
  `*image.NRGBA`. Other images allocate whatever their `At` method does, as
  with `image/jpeg`. `DecodeConfig` allocates 2.8 KB more per call and takes
  28% longer ([One call at a time](#one-call-at-a-time)).
- **Error types.** `jpeg.FormatError` and `jpeg.UnsupportedError` are
  distinct from `image/jpeg`'s types. The messages are the same, but a type
  assertion or `errors.As` against `image/jpeg`'s `FormatError` does not
  match them.
- **Two registrations.** If a program links both `image/jpeg` and this
  package, both register "jpeg", and `image.Decode` uses the first. Go
  initializes, at each step, the first package in import-path order whose
  imports are all initialized. For these two packages that puts this one
  (`github.com/...`) first: Go 1.27.0 binaries for Linux, macOS, Windows and
  FreeBSD on amd64 and arm64, and for linux/386, record that order, with
  and without `-tags purego`. Only in its own tests, which import
  `image/jpeg`, does `image/jpeg` come first. Call `jpeg.Decode` directly to
  be independent of the order.
- **Writers with their own `Flush` and `WriteByte`** that are not a
  `*bufio.Writer`: `image/jpeg` writes the compressed data to them one
  `WriteByte` call at a time, `Encode` with `Write` calls of up to 64 KiB.
- **Panics while encoding.** If `At` or the writer panics, or a malformed
  image makes `Encode` panic, in the middle of the image, fewer bytes may
  have reached the writer, because the output is buffered in larger chunks.
  The panic value is the same.
- **`At` calls.** For image types without a fast path, `Encode` calls `At`
  once per pixel, in the order of `image/jpeg`'s first call for each pixel;
  `image/jpeg` calls it again for the edge pixels of partial blocks. So an
  image whose `At` returns another color on a later call for the same pixel
  (a generator, a live source), or panics after a number of calls, can
  encode to other bytes or panic at another pixel than with `image/jpeg`
  when its width or height is not a multiple of 16.
- **The `Write` buffer.** As with `image/jpeg`, the slice given to `Write`
  is the encoder's buffer and is only valid during the call. Here it is
  pooled and reused by later `Encode` calls, so a writer must not keep it
  (the `io.Writer` contract says so anyway).

#### Supported inputs

`Decode` and `DecodeConfig` accept what Go 1.27's `image/jpeg` accepts:

- baseline (SOF0), extended sequential (SOF1) and progressive (SOF2)
  Huffman-coded JPEG with 8-bit samples, with or without restart intervals
  and optimized Huffman tables, with 8- or 16-bit quantization tables;
- one component (gray), three (YCbCr, or RGB when the Adobe marker or the
  component IDs say so) and four (CMYK or YCCK, with an Adobe APP14 marker);
- sampling factors 1, 2 and 4: 4:4:4, 4:4:0, 4:2:2, 4:2:0, 4:1:1 and 4:1:0,
  and other YCbCr layouts whose factors divide the largest ones (new in Go
  1.27, decoded to 4:4:4). An RGB file decodes only if its second and third
  components have the same factors and these divide the first one's;
  `Decode` rejects other RGB layouts with `luma/chroma subsampling ratio`,
  as `image/jpeg` does. Four components need every factor 1, or 2×2 for
  the first and the fourth component.

Everything else is rejected as `image/jpeg` rejects it, with the same error:
arithmetic coding, lossless and hierarchical JPEG (`unsupported JPEG
feature: unknown marker`), 12-bit samples (`precision`), a sampling factor
of 3 (`luma/chroma subsampling ratio`), sampling factors whose products H×V
add up to more than 10 over the components of a scan, even a scan of one
component, in a file of more than one component (`total sampling factors
too large`; a one-component file is treated as 1×1, so a gray file that
declares 4×4 decodes), two components (`number of components`) and four
components without an Adobe marker.

Like `image/jpeg`, it also rejects some valid files that libjpeg decodes,
with `invalid JPEG format: missing 0xff00 sequence`: mainly chroma-subsampled
files with restart intervals whose scans are not interleaved (progressive
files, or sequential files with a scan per component), and some progressive
files with 4×1 or 1×4 luma sampling. 86 files of `testdata/corpus` are of
this kind; libjpeg-turbo 3.2.0's `djpeg` decodes all of them.
[`testdata/corpus`](testdata/corpus) holds files of most kinds above. It has
no hierarchical, YCCK or two-component file, no four-component file without
an Adobe marker or with 2×2 sampling, and no RGB file of a rejected layout;
the tests' hand-built streams add YCCK and 2×2 four-component files. The
tests compare every error.

## Running on a server

The package has no knobs of its own. What matters for a service, and the
data behind it:

- **Concurrency.** `Decode`, `DecodeConfig` and `Encode` are safe for
  concurrent use. Each `Decode` and `DecodeConfig` call has a decoder of its
  own; `Encode` takes its working state from a `sync.Pool` and puts it back
  at the end (after a panic it is dropped). Besides the pool, the calls
  share only read-only tables and the SIMD level, chosen once at startup.
  `TestConcurrent` mixes calls of all three from 16 goroutines. It compares
  decoded images, configs, errors and encoded bytes with `image/jpeg`'s, and
  checks the errors and panics of failing writers, panicking `At` methods,
  malformed images and panicking writers; CI also runs it under the race
  detector. [`_examples/batch`](_examples/batch/main.go) shows a worker
  pool.
- **One goroutine per call.** The codec starts no goroutines: a call runs on
  the goroutine that makes it, on one CPU at a time, and throughput comes
  from calls running in parallel. On the 15-core M5 Pro, 15 calls at a time
  decoded a 1.2-megapixel photo at 7875 MP/s (4.0x `image/jpeg`) and encoded
  an `*image.RGBA` at 8750 MP/s (8.3x); see [Throughput under
  load](#throughput-under-load). With the data in memory the calls are
  CPU-bound, so more than about `GOMAXPROCS` of them in flight add no
  throughput, but each `Decode` in flight holds its output image: bound them
  to about `GOMAXPROCS` (a semaphore, or a worker pool as in
  `_examples/batch`) and queue or shed the rest.
- **Memory per call** (M5 Pro, [Peak memory](#peak-memory)). `Decode`
  allocates little more than its output image: 18.1 MB for a 12-megapixel
  4:2:0 photo, whose decode made the process peak at 26.7 MB, 8.2 MB of it
  before the call; a progressive file also needs its coefficients (9.49 MB
  against 1.88 MB at 1.2 megapixels), as with `image/jpeg`. A 12-megapixel
  encode raised the peak by 0.2 MB over its input. Check `DecodeConfig`
  against your limits before `Decode` (see the [example](#decode)).
- **`GOMAXPROCS` and containers.** The Go runtime sets the default
  `GOMAXPROCS` from the CPUs the process may use, including, on Linux, the
  cgroup CPU limit of a container. It does so for modules that declare go
  1.25 or later, and a module that uses this package declares at least go
  1.27.
- **`CGO_ENABLED=0`.** No cgo and no C toolchain are needed: a
  `CGO_ENABLED=0` build is a statically linked binary (checked for
  linux/amd64 and linux/arm64) that runs in a scratch or distroless image,
  and cross compilation with `GOOS`/`GOARCH` works as for any Go code. No
  `GOAMD64` setting is needed either (see [Assembly](#assembly-and-the-purego-tag)).
- **The race detector** does not see memory accesses made by assembly: when
  looking for races on buffers shared with the codec (an image another
  goroutine changes during `Encode`), also test with `-race -tags purego`.

## Assembly and the `purego` tag

Three kernels run as assembly, written for the Go assembler (no cgo):

| Kernel | arm64 | amd64 SSE4.1 | amd64 AVX2 | amd64 AVX-512 |
|---|---|---|---|---|
| Dequantize + inverse DCT + store (decoder) | NEON | SSE4.1 | AVX2 | AVX2 |
| Forward DCT + quantize + zig-zag bitmap (encoder) | NEON | SSE4.1 | AVX2 | AVX-512 |
| RGBA/NRGBA → YCbCr 4:4:4 and 4:2:0, 2×2 chroma averaging (encoder) | NEON | SSE4.1 | AVX2 | AVX-512 with VNNI, else AVX2 |

Everything else, the Huffman decoding and encoding included, is Go code in
every build.

- **Levels.** Every arm64 CPU has NEON. On amd64 the highest level the CPU
  and the operating system support is picked once, at startup, from
  `golang.org/x/sys/cpu` (which checks that the OS saves the vector
  registers): SSE4.1 (SSE2 through SSE4.1, SSSE3 included), AVX2 (AVX,
  AVX2, BMI1, BMI2 and FMA) or AVX-512 (AVX-512 F, BW, CD, DQ and VL). A
  CPU without SSE4.1 runs the portable Go code. The AVX-512 level runs the
  AVX2 inverse DCT: its two passes have no pairs of values that go through
  the same operations, so 512-bit registers could only speed up the
  transposes, and on the EPYC 8124P (Zen 4c) they do not. The AVX-512
  forward DCT and color conversion are faster than their AVX2 versions (see
  [Kernels](#kernels)).
- **What each level brings** (EPYC 8124P, one CPU): SSE4.1 does most of the
  work (the 4:2:0 q75 photo decodes at 178 MP/s with the portable kernels of
  `JPEG_TURBO_LEVEL=go`, 181 with `-tags purego`, 372 with SSE4.1, 420 with
  AVX2, 417 with AVX-512). Against AVX2, AVX-512 takes 1.5–6.3% less time
  per encode and decodes at the same speed.
- **`GOAMD64`** does not change the level, only the code the Go compiler
  generates for the Go parts: `GOAMD64=v3` took 4.6–5.6% off every encode of
  this package on the EPYC and changed none of its decodes significantly
  ([Building with GOAMD64=v3](#building-with-goamd64v3)).
- **`GODEBUG`** lowers the level of a process, for example to rule the newer
  kernels out while investigating a problem: `x/sys/cpu` reads
  `GODEBUG=cpu.avx512f=off` (AVX2 level), `cpu.avx2=off` (SSE4.1) and
  `cpu.sse41=off` (portable Go).
- **`JPEG_TURBO_LEVEL`** (`go`, `sse41`, `avx2`, `avx512`) is read only by
  the tests and benchmarks of this repository, never by the library. It
  makes `go test ./jpeg` run every comparison with `image/jpeg` through the
  kernels of one level; `make levels` runs the package tests once per level
  the machine supports. A level above the CPU's stops the test binary at
  start. On arm64 only `go` is accepted, and the NEON kernels still run.
- **`-tags purego`** drops the assembly on arm64 and amd64; every other
  architecture always runs the portable code. That code is the reference
  functions rearranged for speed (the color conversion two pixels per 64-bit
  word, an integer DCT that skips the arithmetic of zero rows), and is
  2.2–4.6x as fast as `image/jpeg` on the M5 Pro and 1.9–3.3x on the EPYC.
  CI tests it on linux/386 and vets it for linux/arm, riscv64, ppc64le,
  s390x (big-endian) and loong64, js/wasm and wasip1/wasm.

**The output does not depend on any of this.** Every kernel computes exactly
what its Go reference computes, at every level, so decoded pixels and
encoded bytes are the same with and without `-tags purego`, on every
architecture and CPU. The references are `image/jpeg`'s own transforms
(copied from Go 1.27) and its per-pixel color conversion. The tests check:

- each kernel at every level the machine supports against its reference,
  exhaustively where the input space allows (the quantizer on every
  coefficient value and quantizer value, every DC-only block, the color
  conversion on all 2^24 RGB and all 2^32 NRGBA values, a dozen alpha
  values with `-short`), otherwise on random, boundary and photo inputs;
- the kernels next to inaccessible pages on both sides, so that an
  out-of-bounds read or write faults, and at every offset Go's alignment
  allows;
- the machine code of the amd64 kernels against the instructions of their
  level (`TestKernelISA`), so that an SSE4.1 kernel cannot contain an AVX
  instruction that a CPU without AVX would fault on;
- the whole decoder and encoder against `image/jpeg` at each level (`make
  levels`, and the CI levels job) and with `-tags purego`;
- that the DCT assembly is what its generators (`internal/dct/asmgen*.go`)
  write (`TestGenerated`); the color conversion assembly is written by hand.

## Benchmarks

### Method

- **Apple M5 Pro:** [`benchmarks/run.sh`](benchmarks/run.sh) (`make
  bench-compare`) with its defaults, on 2026-11-23 at eac4c8b: `image/jpeg`
  of Go 1.27.0, this package with `-tags purego`, and this package as built
  by default (NEON). 10 rounds, the builds one after another in a rotating
  order within each, so that a change of the machine's state affects all of
  them alike; 1 s per single-call and 2 s per saturation benchmark, with
  `-benchmem`; 5 peak-memory runs. `GOMAXPROCS`, `GOGC` and `GOMEMLIMIT` at
  their defaults, no CPU pinning, load average 3.1 at the start and 4.8 at
  the end. Before measuring, run.sh checks that each build gives
  `image/jpeg`'s results on every workload (`TestSameAsStd`).
- **AMD EPYC 8124P:** `BenchmarkCompatDecode` and `BenchmarkCompatEncode`
  in [`jpeg/bench_compat_test.go`](jpeg/bench_compat_test.go), on the same
  files and encode inputs, without the 12-megapixel and `DecodeConfig`
  workloads. 8 runs per build, single thread: pinned to one CPU with
  `taskset -c 2`, `performance` frequency governor. Each SIMD level is the
  default build with `JPEG_TURBO_LEVEL`; purego is `-tags purego`. Measured
  at 8ebae17; the library code has not changed since.
- **MP/s** is decimal megapixels per second: width × height divided by the
  time of one call (in the saturation benchmarks, of all goroutines
  together). Each call decodes from a `[]byte` in memory and allocates its
  output image, or encodes a prepared image: on the M5 Pro into a writer
  that only counts the bytes, which `image/jpeg` wraps in a `bufio.Writer`
  and `Encode` in its pooled buffer, on the EPYC into a reused
  `bytes.Buffer`. **CPU time** is the user and system time of the whole
  process per call, garbage collection included.
- **The tables show medians**: MP/s is the pixel count divided by the median
  time of a call, and the speed-ups are ratios of the median times.
  benchstat's 95% confidence intervals, left out here, are within ±5% of the
  median for the times, MP/s and CPU times shown of `image/jpeg` and NEON
  and ±7% for purego on the M5 Pro, except for this package's
  `DecodeConfig` times (±6% purego, ±8% NEON). On the EPYC they are within
  ±5% except for three SSE4.1 decodes: the progressive photo (±8%), the gray
  photo and the logo (±6%). The bytes per encode of this package in
  [Allocations](#allocations) vary much more (±93–98%). Absolute numbers
  differ on other hardware; compare the columns of a table.
- **Inputs:** the decodes read the photos of
  [`testdata/bench`](testdata/bench), `testdata/photo-960x1280.jpg` (4:4:4
  q92, with restart markers and 23 KB of Exif, XMP, ICC and Photoshop
  segments) and `testdata/logo-1750x697.jpg`. The encodes take
  `photo-960x1280.jpg` converted to `*image.RGBA`, `*image.NRGBA` and
  `*image.Gray` with `image/draw`, the same file as decoded (`*image.YCbCr`
  4:4:4) and the q75 4:2:0 photo as decoded; the rows name the input, and
  the output is 4:2:0, or gray for `*image.Gray`. The 12-megapixel photo
  (M5 Pro only) is made by [`benchmarks/suite/data.go`](benchmarks/suite/data.go):
  the 1920×2560 4:2:0 photo decoded, mirror-tiled to 4000×3000 and encoded
  with `image/jpeg` at q90, checked against SHA-256 sums.

### Apple M5 Pro (arm64)

15 cores, 24 GiB RAM, macOS 27.2, Go 1.27.0, darwin/arm64
([`benchmarks/results/m5pro`](benchmarks/results/m5pro)).

#### One call at a time

| MP/s | `image/jpeg` | purego | NEON | purego vs `image/jpeg` | NEON vs `image/jpeg` |
|---|---:|---:|---:|---:|---:|
| Decode 4:2:0 q75, 960×1280 | 183 | 444 | 915 | 2.4x | 5.0x |
| Decode 4:4:4 q92, 960×1280, 23 KB of metadata | 86 | 224 | 416 | 2.6x | 4.8x |
| Decode 4:2:2 q92, 960×1280, restart markers | 112 | 281 | 510 | 2.5x | 4.6x |
| Decode 4:2:0 q85, 960×1280, progressive | 94 | 205 | 269 | 2.2x | 2.9x |
| Decode 4:2:0 q85, 1920×2560 | 196 | 458 | 924 | 2.3x | 4.7x |
| Decode gray q85, 1920×2560 | 260 | 602 | 1180 | 2.3x | 4.5x |
| Decode 4:4:4, 1750×697 logo | 136 | 595 | 1092 | 4.4x | 8.0x |
| Decode 4:2:0 q90, 4000×3000 (12 MP) | 183 | 425 | 786 | 2.3x | 4.3x |
| Encode `*image.RGBA` q75 | 91 | 308 | 817 | 3.4x | 9.0x |
| Encode `*image.RGBA` q90 | 77 | 278 | 641 | 3.6x | 8.3x |
| Encode `*image.NRGBA` q75 | 57 | 255 | 751 | 4.5x | 13.1x |
| Encode `*image.NRGBA` q90 | 51 | 234 | 599 | 4.6x | 11.6x |
| Encode `*image.YCbCr` 4:4:4 q75 | 93 | 412 | 969 | 4.4x | 10.5x |
| Encode `*image.YCbCr` 4:4:4 q90 | 78 | 359 | 728 | 4.6x | 9.3x |
| Encode `*image.YCbCr` 4:2:0 q75 | 100 | 437 | 982 | 4.4x | 9.8x |
| Encode `*image.YCbCr` 4:2:0 q90 | 97 | 424 | 935 | 4.4x | 9.6x |
| Encode `*image.Gray` q75 | 146 | 612 | 1292 | 4.2x | 8.8x |
| Encode `*image.Gray` q90 | 116 | 516 | 930 | 4.4x | 8.0x |
| Encode `*image.RGBA` 4000×3000 (12 MP) q75 | 99 | 310 | 821 | 3.1x | 8.3x |
| Encode `*image.RGBA` 4000×3000 (12 MP) q90 | 96 | 304 | 783 | 3.2x | 8.1x |

The 12-megapixel decode takes 65.5 ms with `image/jpeg`, 28.2 ms with purego
and 15.3 ms with NEON; the 12-megapixel q90 encode 124.6, 39.5 and 15.3 ms.
`DecodeConfig` of the 4:4:4 photo, which has to skip its 23 KB of metadata,
takes 1.12 µs with `image/jpeg` and 1.44 µs with both builds of this package
(+28%).

#### Throughput under load

The same calls, 15 at a time on the 15 cores (`GOMAXPROCS` goroutines,
`b.RunParallel`), as on a busy server. MP/s is the throughput of all of them
together; CPU time is per call.

| 15 calls at a time | `image/jpeg` | purego | NEON |
|---|---:|---:|---:|
| Decode 4:2:0 q75, 960×1280: MP/s | 1955 | 4328 (2.2x) | 7875 (4.0x) |
| ↳ CPU time per call | 8.40 ms | 3.60 ms | 1.80 ms |
| Encode `*image.RGBA` q75, 960×1280: MP/s | 1051 | 2956 (2.8x) | 8750 (8.3x) |
| ↳ CPU time per call | 16.5 ms | 5.99 ms | 2.01 ms |

That is 6,410 decodes or 7,120 encodes per second with NEON, against 1,590
and 855 with `image/jpeg`. Under load every build takes more CPU time per
call than one call at a time (for the NEON decode 1.80 ms against 1.43 ms).

#### Peak memory

Peak resident memory (`getrusage` maxrss) of a process that makes one call,
median of 5 runs (the five peaks of each cell within 0.4 MB of each other),
with the peak before the call (the Go runtime and the input) and the wall
time of the call:

| One call per process | `image/jpeg` | purego | NEON |
|---|---:|---:|---:|
| Decode the 12 MP 4:2:0 q90 JPEG (2.0 MB) | 26.5 MB, 64.6 ms | 26.8 MB, 29.7 ms | 26.7 MB, 16.7 ms |
| ↳ before the call | 8.1 MB | 8.2 MB | 8.2 MB |
| Encode the 12 MP `*image.RGBA` (48 MB) at q90 | 54.6 MB, 125.1 ms | 55.0 MB, 40.6 ms | 54.9 MB, 16.3 ms |
| ↳ before the call | 54.5 MB | 54.8 MB | 54.7 MB |

The decode raises the peak by about 18.5 MB, little more than its 18 MB
output image; the encode raises it by 0.1–0.2 MB over its input.

#### Allocations

Per call, from the same runs (MB and KB are 10^6 and 10^3 bytes); the purego
build allocates the same as NEON for decodes:

| Per call | `image/jpeg` | this package |
|---|---:|---:|
| Decode 4:2:0 q75, 960×1280 | 1.86 MB, 7 allocs | 1.88 MB (+1.0%), 11 allocs |
| Decode 4:4:4 q92, 960×1280, 23 KB of metadata | 3.70 MB, 196 allocs | 3.72 MB (+0.5%), 200 allocs |
| Decode 4:2:0 q85, 960×1280, progressive | 9.23 MB, 14 allocs | 9.49 MB (+2.8%), 21 allocs |
| Decode 4:2:0 q90, 4000×3000 (12 MP) | 18.07 MB, 6 allocs | 18.09 MB (+0.1%), 10 allocs |
| `DecodeConfig` of the 4:4:4 photo | 13.6 KB, 2 allocs | 16.4 KB (+20.7%), 2 allocs |
| Encode `*image.RGBA` q75 | 4.55 KB, 8 allocs | 137 B, 1 alloc |
| Encode `*image.NRGBA` q75 | 4.92 MB, 1,228,808 allocs | 148 B, 1 alloc |
| Encode `*image.RGBA` 4000×3000 (12 MP) q90 | 4.55 KB, 8 allocs | 2.26 KB, 1 alloc |

The one allocation per encode is the benchmark's counting writer, which
escapes as an `io.Writer` (`image/jpeg`'s 8 include it); encoding into
`io.Discard` allocates nothing (`TestEncAllocs`). The bytes beyond it are
the pooled encoder state (about 100 KB for the 960×1280 `*image.RGBA`),
which `Encode` makes again when `sync.Pool` has dropped it (after two
garbage collections without an `Encode` call), averaged over a run.
`image/jpeg` reads an `*image.NRGBA` through `At`, which allocates a color
value per pixel.

### AMD EPYC 8124P (amd64)

Zen 4c, 16 cores, Linux, linux/amd64 (`GOAMD64=v1` unless stated), one CPU
(`taskset -c 2`), `performance` governor
([`benchmarks/results`](benchmarks/results), files `epyc8124p-*`).

| MP/s | `image/jpeg` | SSE4.1 | AVX2 | AVX-512 | AVX-512 vs `image/jpeg` | purego | purego vs `image/jpeg` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Decode 4:2:0 q75, 960×1280 | 89 | 372 | 420 | 417 | 4.7x | 181 | 2.0x |
| Decode 4:4:4 q92, 960×1280, 23 KB of metadata | 42 | 175 | 195 | 195 | 4.6x | 90 | 2.1x |
| Decode 4:2:2 q92, 960×1280, restart markers | 57 | 221 | 245 | 245 | 4.3x | 116 | 2.0x |
| Decode 4:2:0 q85, 960×1280, progressive | 45 | 119 | 123 | 122 | 2.7x | 86 | 1.9x |
| Decode 4:2:0 q85, 1920×2560 | 95 | 388 | 434 | 439 | 4.6x | 185 | 2.0x |
| Decode gray q85, 1920×2560 | 129 | 517 | 583 | 585 | 4.5x | 250 | 1.9x |
| Decode 4:4:4, 1750×697 logo | 58 | 444 | 474 | 483 | 8.4x | 193 | 3.3x |
| Encode `*image.RGBA` q75 | 40 | 309 | 401 | 425 | 10.5x | 101 | 2.5x |
| Encode `*image.RGBA` q90 | 35 | 240 | 292 | 305 | 8.6x | 92 | 2.6x |
| Encode `*image.NRGBA` q75 | 24 | 304 | 397 | 423 | 17.6x | 76 | 3.2x |
| Encode `*image.NRGBA` q90 | 22 | 237 | 290 | 304 | 13.7x | 71 | 3.2x |
| Encode `*image.YCbCr` 4:4:4 q75 | 44 | 356 | 439 | 450 | 10.3x | 143 | 3.3x |
| Encode `*image.YCbCr` 4:4:4 q90 | 38 | 267 | 312 | 317 | 8.4x | 126 | 3.3x |
| Encode `*image.YCbCr` 4:2:0 q75 | 45 | 364 | 447 | 458 | 10.1x | 150 | 3.3x |
| Encode `*image.YCbCr` 4:2:0 q90 | 45 | 339 | 411 | 421 | 9.5x | 146 | 3.3x |
| Encode `*image.Gray` q75 | 69 | 465 | 558 | 572 | 8.3x | 211 | 3.0x |
| Encode `*image.Gray` q90 | 57 | 339 | 386 | 392 | 6.9x | 181 | 3.2x |

AVX-512 is this CPU's default level. The `image/jpeg` column and the
AVX-512 ratios come from the AVX-512 run, and the purego ratios divide the
purego run's MP/s by that column (within the purego run, against its own
`image/jpeg`, the logo is 3.4x; every other row is the same either way).
Every other `GOAMD64=v1` run measured `image/jpeg` again, within 2% of these
values. `JPEG_TURBO_LEVEL=go` (the portable kernels in the default build)
runs within 2% of `-tags purego`. Decodes make as many allocations as on
the M5 Pro, and their bytes per call differ by at most 100. The encodes
write into a `bytes.Buffer` that each benchmark reuses, so there is no
counting writer to allocate: this package makes no allocation per call,
and `image/jpeg` 5–7 (1,228,807 and 4.9–5.0 MB for the `*image.NRGBA`).
The other bytes per call that benchstat reports for the encodes, 1.1–3.9 KB
for this package at the AVX-512 level (up to 17 KB at level go and with
`-tags purego`, which run fewer iterations) and 12–36 KB for `image/jpeg`,
mostly come from the buffer growing during the first iterations, spread
over the run. There are no
saturation or peak-memory measurements for amd64.

#### Building with GOAMD64=v3

`GOAMD64=v3` lets the Go compiler use x86-64-v3 instructions in Go code, and
the binary then runs only on such CPUs; the kernels are picked at run time
either way. Same server and method, AVX-512 level, MP/s with `GOAMD64=v1`
and `v3`, and the change in time per call as benchstat reports it ("~": no
significant difference, p > 0.05):

| MP/s | `image/jpeg` v1 | `image/jpeg` v3 | time | this package v1 | this package v3 | time |
|---|---:|---:|---:|---:|---:|---:|
| Decode 4:2:0 q75, 960×1280 | 89 | 90 | ~ | 417 | 425 | ~ |
| Decode 4:2:0 q85, 960×1280, progressive | 45 | 45 | −0.7% | 122 | 124 | ~ |
| Decode gray q85, 1920×2560 | 129 | 129 | ~ | 585 | 577 | ~ |
| Encode `*image.RGBA` q75 | 40 | 41 | −1.1% | 425 | 446 | −4.7% |
| Encode `*image.RGBA` q90 | 35 | 36 | −0.9% | 305 | 321 | −5.1% |
| Encode `*image.NRGBA` q75 | 24 | 24 | −0.6% | 423 | 444 | −4.7% |
| Encode `*image.NRGBA` q90 | 22 | 22 | −0.5% | 304 | 320 | −5.0% |
| Encode `*image.YCbCr` 4:4:4 q75 | 44 | 44 | −1.5% | 450 | 474 | −5.0% |
| Encode `*image.YCbCr` 4:4:4 q90 | 38 | 38 | −1.0% | 317 | 335 | −5.3% |
| Encode `*image.YCbCr` 4:2:0 q75 | 45 | 46 | −0.7% | 458 | 483 | −5.1% |
| Encode `*image.YCbCr` 4:2:0 q90 | 45 | 45 | −0.8% | 421 | 441 | −4.6% |
| Encode `*image.Gray` q75 | 69 | 72 | −3.7% | 572 | 606 | −5.6% |
| Encode `*image.Gray` q90 | 57 | 59 | −3.1% | 392 | 414 | −5.3% |

Every encode of this package takes 4.6–5.6% less time, and none of its seven
decodes changes significantly; of the four decodes left out of the table,
only `image/jpeg`'s of the 1920×2560 4:2:0 photo changed (−0.8%). Choose
`GOAMD64` for the rest of the service; the codec does not need it.

#### Kernels

Per call, from the benchmarks of `internal/dct` and `internal/ycc`; the
reference is `image/jpeg`'s code for the transforms and its per-pixel
conversion for the color kernel, "portable" the Go code of `-tags purego`:

| Kernel, EPYC 8124P | reference | portable | SSE4.1 | AVX2 | AVX-512 |
|---|---:|---:|---:|---:|---:|
| Dequantize + inverse DCT + store, a photo's 8×8 block | 277 ns | 160 ns | 47.1 ns | 38.6 ns | 38.6 ns |
| Forward DCT + quantize, a photo's 8×8 block | 297 ns | 214 ns | 52.8 ns | 31.7 ns | 27.9 ns |
| RGBA → YCbCr 4:2:0, two rows of 1920 pixels | 24.4 µs | 12.7 µs | 2.04 µs | 1.12 µs | 0.750 µs |

The inverse DCT times include about 14.5 ns of filling the block the way the
decoder does; at the AVX-512 level the inverse DCT is the AVX2 code.

#### libjpeg-turbo for reference

On the same server, `tjbench` with Ubuntu's libjpeg-turbo 2.1.5 (the
`libturbojpeg0` package that
[`epyc8124p-tjbench.txt`](benchmarks/results/epyc8124p-tjbench.txt) records;
no libjpeg-turbo 3.x result is recorded; medians of three runs; the file
does not record CPU pinning) decompresses the 4:2:0 q75 960×1280 file to
planar YUV at 428 MP/s, where this package decodes it into an
`*image.YCbCr` at 417 MP/s at its AVX-512 level, and compresses a 960×1280
photo from planar 4:2:0 YUV at q75 at 429 MP/s, where `Encode` of the 4:2:0
`*image.YCbCr` runs at 458 MP/s. `tjbench` reuses its buffers; the Go
numbers include allocating the output image on every decode.

### Reproducing the measurements

The benchmarks live in [`benchmarks`](benchmarks), a separate Go module, so
that the library's module carries neither the harness nor its generated
inputs. The workloads are defined once in
[`benchmarks/suite`](benchmarks/suite/suite.go) and run through the adapters
`benchmarks/std` (`image/jpeg`) and `benchmarks/turbo` (this repository).

```bash
go install golang.org/x/perf/cmd/benchstat@latest

make bench-compare                                 # benchmarks/run.sh: the M5 Pro tables, about 20 minutes
make bench-compare ROUNDS=3 RSS_ROUNDS=1 SAT=0     # a quick look, about 5 minutes
BUILDS="std turbo-purego turbo-sse41 turbo-avx2 turbo-avx512" PIN=2 benchmarks/run.sh   # amd64 levels on one CPU (Linux)
```

The knobs are environment variables of run.sh, or make variables:

- `BUILDS`: the builds to compare, the first being the reference column
  (default `std turbo-purego turbo`); `turbo-LEVEL` is the default build
  with `JPEG_TURBO_LEVEL=LEVEL` (arm64 accepts only `turbo-go`, which still
  runs NEON).
- `ROUNDS` (10), `BENCHTIME` (1s), `SAT` (1; 0 skips the saturation
  benchmarks), `SATTIME` (2s), `RSS_ROUNDS` (5; 0 skips the peak-memory
  runs), `OUT` (`benchmarks/out`).
- `PIN`: on Linux, a `taskset -c` CPU list for the single-call and
  peak-memory runs, as the EPYC tables were measured (set the frequency
  governor yourself; env.txt records it).

`$OUT` then holds `benchstat.txt` (`sec/op`, `MP/s`, `cpu-sec/op`, `B/op`,
`allocs/op`, and `bytes`, the encoded size, the same in every column),
`benchstat-saturation.txt`, `rss.txt`, `env.txt` (the machine, Go, each
build's kernels, the inputs' SHA-256 sums) and `raw/`. Single benchmarks,
and the in-package benchmarks of the EPYC tables:

```bash
cd benchmarks
go test -run '^$' -bench '^BenchmarkDecode$/^img=420q75' -benchmem -count 6 ./std ./turbo
go test -run '^$' -bench '^BenchmarkEncode$/^img=RGBA$/' -benchmem -count 6 -tags purego ./turbo
go test -short ./...                 # TestSameAsStd, without the 12 MP workloads
go run ./turbo/rss -mode gen         # make the 12 MP inputs first, as run.sh does
go run ./turbo/rss -mode dec12       # one peak-memory measurement (or -mode enc12 -q 90)
cd ..

make bench                           # BenchmarkCompatDecode and BenchmarkCompatEncode in ./jpeg
mkdir -p benchmarks/out
go test -run '^$' -bench Compat -count 8 ./jpeg > benchmarks/out/compat.txt
benchstat -col /impl benchmarks/out/compat.txt    # std against turbo
JPEG_TURBO_LEVEL=sse41 make bench    # amd64: the kernels of one level
go test -run '^$' -bench . ./internal/dct ./internal/ycc    # the kernels
```

In the Compat benchmarks the `MB/s` column is megapixels per second, which
benchstat shows as `B/s` with binary prefixes (`84.63Mi` B/s is 88.7 MP/s).

## Tests

```bash
make test           # go test ./...: every test, in full
make test-purego    # the same on the portable code (-tags purego)
make race           # the race detector (the jpeg tests then run their -short sweeps)
make levels         # go test ./jpeg once per SIMD level this machine supports
make fuzz           # each fuzz target for FUZZTIME (default 60s)
make vet            # go vet: default, purego, both assembly sets, the examples, the benchmarks
make vet-ports      # go vet for linux/386, arm, riscv64, ppc64le, s390x, loong64, wasm, freebsd
make cover          # coverage of the whole module (CI measures it with -short)
go test -short ./...    # fewer inputs in the sweeps: about 15 s on the M5 Pro
```

`-short` keeps every kernel at every level but runs fewer inputs through the
comparisons. With it the tests cover 98.2% of the module's statements on the
M5 Pro (`go test -short -coverpkg=./...`; Go's coverage does not see inside
the assembly). The tests check:

- **The decoder against `image/jpeg`:** every file must give a `DeepEqual`
  image or the same error through several reader kinds (`bytes.Reader`,
  `iotest.OneByteReader`, `HalfReader`, `DataErrReader`, a 16-byte
  `bufio.Reader`), and also truncated at dozens of points (small files, and
  the bytes around the decoder's buffer boundaries, at every byte), mutated,
  and through readers that fail. The inputs are the standard library's test
  images, the benchmark photos and the 1196 files of
  [`testdata/corpus`](testdata/corpus). Every `Read` call's buffer length
  and result must match too, with short reads, `(0, nil)` reads, errors
  with data, and data after the end of the image.
- **The decoder's fast paths,** which `image/jpeg` does not have, against
  `image/jpeg`'s exact code: the entropy decoder at its buffer boundaries,
  with odd read sizes, byte stuffing and markers, on hand-built streams
  (malformed Huffman tables, EOB runs in baseline scans, excessive DC values,
  non-interleaved scans, tables redefined between scans) and on every 2-byte
  stream prefix for small tables; generated progressive streams with random
  scan scripts, compared pixel by pixel and coefficient by coefficient;
  coefficients and 16-bit quantization tables at the limits, where the
  arithmetic wraps around; headers that declare another size than the data
  holds.
- **The encoder against `image/jpeg`:** the bytes for every image type and
  layout (sub-images, odd sizes and origins, every subsampling and quality,
  images 65535 pixels wide), the panic values for malformed images and
  failing `At` methods, writers that fail or write short at many points, the
  `Flush`/`WriteByte` writer protocols, misuse (nil writers and images), and
  inputs next to inaccessible pages.
- **Memory, concurrency, registration:** `Decode` within 10% plus 64 KiB of
  `image/jpeg`'s allocation, also for files that declare a larger size than
  their data hold, and no allocation per `Encode` of an `*image.Gray`,
  `*image.YCbCr` (4:4:4, 4:2:2, 4:2:0, 4:1:0), `*image.RGBA`,
  `*image.NRGBA` or `*image.CMYK` into `io.Discard` or a `*bufio.Writer`
  once the pool holds an encoder (`TestEncAllocs`); 16 goroutines mixing
  every kind of call, re-entrant calls from callbacks, and switching the
  SIMD level between calls; the registration in a test binary that does
  not link `image/jpeg`
  ([`jpeg/internal/registertest`](jpeg/internal/registertest/register_test.go)).
- **The kernels** of `internal/dct` and `internal/ycc` at every level (see
  [Assembly](#assembly-and-the-purego-tag)), the benchmark workloads
  (`TestSameAsStd` in `benchmarks/turbo`), and `image/jpeg`'s own tests,
  copied with the code (`reader_test.go`, `writer_test.go`, `dct_test.go`,
  `fuzz_test.go`).

**Fuzz targets.** `make fuzz` runs `FuzzCompatDecode` and `FuzzDecCompatFast`
(mutated JPEG files against `image/jpeg`, the second with the fast entropy
decoder forced on), `FuzzCompatEncode` (image kind, size, quality and
pixels), `FuzzLevelKernels` (whose inputs are shaped to reach the paths
inside the assembly, which the fuzzer's coverage does not see) and
`FuzzDecProgGen`, which turns the fuzzer's bytes into progressive streams.
`make fuzz FUZZ=FuzzCompatDecode FUZZTIME=10m` runs one of them longer, and
`JPEG_TURBO_LEVEL` selects the kernels. `FuzzEncStrict`, `FuzzEncLayout`,
`FuzzDecode` (`image/jpeg`'s own) and `FuzzKernels` in `internal/ycc` run
with `go test -fuzz`.

**The corpus.** The 1196 files of `testdata/corpus` are made by
libjpeg-turbo's `cjpeg` and `jpegtran` from crops of the pictures in
`testdata`: every subsampling layout, baseline, progressive and restart
modes, optimized Huffman tables, grayscale and RGB, sequential files with a
scan per component, progressive files with random scan scripts, and the
variants that `image/jpeg` rejects (arithmetic coding, 12-bit, lossless,
unsupported sampling factors, and the restart and scan layouts it fails to
decode). `go run ./_tools/gencorpus` regenerates it, byte for byte with
libjpeg-turbo 3.2.0. It needs both `cjpeg` and `jpegtran` in `PATH`, and it
deletes `testdata/corpus` before it starts (`git checkout testdata/corpus`
restores it).

### CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push to
main and every pull request, with `GOFLAGS=-count=1`, so that no cached test
result stands in for kernels that the run's CPU did not execute.

- **test:** Linux amd64 (`ubuntu-latest`), Linux arm64 (`ubuntu-24.04-arm`,
  NEON) and macOS arm64 (`macos-latest`), each with Go 1.27 and the latest stable Go (both were Go 1.27.4 in the
  first run; the stable leg moves to each new release). The Linux legs check
  gofmt and that both go.mod files are tidy. Every leg builds with
  `CGO_ENABLED=0`, runs the vet steps of `make vet`, checks that the DCT
  assembly is what its generators write, runs the three examples and the
  benchmark module's `TestSameAsStd` (`-short`), then the tests and the
  `-tags purego` tests: in full on Linux, with `-short` on macOS (the same
  kernels on fewer inputs). The Linux amd64 leg with the stable Go
  also measures coverage and writes the total, a per-package table and the
  per-function list to the job summary.
- **SIMD levels:** `make levels` on `ubuntu-latest`, the `jpeg` tests once
  per level the runner's CPU supports.
- **race:** `go test -race -short`, default and `-tags purego` (the race
  detector does not see the assembly's memory accesses), on Linux amd64 and
  arm64.
- **other architectures:** the tests on linux/386 (`-short`, the portable
  code on a 32-bit target) and `make vet-ports`.
- **workflows:** [actionlint](https://github.com/rhysd/actionlint) and
  [zizmor](https://github.com/zizmorcore/zizmor) (pedantic, with its online
  audits) on the workflow and `.github/dependabot.yml`. Every action is
  pinned to a commit, and Dependabot keeps the pins and the Go modules
  current.

Windows is not in CI for now, because of a Go runtime bug
([golang/go#81238](https://github.com/golang/go/issues/81238)): on
windows/amd64 hosts with AVX-512 or AMX, recovering from a nil dereference
can overwrite memory below the goroutine's stack, and the tests that
compare such panics with `image/jpeg`'s then crash the test binary. The
library code is not involved, and `image/jpeg` is exposed in the same way.
The Windows legs come back once a Go release has the fix.

Which amd64 levels a run tests depends on the CPUs GitHub assigns, and the
jobs log the level they detect. In the first run
([45177341901](https://github.com/lEx0/go-jpeg-turbo/actions/runs/45177341901),
at eac4c8b, all 12 jobs passed), the Go 1.27 `ubuntu-latest` leg got an AMD
EPYC 7763 (up to AVX2), the stable-Go leg an Intel Xeon 6975P-C (AVX-512)
and the levels job an AMD EPYC 9V74, where it ran all four levels; the
Windows legs, which CI had then, got an EPYC 7763 and an EPYC 9V74 and
both reported AVX2 as their highest level. A run on CPUs without AVX-512 does not run the
AVX-512 kernels and checks them only statically: `TestKernelISA`
disassembles them on every leg, whatever the CPU, and audits their
instruction encodings, `TestGenerated` checks that those of `internal/dct`
are what their generator writes (those of `internal/ycc` are written by
hand), and `GOARCH=amd64 go vet` checks the arguments of every amd64 kernel
against its Go declaration.
Whole jobs took 9.5–12.5 minutes for the Linux amd64 test legs, 8 for Linux
arm64, 2–2.5 for macOS, 3 for Windows, 11.5 for the levels, 12.5 and 10.5
for race on amd64 and arm64, and 2.5 for the other architectures. The
guard-page tests need `mmap` and `mprotect` and run on Linux and macOS
only.

## Security

Please report vulnerabilities privately, through GitHub's
[private vulnerability reporting](https://github.com/lEx0/go-jpeg-turbo/security/advisories/new);
[SECURITY.md](SECURITY.md) says what is in scope and what to expect.

## License

MIT, see [LICENSE](LICENSE). The decoder and the encoder front end are
derived from the Go standard library's `image/jpeg` (Go 1.27, BSD-style
license), and `internal/dct` and `internal/imageutil` contain code copied
from it and from `image/internal/imageutil`; see [NOTICE](NOTICE).
