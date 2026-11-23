# go-jpeg-turbo

A drop-in replacement for Go's `image/jpeg` that decodes about 4.5x faster
(2.7x for progressive files) and encodes 8–17x faster. It is pure Go with Go assembly:
**no cgo**, and no libjpeg. The only dependency outside the standard library
is `golang.org/x/sys`, used to detect CPU features. On arm64 and amd64 the
hot loops run as SIMD assembly: NEON on arm64, and on amd64 SSE4.1, AVX2 or
AVX-512, picked at run time. Every other architecture, and builds with
`-tags purego`, use portable Go code that gives the same results.

- **Same API.** `Decode`, `DecodeConfig`, `Encode`, `Options`,
  `FormatError`, `UnsupportedError`, `Reader` and `DefaultQuality`, with the
  same signatures. To switch, change the import path and nothing else.
- **Same results as `image/jpeg` from Go 1.27:**
  - decoded images are `reflect.DeepEqual` to the standard library's,
    including the plane padding past the image edge;
  - `Encode` writes the same bytes;
  - errors and panics have the same values in the same cases;
  - `DecodeConfig` returns the same configs.
- Baseline and progressive JPEG, restart intervals, every subsampling
  `image/jpeg` accepts, grayscale, RGB, CMYK and YCCK.
- Works wherever Go does, including `CGO_ENABLED=0`, cross compilation and
  scratch or distroless containers.

## Install

```bash
go get github.com/lEx0/go-jpeg-turbo
```

Requires Go 1.27 or newer.

```go
import (
	"image"

	"github.com/lEx0/go-jpeg-turbo/jpeg" // was: "image/jpeg"
)

img, err := jpeg.Decode(r)
err = jpeg.Encode(w, img, &jpeg.Options{Quality: 90})

// image.Decode finds the format through the "jpeg" registration of this
// package, so a blank import is enough for code that only uses image.Decode:
//	_ "github.com/lEx0/go-jpeg-turbo/jpeg"
```

## Compatibility

The tests compare the package with `image/jpeg` from the Go toolchain it is
built with. The code follows Go 1.27; a later Go release that changes
`image/jpeg` can make results differ until this package catches up.

A few things can be observed differently. None of them changes the decoded
pixels or the encoded bytes:

- **Write calls.** Like `image/jpeg`, `Encode` wraps a plain `io.Writer` in
  a 4096-byte buffer and makes exactly the same `Write` calls, and it leaves
  a `*bufio.Writer` in the same state. One case differs: a writer that is
  not a `*bufio.Writer` but has its own `Flush` and `WriteByte` methods.
  `image/jpeg` writes the scan to it one `WriteByte` call at a time;
  `Encode` writes the same bytes with `Write` calls of up to 64 KiB.
- **Panics while encoding.** If `At` or the writer panics in the middle of
  the image, fewer bytes may have reached the writer before the panic,
  because output is buffered in larger chunks. The panic value is the
  same.
- **`At` calls.** For image types without a fast path (anything other than
  `*image.RGBA`, `*image.NRGBA`, `*image.YCbCr`, `*image.Gray` and a few
  more), `At` is called in the order of `image/jpeg`'s first read of each
  pixel. The number of calls can differ: `image/jpeg` reads the edge pixels
  of partial blocks again, this package does not.
- **Two registrations.** If a program links both `image/jpeg` and this
  package, both register the "jpeg" format, and `image.Decode` uses
  whichever was registered first. Errors are this package's
  `jpeg.FormatError` and `jpeg.UnsupportedError`, which are distinct types
  from the standard library's. A type assertion or `errors.As` against
  `image/jpeg.FormatError` does not match them.

As with `image/jpeg`, the slice given to `Write` is the encoder's buffer
and is only valid during the call. Here that buffer is pooled and reused by
later `Encode` calls, so a writer must not keep it (this is the `io.Writer`
contract anyway).

## Assembly and the `purego` tag

| Kernel | arm64 | amd64 SSE4.1 | amd64 AVX2 | amd64 AVX-512 |
|---|---|---|---|---|
| Dequantize + IDCT + store | NEON | SSE4.1 | AVX2 | AVX2 (zmm gave no gain) |
| FDCT + quantize + zig-zag bitmap | NEON | SSE4.1 | AVX2 | AVX-512 BW |
| RGBA/NRGBA → YCbCr 4:4:4 / 4:2:0, chroma downsampling | NEON | SSE4.1 | AVX2 | AVX-512 VNNI (AVX2 without VNNI) |

- On amd64 the best level the CPU and the OS support is chosen once at
  startup, through `golang.org/x/sys/cpu`. Building with `GOAMD64=v3` or
  `v4` is not needed. It makes the Go code around the kernels about 5%
  faster when encoding.
- Every kernel has a scalar Go reference, and the tests check each level
  against it, exhaustively where the input space allows. Examples:
  - the quantizer on every coefficient value and every quantizer;
  - the color conversion on all 2^24 RGB values and all 2^32 NRGBA values;
  - the kernels against guard pages and unaligned buffers.
- `go build -tags purego` drops the assembly everywhere. The portable code
  (SWAR color conversion and a restructured integer DCT, with the same
  results) is still about 2–4x faster than `image/jpeg`.
- Tests can force a level: `JPEG_TURBO_LEVEL=go|sse41|avx2|avx512 go test
  ./jpeg`. `make levels` runs the package tests once at every level the
  machine supports.

## Benchmarks

Megapixels per second, single thread, median of 8 or more runs, measured by
`BenchmarkCompatDecode` and `BenchmarkCompatEncode` in `jpeg/`. The decoded
images are photos from `testdata/`. The encode input is a 960×1280 photo,
converted to each image type.

### Apple M5 Pro (arm64)

| | `image/jpeg` | NEON | speed-up | purego |
|---|---:|---:|---:|---:|
| Decode 4:2:0 q75, 960×1280 | 187 | 883 | 4.7x | 445 |
| Decode 4:4:4 q92, 960×1280 | 88 | 418 | 4.8x | 222 |
| Decode 4:2:2 q92, restart markers | 113 | 500 | 4.4x | 283 |
| Decode 4:2:0 q85, progressive | 94 | 265 | 2.8x | 208 |
| Decode 4:2:0 q85, 1920×2560 | 203 | 905 | 4.5x | 462 |
| Decode gray q85, 1920×2560 | 270 | 1175 | 4.4x | 599 |
| Encode `*image.RGBA` q75 | 90 | 834 | 9.3x | 307 |
| Encode `*image.RGBA` q90 | 77 | 648 | 8.4x | 278 |
| Encode `*image.NRGBA` q75 | 55 | 766 | 13.9x | 256 |
| Encode `*image.YCbCr` 4:2:0 q75 | 101 | 1002 | 9.9x | 436 |
| Encode `*image.YCbCr` 4:4:4 q75 | 94 | 978 | 10.4x | 414 |
| Encode `*image.Gray` q75 | 147 | 1322 | 9.0x | 610 |

### AMD EPYC 8124P (amd64, Zen 4c)

Pinned to one CPU with `taskset`, with the `performance` governor.

| | `image/jpeg` | SSE4.1 | AVX2 | AVX-512 | speed-up | purego |
|---|---:|---:|---:|---:|---:|---:|
| Decode 4:2:0 q75, 960×1280 | 89 | 372 | 420 | 417 | 4.7x | 181 |
| Decode 4:4:4 q92, 960×1280 | 42 | 175 | 195 | 195 | 4.6x | 90 |
| Decode 4:2:2 q92, restart markers | 57 | 221 | 245 | 245 | 4.3x | 116 |
| Decode 4:2:0 q85, progressive | 45 | 119 | 123 | 122 | 2.7x | 86 |
| Decode 4:2:0 q85, 1920×2560 | 95 | 388 | 434 | 439 | 4.6x | 185 |
| Decode gray q85, 1920×2560 | 129 | 517 | 583 | 585 | 4.5x | 250 |
| Encode `*image.RGBA` q75 | 40 | 309 | 401 | 425 | 10.5x | 101 |
| Encode `*image.RGBA` q90 | 35 | 240 | 292 | 305 | 8.6x | 92 |
| Encode `*image.NRGBA` q75 | 24 | 304 | 397 | 423 | 17.6x | 76 |
| Encode `*image.YCbCr` 4:2:0 q75 | 45 | 364 | 447 | 458 | 10.1x | 150 |
| Encode `*image.YCbCr` 4:4:4 q75 | 44 | 356 | 439 | 450 | 10.3x | 143 |
| Encode `*image.Gray` q75 | 69 | 465 | 558 | 572 | 8.3x | 211 |

For reference, libjpeg-turbo 3.2.0's own `tjbench` decodes the first file to
planar YUV at 862 MP/s on the M5 Pro and 418 MP/s on the EPYC. It reuses
its buffers; the Go numbers include allocating the output image on every
call.

The raw outputs, the per-kernel benchmarks and the benchstat tables are in
[`benchmarks/results`](benchmarks/results). To reproduce:

```bash
go test -run '^$' -bench Compat -count 10 ./jpeg > new.txt
benchstat -col /impl new.txt      # std vs turbo; MB/s is megapixels/s
```

## Tests

`make test` runs everything; `make race`, `make test-purego`, `make levels`,
`make fuzz` and `make vet` cover the variants. The suite includes:

- **Decoder against `image/jpeg`.** Each file must give a `DeepEqual`
  image, error or panic, through several reader kinds (`bytes.Reader`,
  `iotest.OneByteReader`, `HalfReader`, `DataErrReader`, a small
  `bufio.Reader`). The inputs:
  - the standard library's test files;
  - 1120 files made by libjpeg-turbo covering every subsampling, progressive
    and restart layout, optimized Huffman tables and the variants that must
    be rejected (`_tools/gencorpus`);
  - every truncation and many mutations of each.
- **Encoder against `image/jpeg`.** Byte comparison for every image type and
  layout: sub-images, odd sizes, all subsamplings and qualities, malformed
  images (panic values compared), failing writers, and the
  `Flush`/`WriteByte` writer protocols.
- **Fuzz targets.** `FuzzCompatDecode`, `FuzzDecCompatFast`,
  `FuzzCompatEncode`, `FuzzLevelKernels` (whose inputs are shaped to reach
  the paths inside the assembly) and `FuzzDecProgGen`, which turns the
  fuzzer's bytes into progressive streams.
- **Kernels.** Exhaustive and boundary tests for every SIMD level against
  the scalar references.

## License

MIT, see [LICENSE](LICENSE). The decoder and the encoder front end are
derived from the Go standard library's `image/jpeg` (BSD-style license);
see [NOTICE](NOTICE).
