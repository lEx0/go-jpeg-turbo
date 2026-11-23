// The MIT License (MIT)
//
// Copyright (c) 2026 Amangeldy Kadyl
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// Package turbo adapts this repository's jpeg package to the benchmark
// suite. Build with -tags purego for the portable Go code that the other
// architectures run. On amd64 the environment variable JPEG_TURBO_LEVEL
// (go, sse41, avx2 or avx512) selects the kernels of a lower SIMD level
// than the CPU's best, as it does for the jpeg package's tests; see
// level.go.
package turbo

import (
	"fmt"
	"image"
	"io"
	"os"
	"runtime"

	"github.com/lEx0/go-jpeg-turbo/benchmarks/suite"
	"github.com/lEx0/go-jpeg-turbo/internal/cpuinfo"
	"github.com/lEx0/go-jpeg-turbo/internal/dct"
	"github.com/lEx0/go-jpeg-turbo/jpeg"
)

// Codec is the adapter of this repository.
type Codec struct{}

var _ suite.Codec = Codec{}

func (Codec) Decode(r io.Reader) (image.Image, error) { return jpeg.Decode(r) }

func (Codec) DecodeConfig(r io.Reader) (image.Config, error) { return jpeg.DecodeConfig(r) }

func (Codec) Encode(w io.Writer, m image.Image, quality int) error {
	return jpeg.Encode(w, m, &jpeg.Options{Quality: quality})
}

// Describe names the kernels that run, which the build and, on amd64, the
// CPU and JPEG_TURBO_LEVEL decide.
func (Codec) Describe() string {
	const name = "github.com/lEx0/go-jpeg-turbo"
	switch {
	case purego:
		return name + " built with -tags purego: portable Go kernels"
	case runtime.GOARCH == "arm64":
		return name + ": NEON kernels"
	case runtime.GOARCH != "amd64":
		return name + ": portable Go kernels (no assembly for " + runtime.GOARCH + ")"
	}
	s := fmt.Sprintf("%s: %v kernels", name, dct.Level())
	switch {
	case dct.Level() == cpuinfo.Go:
		s = name + ": portable Go kernels"
	case dct.Level() == cpuinfo.AVX512 && !cpuinfo.X86.HasAVX512VNNI:
		// internal/ycc's AVX-512 kernels need VNNI (ycc_amd64.go).
		s += " (color conversion avx2: the CPU has no AVX-512 VNNI)"
	}
	return fmt.Sprintf("%s; %s=%q, the CPU has up to %v", s, levelEnv, os.Getenv(levelEnv), cpuinfo.Max)
}
