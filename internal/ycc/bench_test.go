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

package ycc

import (
	"math/rand/v2"
	"testing"
)

// Kernel throughput: impl=ref is the *Ref specification, impl=go the
// portable code (what -tags purego runs), and the exported function at
// every level with kernels: impl=sse41, avx2 and avx512 on amd64, neon on
// arm64. MB/s reads as megapixels per second, counting source pixels (both
// rows for the 4:2:0 kernels, all four samples of a square for
// Downsample2x2).

const benchWidth = 1920

func benchRows(n int) (src0, src1, y0, y1, cb, cr []byte) {
	r := rand.New(rand.NewPCG(10, 11))
	src0, src1 = make([]byte, 4*n), make([]byte, 4*n)
	fill(r, src0, 0, true)
	fill(r, src1, 2, true)
	return src0, src1, make([]byte, n), make([]byte, n), make([]byte, n), make([]byte, n)
}

// benchImpls runs run on the reference, the portable code and the
// exported function at every level of testLevels whose kernels are not
// the portable code.
func benchImpls[F any](b *testing.B, ref, portable, exported F, run func(b *testing.B, fn F)) {
	b.Run("impl=ref", func(b *testing.B) { run(b, ref) })
	b.Run("impl=go", func(b *testing.B) { run(b, portable) })
	for _, l := range testLevels() {
		func() {
			defer SetLevel(SetLevel(l))
			if name := implName(); name != "go" {
				b.Run("impl="+name, func(b *testing.B) { run(b, exported) })
			}
		}()
	}
}

func BenchmarkRGBAToYCbCr(b *testing.B) {
	benchRowKernel(b, rowKernels[0])
}

func BenchmarkNRGBAToYCbCr(b *testing.B) {
	benchRowKernel(b, rowKernels[1])
}

// BenchmarkNRGBAToYCbCrOpaque converts opaque NRGBA pixels (alpha 255),
// which the vector kernels convert as RGBA.
func BenchmarkNRGBAToYCbCrOpaque(b *testing.B) {
	benchRowKernel(b, rowKernels[1], opaqueRows)
}

// opaqueRows sets the alpha of the RGBA pixels of rows to 255.
func opaqueRows(rows ...[]byte) {
	for _, r := range rows {
		for i := 3; i < len(r); i += 4 {
			r[i] = 255
		}
	}
}

func benchRowKernel(b *testing.B, k rowKernel, prep ...func(rows ...[]byte)) {
	n := benchWidth
	src, _, y, _, cb, cr := benchRows(n)
	for _, f := range prep {
		f(src)
	}
	benchImpls(b, k.ref, k.portable, k.fn, func(b *testing.B, fn func(src, y, cb, cr []byte, n int)) {
		b.SetBytes(int64(n))
		for b.Loop() {
			fn(src, y, cb, cr, n)
		}
	})
}

func BenchmarkRGBAToYCbCr420(b *testing.B) {
	benchRows420Kernel(b, rows420Kernels[0])
}

func BenchmarkNRGBAToYCbCr420(b *testing.B) {
	benchRows420Kernel(b, rows420Kernels[1])
}

// BenchmarkNRGBAToYCbCr420Opaque is BenchmarkNRGBAToYCbCrOpaque for the
// 4:2:0 kernel.
func BenchmarkNRGBAToYCbCr420Opaque(b *testing.B) {
	benchRows420Kernel(b, rows420Kernels[1], opaqueRows)
}

func benchRows420Kernel(b *testing.B, k rows420Kernel, prep ...func(rows ...[]byte)) {
	n := benchWidth
	src0, src1, y0, y1, cb, cr := benchRows(n)
	for _, f := range prep {
		f(src0, src1)
	}
	benchImpls(b, k.ref, k.portable, k.fn, func(b *testing.B, fn func(src0, src1, y0, y1, cb, cr []byte, n int)) {
		b.SetBytes(int64(2 * n))
		for b.Loop() {
			fn(src0, src1, y0, y1, cb, cr, n)
		}
	})
}

func BenchmarkDownsample2x2(b *testing.B) {
	n := benchWidth / 2
	src0, src1, out, _, _, _ := benchRows(n)
	benchImpls(b, Downsample2x2Ref, downsample2x2Go, Downsample2x2, func(b *testing.B, fn func(r0, r1, out []byte, n int)) {
		b.SetBytes(int64(4 * n))
		for b.Loop() {
			fn(src0, src1, out, n)
		}
	})
}
