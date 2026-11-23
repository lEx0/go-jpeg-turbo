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

//go:build amd64 && !purego

package ycc

import "github.com/lEx0/go-jpeg-turbo/internal/cpuinfo"

// x86 kernels (ycc_amd64.s) for the SSE41, AVX2 and AVX512 levels, in
// blocks of 16, 32 and 64 pixels (outputs for downsample2x2). A kernel
// needs n >= its block (the 4:2:0 ones an even n) and never accesses
// memory outside the ranges stated below; the wrappers check every bound
// against len before the call, by reslicing with s[:k:len(s)] (a plain
// s[:k] is checked against cap), and convert rows shorter than the SSE4.1
// block with the portable code, and rows shorter than a higher level's
// block with the next lower level's kernels. The outputs must not overlap
// the inputs (the last block may overlap the previous one and is
// recomputed from the inputs).

// implName names the kernels in use, for test and benchmark names: those
// of kernelsOf(level), so avx2 at the AVX512 level on a machine without
// AVX512 VNNI (where Level still reports the level selected, avx512).
func implName() string { return kernelsOf(level).String() }

// hasAVX512Kernels reports whether the AVX512 level has kernels of its own
// on this machine: they also use AVX512 VNNI (VPDPWSSD), without which
// they lose most of their gain over the AVX2 kernels (see ycc_amd64.s),
// so that the AVX512 level runs the AVX2 kernels.
var hasAVX512Kernels = cpuinfo.X86.HasAVX512VNNI

// kernelsOf returns the level whose kernels the level l runs on rows long
// enough for them: l, or AVX2 for the AVX512 level without kernels of its
// own.
func kernelsOf(l cpuinfo.Level) cpuinfo.Level {
	if l >= cpuinfo.AVX512 && !hasAVX512Kernels {
		return cpuinfo.AVX2
	}
	return l
}

// kernelLevel returns the level whose kernels convert a row of n pixels:
// the highest one at or below level that has kernels and whose block n
// fills, or Go for the portable code.
func kernelLevel(n int) cpuinfo.Level {
	switch {
	case n >= 64 && level >= cpuinfo.AVX512 && hasAVX512Kernels:
		return cpuinfo.AVX512
	case n >= 32 && level >= cpuinfo.AVX2:
		return cpuinfo.AVX2
	case n >= 16 && level >= cpuinfo.SSE41:
		return cpuinfo.SSE41
	}
	return cpuinfo.Go
}

// rgbaToYCbCr* read rgba[:4n] and write y[:n], cb[:n] and cr[:n].
//
//go:noescape
func rgbaToYCbCrSSE41(rgba, y, cb, cr *byte, n int)

//go:noescape
func rgbaToYCbCrAVX2(rgba, y, cb, cr *byte, n int)

//go:noescape
func rgbaToYCbCrAVX512(rgba, y, cb, cr *byte, n int)

// nrgbaToYCbCr* read nrgba[:4n] and write y[:n], cb[:n] and cr[:n].
//
//go:noescape
func nrgbaToYCbCrSSE41(nrgba, y, cb, cr *byte, n int)

//go:noescape
func nrgbaToYCbCrAVX2(nrgba, y, cb, cr *byte, n int)

//go:noescape
func nrgbaToYCbCrAVX512(nrgba, y, cb, cr *byte, n int)

// downsample2x2* read r0[:2n] and r1[:2n] and write out[:n].
//
//go:noescape
func downsample2x2SSE41(r0, r1, out *byte, n int)

//go:noescape
func downsample2x2AVX2(r0, r1, out *byte, n int)

//go:noescape
func downsample2x2AVX512(r0, r1, out *byte, n int)

// rgbaToYCbCr420* (n even) read rgba0[:4n] and rgba1[:4n] and write
// y0[:n], y1[:n], cb[:n/2] and cr[:n/2].
//
//go:noescape
func rgbaToYCbCr420SSE41(rgba0, rgba1, y0, y1, cb, cr *byte, n int)

//go:noescape
func rgbaToYCbCr420AVX2(rgba0, rgba1, y0, y1, cb, cr *byte, n int)

//go:noescape
func rgbaToYCbCr420AVX512(rgba0, rgba1, y0, y1, cb, cr *byte, n int)

// nrgbaToYCbCr420* (n even) read nrgba0[:4n] and nrgba1[:4n] and write
// y0[:n], y1[:n], cb[:n/2] and cr[:n/2].
//
//go:noescape
func nrgbaToYCbCr420SSE41(nrgba0, nrgba1, y0, y1, cb, cr *byte, n int)

//go:noescape
func nrgbaToYCbCr420AVX2(nrgba0, nrgba1, y0, y1, cb, cr *byte, n int)

//go:noescape
func nrgbaToYCbCr420AVX512(nrgba0, nrgba1, y0, y1, cb, cr *byte, n int)

func rgbaToYCbCr(rgba, y, cb, cr []byte, n int) {
	l := kernelLevel(n)
	if l == cpuinfo.Go {
		rgbaToYCbCrGo(rgba, y, cb, cr, n)
		return
	}
	rgba = rgba[: 4*n : len(rgba)]
	y, cb, cr = y[:n:len(y)], cb[:n:len(cb)], cr[:n:len(cr)]
	switch l {
	case cpuinfo.AVX512:
		rgbaToYCbCrAVX512(&rgba[0], &y[0], &cb[0], &cr[0], n)
	case cpuinfo.AVX2:
		rgbaToYCbCrAVX2(&rgba[0], &y[0], &cb[0], &cr[0], n)
	default:
		rgbaToYCbCrSSE41(&rgba[0], &y[0], &cb[0], &cr[0], n)
	}
}

func nrgbaToYCbCr(nrgba, y, cb, cr []byte, n int) {
	l := kernelLevel(n)
	if l == cpuinfo.Go {
		nrgbaToYCbCrGo(nrgba, y, cb, cr, n)
		return
	}
	nrgba = nrgba[: 4*n : len(nrgba)]
	y, cb, cr = y[:n:len(y)], cb[:n:len(cb)], cr[:n:len(cr)]
	switch l {
	case cpuinfo.AVX512:
		nrgbaToYCbCrAVX512(&nrgba[0], &y[0], &cb[0], &cr[0], n)
	case cpuinfo.AVX2:
		nrgbaToYCbCrAVX2(&nrgba[0], &y[0], &cb[0], &cr[0], n)
	default:
		nrgbaToYCbCrSSE41(&nrgba[0], &y[0], &cb[0], &cr[0], n)
	}
}

func downsample2x2(r0, r1, out []byte, n int) {
	l := kernelLevel(n)
	if l == cpuinfo.Go {
		downsample2x2Go(r0, r1, out, n)
		return
	}
	r0, r1 = r0[:2*n:len(r0)], r1[:2*n:len(r1)]
	out = out[:n:len(out)]
	switch l {
	case cpuinfo.AVX512:
		downsample2x2AVX512(&r0[0], &r1[0], &out[0], n)
	case cpuinfo.AVX2:
		downsample2x2AVX2(&r0[0], &r1[0], &out[0], n)
	default:
		downsample2x2SSE41(&r0[0], &r1[0], &out[0], n)
	}
}

func rgbaToYCbCr420(rgba0, rgba1, y0, y1, cb, cr []byte, n int) {
	m := n &^ 1
	l := kernelLevel(m)
	if l == cpuinfo.Go {
		rgbaToYCbCr420Go(rgba0, rgba1, y0, y1, cb, cr, n)
		return
	}
	rgba0, rgba1 = rgba0[:4*n:len(rgba0)], rgba1[:4*n:len(rgba1)]
	y0, y1 = y0[:n:len(y0)], y1[:n:len(y1)]
	cb, cr = cb[:n/2:len(cb)], cr[:n/2:len(cr)]
	switch l {
	case cpuinfo.AVX512:
		rgbaToYCbCr420AVX512(&rgba0[0], &rgba1[0], &y0[0], &y1[0], &cb[0], &cr[0], m)
	case cpuinfo.AVX2:
		rgbaToYCbCr420AVX2(&rgba0[0], &rgba1[0], &y0[0], &y1[0], &cb[0], &cr[0], m)
	default:
		rgbaToYCbCr420SSE41(&rgba0[0], &rgba1[0], &y0[0], &y1[0], &cb[0], &cr[0], m)
	}
	if m != n {
		// The last pixel of an odd row has luma only.
		p, q := rgba0[4*m:], rgba1[4*m:]
		y0[m], _, _ = pixel(int32(p[0]), int32(p[1]), int32(p[2]))
		y1[m], _, _ = pixel(int32(q[0]), int32(q[1]), int32(q[2]))
	}
}

func nrgbaToYCbCr420(nrgba0, nrgba1, y0, y1, cb, cr []byte, n int) {
	m := n &^ 1
	l := kernelLevel(m)
	if l == cpuinfo.Go {
		nrgbaToYCbCr420Go(nrgba0, nrgba1, y0, y1, cb, cr, n)
		return
	}
	nrgba0, nrgba1 = nrgba0[:4*n:len(nrgba0)], nrgba1[:4*n:len(nrgba1)]
	y0, y1 = y0[:n:len(y0)], y1[:n:len(y1)]
	cb, cr = cb[:n/2:len(cb)], cr[:n/2:len(cr)]
	switch l {
	case cpuinfo.AVX512:
		nrgbaToYCbCr420AVX512(&nrgba0[0], &nrgba1[0], &y0[0], &y1[0], &cb[0], &cr[0], m)
	case cpuinfo.AVX2:
		nrgbaToYCbCr420AVX2(&nrgba0[0], &nrgba1[0], &y0[0], &y1[0], &cb[0], &cr[0], m)
	default:
		nrgbaToYCbCr420SSE41(&nrgba0[0], &nrgba1[0], &y0[0], &y1[0], &cb[0], &cr[0], m)
	}
	if m != n {
		y0[m], _, _ = pixelN(nrgba0[4*m:])
		y1[m], _, _ = pixelN(nrgba1[4*m:])
	}
}
