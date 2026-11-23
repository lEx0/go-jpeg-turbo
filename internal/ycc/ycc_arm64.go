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

//go:build arm64 && !purego

package ycc

// NEON kernels (ycc_arm64.s). They need n >= 16 (the 4:2:0 ones an even
// n) and never access memory outside the ranges stated below; the
// wrappers check every bound against len before the call, by reslicing
// with s[:k:len(s)] (a plain s[:k] is checked against cap), and convert
// shorter rows with the portable code. The outputs must not overlap the
// inputs (the last 16-pixel block may overlap the previous one and is
// recomputed from the inputs).

// implName names the kernels in use, for test and benchmark names.
func implName() string { return "neon" }

// rgbaToYCbCrNEON reads rgba[:4n] and writes y[:n], cb[:n] and cr[:n].
//
//go:noescape
func rgbaToYCbCrNEON(rgba, y, cb, cr *byte, n int)

// nrgbaToYCbCrNEON reads nrgba[:4n] and writes y[:n], cb[:n] and cr[:n].
//
//go:noescape
func nrgbaToYCbCrNEON(nrgba, y, cb, cr *byte, n int)

// downsample2x2NEON reads r0[:2n] and r1[:2n] and writes out[:n].
//
//go:noescape
func downsample2x2NEON(r0, r1, out *byte, n int)

// rgbaToYCbCr420NEON (n even) reads rgba0[:4n] and rgba1[:4n] and writes
// y0[:n], y1[:n], cb[:n/2] and cr[:n/2].
//
//go:noescape
func rgbaToYCbCr420NEON(rgba0, rgba1, y0, y1, cb, cr *byte, n int)

// nrgbaToYCbCr420NEON (n even) reads nrgba0[:4n] and nrgba1[:4n] and
// writes y0[:n], y1[:n], cb[:n/2] and cr[:n/2].
//
//go:noescape
func nrgbaToYCbCr420NEON(nrgba0, nrgba1, y0, y1, cb, cr *byte, n int)

// minNEON is the shortest row the kernels take.
const minNEON = 16

func rgbaToYCbCr(rgba, y, cb, cr []byte, n int) {
	if n < minNEON {
		rgbaToYCbCrGo(rgba, y, cb, cr, n)
		return
	}
	rgba = rgba[: 4*n : len(rgba)]
	y, cb, cr = y[:n:len(y)], cb[:n:len(cb)], cr[:n:len(cr)]
	rgbaToYCbCrNEON(&rgba[0], &y[0], &cb[0], &cr[0], n)
}

func nrgbaToYCbCr(nrgba, y, cb, cr []byte, n int) {
	if n < minNEON {
		nrgbaToYCbCrGo(nrgba, y, cb, cr, n)
		return
	}
	nrgba = nrgba[: 4*n : len(nrgba)]
	y, cb, cr = y[:n:len(y)], cb[:n:len(cb)], cr[:n:len(cr)]
	nrgbaToYCbCrNEON(&nrgba[0], &y[0], &cb[0], &cr[0], n)
}

func downsample2x2(r0, r1, out []byte, n int) {
	if n < minNEON {
		downsample2x2Go(r0, r1, out, n)
		return
	}
	r0, r1 = r0[:2*n:len(r0)], r1[:2*n:len(r1)]
	out = out[:n:len(out)]
	downsample2x2NEON(&r0[0], &r1[0], &out[0], n)
}

func rgbaToYCbCr420(rgba0, rgba1, y0, y1, cb, cr []byte, n int) {
	if n < minNEON { // n &^ 1 >= 16 below
		rgbaToYCbCr420Go(rgba0, rgba1, y0, y1, cb, cr, n)
		return
	}
	rgba0, rgba1 = rgba0[:4*n:len(rgba0)], rgba1[:4*n:len(rgba1)]
	y0, y1 = y0[:n:len(y0)], y1[:n:len(y1)]
	cb, cr = cb[:n/2:len(cb)], cr[:n/2:len(cr)]
	m := n &^ 1
	rgbaToYCbCr420NEON(&rgba0[0], &rgba1[0], &y0[0], &y1[0], &cb[0], &cr[0], m)
	if m != n {
		// The last pixel of an odd row has luma only.
		p, q := rgba0[4*m:], rgba1[4*m:]
		y0[m], _, _ = pixel(int32(p[0]), int32(p[1]), int32(p[2]))
		y1[m], _, _ = pixel(int32(q[0]), int32(q[1]), int32(q[2]))
	}
}

func nrgbaToYCbCr420(nrgba0, nrgba1, y0, y1, cb, cr []byte, n int) {
	if n < minNEON { // n &^ 1 >= 16 below
		nrgbaToYCbCr420Go(nrgba0, nrgba1, y0, y1, cb, cr, n)
		return
	}
	nrgba0, nrgba1 = nrgba0[:4*n:len(nrgba0)], nrgba1[:4*n:len(nrgba1)]
	y0, y1 = y0[:n:len(y0)], y1[:n:len(y1)]
	cb, cr = cb[:n/2:len(cb)], cr[:n/2:len(cr)]
	m := n &^ 1
	nrgbaToYCbCr420NEON(&nrgba0[0], &nrgba1[0], &y0[0], &y1[0], &cb[0], &cr[0], m)
	if m != n {
		p, q := nrgba0[4*m:], nrgba1[4*m:]
		mp, mq := premulMul(p[3]), premulMul(q[3])
		y0[m], _, _ = pixel(premul8(p[0], mp), premul8(p[1], mp), premul8(p[2], mp))
		y1[m], _, _ = pixel(premul8(q[0], mq), premul8(q[1], mq), premul8(q[2], mq))
	}
}
