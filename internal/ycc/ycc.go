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

// Package ycc holds the encoder's color conversion and chroma downsampling
// kernels. Every function must produce exactly what image/jpeg's encoder
// computes per pixel (color.RGBToYCbCr on the 8-bit samples it takes from
// the image, and the 2x2 rounded average of its scale function).
//
// The *Ref functions are the specification; the exported functions without
// the suffix are what the encoder calls and may be implemented in assembly.
// A slice holds what n requires only if its len does: the functions panic
// on a shorter slice whatever its cap.
package ycc

import "image/color"

// RGBAToYCbCrRef converts n pixels of rgba (4 bytes per pixel, R G B A;
// A is ignored, as image/jpeg's rgbaToYCbCr ignores it) with
// color.RGBToYCbCr into y[:n], cb[:n] and cr[:n].
func RGBAToYCbCrRef(rgba, y, cb, cr []byte, n int) {
	for i := 0; i < n; i++ {
		p := rgba[4*i : 4*i+4 : len(rgba)] // checked against len, not cap
		y[i], cb[i], cr[i] = color.RGBToYCbCr(p[0], p[1], p[2])
	}
}

// NRGBAToYCbCrRef converts n pixels of nrgba (an *image.NRGBA row: 4 bytes
// per pixel, non-premultiplied) exactly as image/jpeg's generic toYCbCr
// path does for an *image.NRGBA: color.NRGBA.RGBA() (which premultiplies
// by alpha in 16 bits), the top 8 bits of each channel, then
// color.RGBToYCbCr.
func NRGBAToYCbCrRef(nrgba, y, cb, cr []byte, n int) {
	for i := 0; i < n; i++ {
		p := nrgba[4*i : 4*i+4 : len(nrgba)] // checked against len, not cap
		r, g, b, _ := color.NRGBA{p[0], p[1], p[2], p[3]}.RGBA()
		y[i], cb[i], cr[i] = color.RGBToYCbCr(uint8(r>>8), uint8(g>>8), uint8(b>>8))
	}
}

// Downsample2x2Ref averages 2x2 squares as image/jpeg's scale does:
// out[i] = (r0[2i] + r0[2i+1] + r1[2i] + r1[2i+1] + 2) >> 2 for i < n.
func Downsample2x2Ref(r0, r1, out []byte, n int) {
	for i := 0; i < n; i++ {
		s := uint32(r0[2*i]) + uint32(r0[2*i+1]) + uint32(r1[2*i]) + uint32(r1[2*i+1])
		out[i] = uint8((s + 2) >> 2)
	}
}

// RGBAToYCbCr420Ref converts two rows of n pixels (n even) of rgba (as in
// RGBAToYCbCrRef) to full-resolution luma y0[:n], y1[:n] and to the 2x2
// averaged chroma cb[:n/2], cr[:n/2] (Downsample2x2Ref of the per-pixel
// chroma of both rows).
func RGBAToYCbCr420Ref(rgba0, rgba1, y0, y1, cb, cr []byte, n int) {
	cb0 := make([]byte, n)
	cr0 := make([]byte, n)
	cb1 := make([]byte, n)
	cr1 := make([]byte, n)
	RGBAToYCbCrRef(rgba0, y0, cb0, cr0, n)
	RGBAToYCbCrRef(rgba1, y1, cb1, cr1, n)
	Downsample2x2Ref(cb0, cb1, cb, n/2)
	Downsample2x2Ref(cr0, cr1, cr, n/2)
}

// NRGBAToYCbCr420Ref is RGBAToYCbCr420Ref for *image.NRGBA rows (see
// NRGBAToYCbCrRef).
func NRGBAToYCbCr420Ref(nrgba0, nrgba1, y0, y1, cb, cr []byte, n int) {
	cb0 := make([]byte, n)
	cr0 := make([]byte, n)
	cb1 := make([]byte, n)
	cr1 := make([]byte, n)
	NRGBAToYCbCrRef(nrgba0, y0, cb0, cr0, n)
	NRGBAToYCbCrRef(nrgba1, y1, cb1, cr1, n)
	Downsample2x2Ref(cb0, cb1, cb, n/2)
	Downsample2x2Ref(cr0, cr1, cr, n/2)
}
