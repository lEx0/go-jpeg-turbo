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

// The functions the encoder calls. Each produces exactly what its *Ref
// specification in ycc.go produces, without allocating. The implementation
// is NEON assembly on arm64 (ycc_arm64.go, ycc_arm64.s), SSE4.1, AVX2 or
// AVX-512 assembly on amd64 as Level selects (ycc_amd64.go, ycc_amd64.s)
// and portable Go elsewhere or with -tags purego (ycc_generic.go); all
// share the Go code in ycc_go.go for rows too short for a vector block
// (and amd64 for the Go level).
//
// Differences from the *Ref functions, none of which the encoder relies
// on:
//   - n <= 0 does nothing (RGBAToYCbCr420Ref panics for n < 0);
//   - if a slice is shorter than n requires, the function panics before
//     it writes anything (the Ref panics part way). Only len counts, as
//     for the Ref: a slice with len < n <= cap panics too, so that a row
//     cut from a larger plane is never written (or read) past its end;
//   - aliasing is not supported: no output may overlap an input or another
//     output (the vector code may recompute and rewrite the last block
//     from the inputs). Inputs may overlap each other.
//
// An odd n in the 4:2:0 functions is handled as by the Ref: n luma samples
// per row and n/2 chroma samples.

// RGBAToYCbCr: see RGBAToYCbCrRef.
func RGBAToYCbCr(rgba, y, cb, cr []byte, n int) { rgbaToYCbCr(rgba, y, cb, cr, n) }

// NRGBAToYCbCr: see NRGBAToYCbCrRef.
func NRGBAToYCbCr(nrgba, y, cb, cr []byte, n int) { nrgbaToYCbCr(nrgba, y, cb, cr, n) }

// Downsample2x2: see Downsample2x2Ref.
func Downsample2x2(r0, r1, out []byte, n int) { downsample2x2(r0, r1, out, n) }

// RGBAToYCbCr420: see RGBAToYCbCr420Ref.
func RGBAToYCbCr420(rgba0, rgba1, y0, y1, cb, cr []byte, n int) {
	rgbaToYCbCr420(rgba0, rgba1, y0, y1, cb, cr, n)
}

// NRGBAToYCbCr420: see NRGBAToYCbCr420Ref.
func NRGBAToYCbCr420(nrgba0, nrgba1, y0, y1, cb, cr []byte, n int) {
	nrgbaToYCbCr420(nrgba0, nrgba1, y0, y1, cb, cr, n)
}
