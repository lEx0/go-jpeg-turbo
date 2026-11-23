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

// Package dct holds the 8x8 DCT kernels of the codec: the forward and
// inverse transforms of image/jpeg (Go 1.27, dct.go, copied verbatim) and
// the fused operations built on them that the decoder and the encoder call
// per block. Every function here must produce exactly the bits that the
// image/jpeg code it replaces produces, for every input, including inputs
// whose int32 arithmetic wraps around.
//
// The *Ref functions are the specification: straightforward Go written the
// way image/jpeg writes it. The exported functions without the suffix are
// what the codec calls; they may be implemented in assembly and are tested
// against the *Ref functions.
package dct

// Unzig maps zig-zag order to natural (row-major) order:
// Unzig[zig] is the natural index of the zig'th coefficient (image/jpeg's
// unzig table).
var Unzig = [BlockSize]int{
	0, 1, 8, 16, 9, 2, 3, 10,
	17, 24, 32, 25, 18, 11, 4, 5,
	12, 19, 26, 33, 40, 48, 41, 34,
	27, 20, 13, 6, 7, 14, 21, 28,
	35, 42, 49, 56, 57, 50, 43, 36,
	29, 22, 15, 23, 30, 37, 44, 51,
	58, 59, 52, 45, 38, 31, 39, 46,
	53, 60, 61, 54, 47, 55, 62, 63,
}

// DequantIDCTStoreRef is image/jpeg's reconstructBlock for the common
// (non-flex) layout. qt is the quantization table in natural order
// (qt[Unzig[zig]] = table[zig]):
//
//	b[i] *= qt[i] for every i (int32, wrapping)
//	IDCT(b)
//	dst[y*stride+x] = uint8(max(0, min(255, b[8*y+x]+128))) for x, y in [0, 8)
//
// The contents of b are unspecified afterwards. dst must hold at least
// 7*stride+8 bytes.
func DequantIDCTStoreRef(b *Block, qt *Block, dst []byte, stride int) {
	for i := range b {
		b[i] *= qt[i]
	}
	IDCT(b)
	for y := 0; y < 8; y++ {
		y8 := y * 8
		yStride := y * stride
		for x := 0; x < 8; x++ {
			dst[yStride+x] = uint8(max(0, min(255, b[y8+x]+128)))
		}
	}
}

// DequantIDCTRef multiplies b by qt (natural order, int32, wrapping) and
// runs the inverse DCT in place; the decoder's flex-subsampling path then
// expands the samples itself.
func DequantIDCTRef(b *Block, qt *Block) {
	for i := range b {
		b[i] *= qt[i]
	}
	IDCT(b)
}

// Quant is an encoder quantization table, prepared for FDCTQuant.
// The fields after Q are private to the implementation (reciprocals and
// the like); Init fills all of them.
type Quant struct {
	// Q is the table in zig-zag order, as written to the DQT segment.
	Q [BlockSize]byte
	// Div[zig] = 8 * Q[zig], the divisor image/jpeg's writeBlock uses.
	Div [BlockSize]int32

	// ok reports that init accepted Q (no zero entry); otherwise FDCTQuant
	// runs FDCTQuantRef, which divides by zero like image/jpeg.
	ok bool
	// qm and qc: the rounded division of the portable kernel (see quantGo).
	qm, qc [BlockSize]int64
	// vtab is the table of the NEON kernel, in its own layout (initVec in
	// dct_arm64.go).
	vtab [3 * BlockSize]int16
	// xtab holds the tables of the x86 kernels, in their two layouts
	// (initVec in dct_amd64.go).
	xtab [2][4 * BlockSize]int16
}

// Init prepares t for the table q (zig-zag order, every entry in [1, 255]).
func (t *Quant) Init(q *[BlockSize]byte) {
	t.Q = *q
	for i, v := range q {
		t.Div[i] = 8 * int32(v)
	}
	t.init()
}

// div is image/jpeg's rounded division (writer.go).
func div(a, b int32) int32 {
	if a >= 0 {
		return (a + (b >> 1)) / b
	}
	return -((-a + (b >> 1)) / b)
}

// FDCTQuantRef is the transform and quantization part of image/jpeg's
// writeBlock, reading the 8x8 samples from a byte plane:
//
//	b[8*y+x] = int32(src[y*stride+x]) for x, y in [0, 8)
//	FDCT(&b)
//	out[zig] = int16(div(b[Unzig[zig]], t.Div[zig])) for zig in [0, 64)
//
// It returns a bitmap of the non-zero outputs: bit 63-zig is set iff
// out[zig] != 0 (so bits.LeadingZeros64 walks them in zig-zag order).
// src must hold at least 7*stride+8 bytes.
func FDCTQuantRef(src []byte, stride int, t *Quant, out *[BlockSize]int16) uint64 {
	var b Block
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			b[8*y+x] = int32(src[y*stride+x])
		}
	}
	FDCT(&b)
	var nz uint64
	for zig := 0; zig < BlockSize; zig++ {
		v := div(b[Unzig[zig]], t.Div[zig])
		out[zig] = int16(v)
		if v != 0 {
			nz |= 1 << (63 - zig)
		}
	}
	return nz
}
