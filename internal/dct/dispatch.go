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

package dct

import "math/bits"

//go:generate go run asmgen.go
//go:generate go run asmgen_x86.go

// The functions the codec calls. They check the arguments once and run the
// architecture's kernel (dequantIDCTStore, dequantIDCT, fdctQuant: NEON on
// arm64, see dct_arm64.go; SSE4.1, AVX2 or AVX-512 on amd64 by Level, see
// dct_amd64.go; the portable Go of fast.go elsewhere and with -tags
// purego). Arguments the kernels do not handle (strides below 8 for a
// destination, slices too short, tables with a zero entry) go to the *Ref
// functions, which then behave exactly like image/jpeg, panics included.

// DequantIDCTStore: see DequantIDCTStoreRef.
func DequantIDCTStore(b *Block, qt *Block, dst []byte, stride int) {
	if stride < 8 || len(dst) < 8 || (len(dst)-8)/7 < stride {
		DequantIDCTStoreRef(b, qt, dst, stride)
		return
	}
	dequantIDCTStore(b, qt, dst, stride)
}

// DequantIDCT: see DequantIDCTRef.
func DequantIDCT(b *Block, qt *Block) {
	dequantIDCT(b, qt)
}

// FDCTQuant: see FDCTQuantRef.
func FDCTQuant(src []byte, stride int, t *Quant, out *[BlockSize]int16) uint64 {
	if !t.ok || stride < 0 || len(src) < 8 || (len(src)-8)/7 < stride {
		return FDCTQuantRef(src, stride, t, out)
	}
	return fdctQuant(src, stride, t, out)
}

// init prepares the private fields of t from t.Q.
func (t *Quant) init() {
	t.ok = false
	for zig, q := range t.Q {
		if q == 0 {
			return
		}
		d := 8 * int64(q)
		t.qm[zig] = (1<<40 + d - 1) / d
		t.qc[zig] = (d/2+quantK*d)*t.qm[zig] - quantK<<40
	}
	t.initVec()
	t.ok = true
}

// magic15 returns the largest mul = ceil(2^(15+k) / d) below 2^15, and k,
// for the vector kernels' division by d = 8q (see initVec in dct_arm64.go).
// With 2^(L-1) <= d < 2^L, 2^(14+L) / d is in (2^14, 2^15], and doubling
// it would exceed 2^15; it is 2^15 itself only for a power of two.
func magic15(d uint32) (mul uint32, k int) {
	s := 14 + bits.Len32(d)
	if d&(d-1) == 0 {
		s--
	}
	return (uint32(1)<<s + d - 1) / d, s - 15
}

// magic16 returns mul = ceil(2^(16+s) / d) and s for the x86 kernels
// division by d = 8q (see initVec in dct_amd64.go): s is the largest shift
// with mul below 2^16. With 2^(L-1) <= d < 2^L that is s = L-1, for which
// 2^(15+L) / d is in (2^15, 2^16), except for a power of two, where it
// would be 2^16 and s = L-2 gives mul = 2^15 exactly.
func magic16(d uint32) (mul uint32, s int) {
	s = bits.Len32(d) - 1
	if d&(d-1) == 0 {
		s--
	}
	return (uint32(1)<<(16+s) + d - 1) / d, s
}

// zigOf is the inverse of Unzig.
var zigOf = func() (z [BlockSize]uint8) {
	for zig, n := range Unzig {
		z[n] = uint8(zig)
	}
	return z
}()
