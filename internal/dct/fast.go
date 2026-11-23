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

// Portable kernels: the *Ref functions with the same int32 operations,
// rearranged so that Go runs them faster. They are compiled on every
// architecture (and tested everywhere); the generic build calls them.
//
// Inverse DCT. The dequantization is fused into the row pass, which skips
// the arithmetic of rows whose seven AC coefficients are zero: idctRows maps
// such a row to eight copies of x0<<17 (every other term is a product,
// shift or box of zeros). If rows 1-7 of the intermediate block are then
// all zero, idctCols maps column i to eight copies of
// (b[i] + 1<<19) >> 20 (int32, wrapping; the same argument), which covers
// DC-only blocks. If rows 4-7 are zero, a copy of idctCols with those
// inputs as constant zeros runs (the compiler folds the dead arithmetic);
// otherwise idctCols runs as written. The store goes through a local 8x8
// byte block, copied out row by row.
//
// Forward DCT. The samples are copied to a local 8x8 block first (eight
// 8-byte moves) and fdctCols reads them from there; fdctRows is dct.go's.
// The division div(a, D) (D = 8q, h = D/2) is replaced by a multiplication,
// see quantGo.

// dequantIDCTStoreGo is DequantIDCTStoreRef for stride >= 8 and
// len(dst) >= 7*stride+8.
func dequantIDCTStoreGo(b, qt *Block, dst []byte, stride int) {
	rows := idctRowsDequant(b, qt)
	if rows&^1 == 0 {
		var row [8]byte
		for i := range row {
			row[i] = clamp8((b[i] + 1<<19) >> 20)
		}
		for y := 0; y < 8; y++ {
			*(*[8]byte)(dst[y*stride:]) = row
		}
		return
	}
	var out [BlockSize]byte
	if rows&0xF0 == 0 {
		idctColsStoreTop(b, &out)
	} else {
		idctColsStore(b, &out)
	}
	for y := 0; y < 8; y++ {
		*(*[8]byte)(dst[y*stride:]) = *(*[8]byte)(out[8*y:])
	}
}

// dequantIDCTGo is DequantIDCTRef.
func dequantIDCTGo(b, qt *Block) {
	rows := idctRowsDequant(b, qt)
	if rows&^1 == 0 {
		for i := 0; i < 8; i++ {
			v := (b[i] + 1<<19) >> 20
			b[0*8+i] = v
			b[1*8+i] = v
			b[2*8+i] = v
			b[3*8+i] = v
			b[4*8+i] = v
			b[5*8+i] = v
			b[6*8+i] = v
			b[7*8+i] = v
		}
		return
	}
	if rows&0xF0 == 0 {
		idctColsTop(b)
	} else {
		idctCols(b)
	}
}

func clamp8(v int32) uint8 {
	return uint8(max(0, min(255, v+128)))
}

// idctRowsDequant multiplies b by qt and runs idctRows on b. Bit i of the
// result is clear if row i of the result is zero.
func idctRowsDequant(b, qt *Block) (rows uint8) {
	for i := 0; i < 8; i++ {
		x := b[8*i : 8*i+8 : 8*i+8]
		q := qt[8*i : 8*i+8 : 8*i+8]
		x0 := x[0] * q[0]
		x7 := x[1] * q[1]
		x2 := x[2] * q[2]
		x5 := x[3] * q[3]
		x1 := x[4] * q[4]
		x6 := x[5] * q[5]
		x3 := x[6] * q[6]
		x4 := x[7] * q[7]
		if x7|x2|x5|x1|x6|x3|x4 == 0 {
			v := x0 << 17
			if v != 0 {
				rows |= 1 << i
			}
			x[0] = v
			x[1] = v
			x[2] = v
			x[3] = v
			x[4] = v
			x[5] = v
			x[6] = v
			x[7] = v
			continue
		}
		rows |= 1 << i

		// idctRows, see dct.go.
		x0 <<= 17
		x1 <<= 17
		x0, x1 = x0+x1, x0-x1
		x2, x3 = dctBox(x2, x3, c(sqrt2inv_cos6, 18), -c(sqrt2inv_sin6, 18))
		x1, x2 = x1+x2, x1-x2
		x0, x3 = x0+x3, x0-x3

		x4 <<= 7
		x7 <<= 7
		x7, x4 = x7+x4, x7-x4
		x6 = x6 * c(sqrt2inv, 8)
		x5 = x5 * c(sqrt2inv, 8)
		x7, x5 = x7+x5, x7-x5
		x4, x6 = x4+x6, x4-x6
		x4, x7 = dctBox(x4>>2, x7>>2, c(cos3, 12), -c(sin3, 12))
		x5, x6 = dctBox(x5>>2, x6>>2, c(cos1, 12), -c(sin1, 12))

		x0, x7 = x0+x7, x0-x7
		x1, x6 = x1+x6, x1-x6
		x2, x5 = x2+x5, x2-x5
		x3, x4 = x3+x4, x3-x4

		x[0] = x0
		x[1] = x1
		x[2] = x2
		x[3] = x3
		x[4] = x4
		x[5] = x5
		x[6] = x6
		x[7] = x7
	}
	return rows
}

// idctColsStore runs idctCols on b and stores clamp8(v>>18) into out
// (row-major).
func idctColsStore(b *Block, out *[BlockSize]byte) {
	for i := 0; i < 8; i++ {
		x0 := b[0*8+i]
		x7 := b[1*8+i]
		x2 := b[2*8+i]
		x5 := b[3*8+i]
		x1 := b[4*8+i]
		x6 := b[5*8+i]
		x3 := b[6*8+i]
		x4 := b[7*8+i]

		// idctCols, see dct.go.
		x0 += 1 << 19
		x0, x1 = (x0+x1)>>2, (x0-x1)>>2
		x2, x3 = dctBox(x2>>13, x3>>13, c(sqrt2inv_cos6, 12), -c(sqrt2inv_sin6, 12))
		x1, x2 = x1+x2, x1-x2
		x0, x3 = x0+x3, x0-x3

		x7, x4 = x7+x4, x7-x4
		x5 = (x5 >> 13) * c(sqrt2inv, 14)
		x6 = (x6 >> 13) * c(sqrt2inv, 14)
		x7, x5 = x7+x5, x7-x5
		x4, x6 = x4+x6, x4-x6
		x4, x7 = dctBox(x4>>14, x7>>14, c(cos3, 12), -c(sin3, 12))
		x5, x6 = dctBox(x5>>14, x6>>14, c(cos1, 12), -c(sin1, 12))

		x0, x7 = x0+x7, x0-x7
		x1, x6 = x1+x6, x1-x6
		x2, x5 = x2+x5, x2-x5
		x3, x4 = x3+x4, x3-x4

		out[0*8+i] = clamp8(x0 >> 18)
		out[1*8+i] = clamp8(x1 >> 18)
		out[2*8+i] = clamp8(x2 >> 18)
		out[3*8+i] = clamp8(x3 >> 18)
		out[4*8+i] = clamp8(x4 >> 18)
		out[5*8+i] = clamp8(x5 >> 18)
		out[6*8+i] = clamp8(x6 >> 18)
		out[7*8+i] = clamp8(x7 >> 18)
	}
}

// idctColsStoreTop is idctColsStore for rows 4-7 of b zero: the same code
// with x1, x6, x3, x4 = 0, which the compiler folds.
func idctColsStoreTop(b *Block, out *[BlockSize]byte) {
	for i := 0; i < 8; i++ {
		x0 := b[0*8+i]
		x7 := b[1*8+i]
		x2 := b[2*8+i]
		x5 := b[3*8+i]
		x1, x6, x3, x4 := int32(0), int32(0), int32(0), int32(0)

		// idctCols, see dct.go.
		x0 += 1 << 19
		x0, x1 = (x0+x1)>>2, (x0-x1)>>2
		x2, x3 = dctBox(x2>>13, x3>>13, c(sqrt2inv_cos6, 12), -c(sqrt2inv_sin6, 12))
		x1, x2 = x1+x2, x1-x2
		x0, x3 = x0+x3, x0-x3

		x7, x4 = x7+x4, x7-x4
		x5 = (x5 >> 13) * c(sqrt2inv, 14)
		x6 = (x6 >> 13) * c(sqrt2inv, 14)
		x7, x5 = x7+x5, x7-x5
		x4, x6 = x4+x6, x4-x6
		x4, x7 = dctBox(x4>>14, x7>>14, c(cos3, 12), -c(sin3, 12))
		x5, x6 = dctBox(x5>>14, x6>>14, c(cos1, 12), -c(sin1, 12))

		x0, x7 = x0+x7, x0-x7
		x1, x6 = x1+x6, x1-x6
		x2, x5 = x2+x5, x2-x5
		x3, x4 = x3+x4, x3-x4

		out[0*8+i] = clamp8(x0 >> 18)
		out[1*8+i] = clamp8(x1 >> 18)
		out[2*8+i] = clamp8(x2 >> 18)
		out[3*8+i] = clamp8(x3 >> 18)
		out[4*8+i] = clamp8(x4 >> 18)
		out[5*8+i] = clamp8(x5 >> 18)
		out[6*8+i] = clamp8(x6 >> 18)
		out[7*8+i] = clamp8(x7 >> 18)
	}
}

// idctColsTop is idctCols for rows 4-7 of b zero (see idctColsStoreTop).
func idctColsTop(b *Block) {
	for i := 0; i < 8; i++ {
		x0 := b[0*8+i]
		x7 := b[1*8+i]
		x2 := b[2*8+i]
		x5 := b[3*8+i]
		x1, x6, x3, x4 := int32(0), int32(0), int32(0), int32(0)

		// idctCols, see dct.go.
		x0 += 1 << 19
		x0, x1 = (x0+x1)>>2, (x0-x1)>>2
		x2, x3 = dctBox(x2>>13, x3>>13, c(sqrt2inv_cos6, 12), -c(sqrt2inv_sin6, 12))
		x1, x2 = x1+x2, x1-x2
		x0, x3 = x0+x3, x0-x3

		x7, x4 = x7+x4, x7-x4
		x5 = (x5 >> 13) * c(sqrt2inv, 14)
		x6 = (x6 >> 13) * c(sqrt2inv, 14)
		x7, x5 = x7+x5, x7-x5
		x4, x6 = x4+x6, x4-x6
		x4, x7 = dctBox(x4>>14, x7>>14, c(cos3, 12), -c(sin3, 12))
		x5, x6 = dctBox(x5>>14, x6>>14, c(cos1, 12), -c(sin1, 12))

		x0, x7 = x0+x7, x0-x7
		x1, x6 = x1+x6, x1-x6
		x2, x5 = x2+x5, x2-x5
		x3, x4 = x3+x4, x3-x4

		b[0*8+i] = x0 >> 18
		b[1*8+i] = x1 >> 18
		b[2*8+i] = x2 >> 18
		b[3*8+i] = x3 >> 18
		b[4*8+i] = x4 >> 18
		b[5*8+i] = x5 >> 18
		b[6*8+i] = x6 >> 18
		b[7*8+i] = x7 >> 18
	}
}

// fdctQuantGo is FDCTQuantRef for a table accepted by init, stride >= 0 and
// len(src) >= 7*stride+8.
func fdctQuantGo(src []byte, stride int, t *Quant, out *[BlockSize]int16) uint64 {
	var s [BlockSize]byte
	for y := 0; y < 8; y++ {
		*(*[8]byte)(s[8*y:]) = *(*[8]byte)(src[y*stride:])
	}
	var b Block
	fdctColsBytes(&s, &b)
	fdctRows(&b)
	// Unrolled with constant indices (zig, Unzig[zig]).
	nz := quantZig(&b, t, out, 0, 0) |
		quantZig(&b, t, out, 1, 1) |
		quantZig(&b, t, out, 2, 8) |
		quantZig(&b, t, out, 3, 16) |
		quantZig(&b, t, out, 4, 9) |
		quantZig(&b, t, out, 5, 2) |
		quantZig(&b, t, out, 6, 3) |
		quantZig(&b, t, out, 7, 10)
	nz |= quantZig(&b, t, out, 8, 17) |
		quantZig(&b, t, out, 9, 24) |
		quantZig(&b, t, out, 10, 32) |
		quantZig(&b, t, out, 11, 25) |
		quantZig(&b, t, out, 12, 18) |
		quantZig(&b, t, out, 13, 11) |
		quantZig(&b, t, out, 14, 4) |
		quantZig(&b, t, out, 15, 5)
	nz |= quantZig(&b, t, out, 16, 12) |
		quantZig(&b, t, out, 17, 19) |
		quantZig(&b, t, out, 18, 26) |
		quantZig(&b, t, out, 19, 33) |
		quantZig(&b, t, out, 20, 40) |
		quantZig(&b, t, out, 21, 48) |
		quantZig(&b, t, out, 22, 41) |
		quantZig(&b, t, out, 23, 34)
	nz |= quantZig(&b, t, out, 24, 27) |
		quantZig(&b, t, out, 25, 20) |
		quantZig(&b, t, out, 26, 13) |
		quantZig(&b, t, out, 27, 6) |
		quantZig(&b, t, out, 28, 7) |
		quantZig(&b, t, out, 29, 14) |
		quantZig(&b, t, out, 30, 21) |
		quantZig(&b, t, out, 31, 28)
	nz |= quantZig(&b, t, out, 32, 35) |
		quantZig(&b, t, out, 33, 42) |
		quantZig(&b, t, out, 34, 49) |
		quantZig(&b, t, out, 35, 56) |
		quantZig(&b, t, out, 36, 57) |
		quantZig(&b, t, out, 37, 50) |
		quantZig(&b, t, out, 38, 43) |
		quantZig(&b, t, out, 39, 36)
	nz |= quantZig(&b, t, out, 40, 29) |
		quantZig(&b, t, out, 41, 22) |
		quantZig(&b, t, out, 42, 15) |
		quantZig(&b, t, out, 43, 23) |
		quantZig(&b, t, out, 44, 30) |
		quantZig(&b, t, out, 45, 37) |
		quantZig(&b, t, out, 46, 44) |
		quantZig(&b, t, out, 47, 51)
	nz |= quantZig(&b, t, out, 48, 58) |
		quantZig(&b, t, out, 49, 59) |
		quantZig(&b, t, out, 50, 52) |
		quantZig(&b, t, out, 51, 45) |
		quantZig(&b, t, out, 52, 38) |
		quantZig(&b, t, out, 53, 31) |
		quantZig(&b, t, out, 54, 39) |
		quantZig(&b, t, out, 55, 46)
	nz |= quantZig(&b, t, out, 56, 53) |
		quantZig(&b, t, out, 57, 60) |
		quantZig(&b, t, out, 58, 61) |
		quantZig(&b, t, out, 59, 54) |
		quantZig(&b, t, out, 60, 47) |
		quantZig(&b, t, out, 61, 55) |
		quantZig(&b, t, out, 62, 62) |
		quantZig(&b, t, out, 63, 63)
	return nz
}

// quantZig stores the quantized coefficient b[n] (table entry zig) to
// out[zig] and returns its bit of the bitmap.
func quantZig(b *Block, t *Quant, out *[BlockSize]int16, zig, n int) uint64 {
	q := quantGo(b[n], t.qm[zig], t.qc[zig])
	out[zig] = int16(q)
	var bit uint64
	if q != 0 {
		bit = 1 << (63 - zig)
	}
	return bit
}

// quantK is the offset of quantGo.
const quantK = 2048

// quantGo is div(a, D) for qm = ceil(2^40 / D), qc = (h + K*D)*qm - K<<40
// (h = D/2, K = quantK, D = 8q <= 2040) and a in [-16387, 2^24]; the
// outputs of the FDCT are in [-8192, 8191] (int32 >> 18).
//
// With s = a>>31 (0 or -1), div(a, D) = floor((a + s + h) / D): for a >= 0
// that is (a + h) / D, and for a < 0, -((-a + h) / D) = ceil((a - h) / D) =
// floor((a - h + D - 1) / D) = floor((a + h - 1) / D) as D = 2h. Adding K*D
// makes the dividend n = a + s + h + K*D positive and below 2^29 (and
// (a+s)*qm stays below 2^62), and for n
// in [0, 2^29) floor(n*qm / 2^40) = floor(n / D): qm*D = 2^40 + e with
// 0 <= e < D < 2^11, so n*qm / 2^40 = n/D + n*e/(D*2^40), and the excess
// n*e/(D*2^40) < 1/D cannot reach the next multiple of 1/D. So
// (a+s)*qm + qc = n*qm - K<<40, and its arithmetic shift by 40 is
// floor(n / D) - K = div(a, D).
func quantGo(a int32, qm, qc int64) int32 {
	return int32((int64(a+a>>31)*qm + qc) >> 40)
}

// fdctColsBytes is fdctCols with its input as bytes.
func fdctColsBytes(s *[BlockSize]byte, b *Block) {
	for i := 0; i < 8; i++ {
		x0 := int32(s[0*8+i])
		x1 := int32(s[1*8+i])
		x2 := int32(s[2*8+i])
		x3 := int32(s[3*8+i])
		x4 := int32(s[4*8+i])
		x5 := int32(s[5*8+i])
		x6 := int32(s[6*8+i])
		x7 := int32(s[7*8+i])

		// fdctCols, see dct.go.
		x0, x7 = x0+x7, x0-x7
		x1, x6 = x1+x6, x1-x6
		x2, x5 = x2+x5, x2-x5
		x3, x4 = x3+x4, x3-x4

		x4, x7 = dctBox(x4, x7, c(cos3, 18), c(sin3, 18))
		x5, x6 = dctBox(x5, x6, c(cos1, 18), c(sin1, 18))
		x0, x3 = x0+x3, x0-x3
		x1, x2 = x1+x2, x1-x2

		x2, x3 = dctBox(x2, x3, c(sqrt2_cos6, 18), c(sqrt2_sin6, 18))
		x0, x1 = x0+x1, x0-x1
		b[0*8+i] = (x0 - 128*8) << 18
		b[4*8+i] = x1 << 18
		b[2*8+i] = x2
		b[6*8+i] = x3

		x4, x6 = x4+x6, x4-x6
		x7, x5 = x7+x5, x7-x5
		x5 = (x5 >> 12) * c(sqrt2, 12)
		x6 = (x6 >> 12) * c(sqrt2, 12)
		x7, x4 = x7+x4, x7-x4
		b[1*8+i] = x7
		b[3*8+i] = x5
		b[5*8+i] = x6
		b[7*8+i] = x4
	}
}
