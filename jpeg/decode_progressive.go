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

package jpeg

import (
	"encoding/binary"
	mathbits "math/bits"
)

// Fast decoding of progressive scans, with the reader and tables of
// decode_huffman.go and under the same rules: every function mirrors one
// branch of decBlockExact (processSOS's block loop) or refine, checks that
// image/jpeg's reads stay within the safe bits it has loaded, and returns
// false, leaving the reader, d.eobRun, the DC prediction and the block as
// they were, whenever it cannot reproduce image/jpeg's result; the caller
// then decodes the block with the exact code.

// decLoad8 loads as many whole bytes from buf[pos:] as fit in the bit
// buffer if the next 8 bytes are available and none of them is 0xFF (see
// decRefill); ok reports whether it did.
func decLoad8(buf []byte, acc uint64, nbits uint, pos int) (uint64, uint, int, bool) {
	if pos+8 <= len(buf) {
		w := binary.BigEndian.Uint64(buf[pos:])
		if x := ^w; (x-0x0101010101010101)&^x&0x8080808080808080 == 0 {
			return acc | w>>(nbits&63), nbits | 56, pos + int((63-nbits)>>3), true
		}
	}
	return acc, nbits, pos, false
}

// decBlockDCFirst is decBlockExact for a DC first scan (zigStart = zigEnd =
// ah = 0): b[0] = dc<<al.
func (d *decoder) decBlockDCFirst(br *decBitReader, b *block, dcH *huffman, dcPred *int32, al uint) bool {
	buf := d.bytes.buf[:d.bytes.j]
	acc, nbits, pos := br.acc, br.nbits, br.pos
	if nbits < 32 {
		var ok bool
		if acc, nbits, pos, ok = decLoad8(buf, acc, nbits, pos); !ok {
			acc, nbits, pos = decRefill(buf, acc, nbits, pos)
			if nbits < 8 {
				return false
			}
		}
	}
	e := dcH.decLUT[acc>>(64-decLUTBits)]
	if e&0xff == 0 {
		if e = decSlowSym(dcH, acc, nbits); e == 0 || e>>8 > 16 {
			return false
		}
	}
	nb := uint(e) & 63
	if nb > nbits {
		return false
	}
	acc <<= nb
	nbits -= nb
	delta := int32(e) >> 16
	if s := uint(e>>8) & 0xff; s != 0 {
		if s > nbits {
			return false
		}
		delta = decExtend(acc>>((64-s)&63), s)
		acc <<= s & 63
		nbits -= s
	}
	*dcPred += delta
	b[0] = *dcPred << (al & 31)
	br.acc, br.nbits, br.pos = acc, nbits, pos
	return true
}

// decBlockDCRefine is refine for a DC scan: one bit.
func (d *decoder) decBlockDCRefine(br *decBitReader, b *block, delta int32) bool {
	acc, nbits, pos := br.acc, br.nbits, br.pos
	if nbits == 0 {
		buf := d.bytes.buf[:d.bytes.j]
		var ok bool
		if acc, nbits, pos, ok = decLoad8(buf, acc, nbits, pos); !ok {
			if acc, nbits, pos = decRefill(buf, acc, nbits, pos); nbits == 0 {
				return false
			}
		}
	}
	if acc>>63 != 0 {
		b[0] |= delta
	}
	br.acc, br.nbits, br.pos = acc<<1, nbits-1, pos
	return true
}

// decBlockACFirst is decBlockExact for an AC first scan (zigStart > 0,
// ah = 0) when d.eobRun is 0.
//
// If nz is not nil, it is the block's non-zero mask (see decoder.decNZ),
// which it updates.
func (d *decoder) decBlockACFirst(br *decBitReader, b *block, acH *huffman, zigStart, zigEnd, al uint, nz *uint64) bool {
	buf := d.bytes.buf[:d.bytes.j]
	acc, nbits, pos := br.acc, br.nbits, br.pos
	acT := acH.decLUT
	_ = acT[0]
	al &= 31
	eobRun := uint16(0)
	var written uint64 // The coefficients set (never to 0).
	for zig := zigStart; zig <= zigEnd; {
		if nbits < 32 {
			var ok bool
			if acc, nbits, pos, ok = decLoad8(buf, acc, nbits, pos); !ok {
				acc, nbits, pos = decRefill(buf, acc, nbits, pos)
				if nbits < 8 {
					return false
				}
			}
		}
		e := acT[acc>>(64-decLUTBits)]
		if v := int32(e) >> 16; v != 0 {
			nb := uint(e) & 63
			if nb > nbits {
				return false
			}
			zig += uint(e>>12) & 15
			if zig > zigEnd {
				nb -= uint(e>>8) & 15
				acc <<= nb
				nbits -= nb
				break
			}
			acc <<= nb
			nbits -= nb
			b[decUnzig[zig&63]&63] = v << al
			written |= 1 << (zig & 63)
			zig++
			continue
		}
		if e&0xff == 0 {
			if e = decSlowSym(acH, acc, nbits); e == 0 {
				return false
			}
		}
		nb := uint(e) & 63
		if nb > nbits {
			return false
		}
		acc <<= nb
		nbits -= nb
		r, s := uint(e>>12)&15, uint(e>>8)&15
		if s == 0 {
			if r == 0x0f {
				zig += 0x10
				continue
			}
			// End of band, with an EOB run of 1<<r plus r more bits.
			eobRun = uint16(1) << r
			if r != 0 {
				if r > nbits {
					return false
				}
				eobRun |= uint16(acc >> ((64 - r) & 63))
				acc <<= r
				nbits -= r
			}
			eobRun--
			break
		}
		zig += r
		if zig > zigEnd {
			break
		}
		if s > nbits {
			return false
		}
		b[decUnzig[zig&63]&63] = decExtend(acc>>((64-s)&63), s) << al
		written |= 1 << (zig & 63)
		acc <<= s
		nbits -= s
		zig++
	}
	br.acc, br.nbits, br.pos = acc, nbits, pos
	d.eobRun = eobRun
	if nz != nil {
		*nz |= written
	}
	return true
}

// decNonZero returns the non-zero mask of b's AC coefficients (bit zig set
// if b[unzig[zig]] != 0, for zig in 1..63).
func decNonZero(b *block) uint64 {
	var m uint64
	for zig := 1; zig < blockSize; zig++ {
		v := b[decUnzig[zig]&63]
		m |= uint64(uint32(v|-v)>>31) << zig
	}
	return m
}

// decBlockACRefine is refine (with refineNonZeroes) for an AC scan. It
// changes b in place, recording the original values in d.decUndo, and
// restores them if it gives up.
//
// Instead of testing the coefficients one by one as refineNonZeroes does, it
// keeps a mask of the non-zero ones in zig-zag order (bit zig), finds where
// refineNonZeroes stops (at the (nz+1)th zero coefficient) with bit
// operations, and applies the correction bits, one per non-zero coefficient
// in zig-zag order, without branches.
//
// If nz is not nil, it is the block's non-zero mask (see decoder.decNZ),
// which it uses and updates; otherwise it computes the mask from b.
func (d *decoder) decBlockACRefine(br *decBitReader, b *block, acH *huffman, zigStart, zigEnd uint, delta int32, nz *uint64) bool {
	buf := d.bytes.buf[:d.bytes.j]
	acc, nbits, pos := br.acc, br.nbits, br.pos
	acT := acH.decLUT
	_ = acT[0]
	undo := &d.decUndo
	var touched uint64 // Bit k: b[k] changed, its original value is undo[k].
	eobRun := d.eobRun
	zigStart &= 63
	zigEnd &= 63

	// band has the bits zigStart..zigEnd, and nzm the non-zero
	// coefficients in the band.
	band := (^uint64(0) >> (63 - zigEnd)) &^ (1<<zigStart - 1)
	var nzm uint64
	if nz != nil {
		nzm = *nz & band
	} else {
		for zig := zigStart; zig <= zigEnd; zig++ {
			v := b[decUnzig[zig]&63]
			nzm |= uint64(uint32(v|-v)>>31) << zig
		}
	}

	zig := zigStart
	if eobRun == 0 {
	loop:
		for zig <= zigEnd {
			if nbits < 32 {
				var ok bool
				if acc, nbits, pos, ok = decLoad8(buf, acc, nbits, pos); !ok {
					acc, nbits, pos = decRefill(buf, acc, nbits, pos)
					if nbits < 8 {
						goto fail
					}
				}
			}
			// decodeHuffman, and for a new coefficient its sign bit (which
			// the table entries of size 1 include as the value ±1).
			e := acT[acc>>(64-decLUTBits)]
			z := int32(0)
			var r uint
			if v := int32(e) >> 16; v != 0 {
				nb := uint(e) & 63
				if uint(e>>8)&15 != 1 || nb > nbits {
					goto fail // "unexpected Huffman code", or too few bits.
				}
				acc <<= nb
				nbits -= nb
				z = v * delta
				r = uint(e>>12) & 15
			} else {
				if e&0xff == 0 {
					if e = decSlowSym(acH, acc, nbits); e == 0 {
						goto fail
					}
				}
				nb := uint(e) & 63
				if nb > nbits {
					goto fail
				}
				acc <<= nb
				nbits -= nb
				r = uint(e>>12) & 15
				switch uint(e>>8) & 15 {
				case 0:
					if r != 0x0f {
						eobRun = uint16(1) << r
						if r != 0 {
							if r > nbits {
								goto fail
							}
							eobRun |= uint16(acc >> ((64 - r) & 63))
							acc <<= r
							nbits -= r
						}
						break loop
					}
				case 1:
					if nbits == 0 {
						goto fail
					}
					z = delta
					if acc>>63 == 0 {
						z = -z
					}
					acc <<= 1
					nbits--
				default:
					goto fail // "unexpected Huffman code".
				}
			}

			// refineNonZeroes(b, zig, zigEnd, r, delta) stops at the
			// (r+1)th zero coefficient from zig on.
			zeros := ^nzm & band &^ (1<<zig - 1)
			for ; r > 0 && zeros != 0; r-- {
				zeros &= zeros - 1
			}
			if zeros == 0 {
				goto fail // "too many coefficients".
			}
			stop := uint(mathbits.TrailingZeros64(zeros))
			for m := nzm &^ (1<<zig - 1) & (1<<stop - 1); m != 0; m &= m - 1 {
				if nbits == 0 {
					var ok bool
					if acc, nbits, pos, ok = decLoad8(buf, acc, nbits, pos); !ok {
						if acc, nbits, pos = decRefill(buf, acc, nbits, pos); nbits == 0 {
							goto fail
						}
					}
				}
				bit := uint32(acc >> 63)
				acc <<= 1
				nbits--
				u := decUnzig[mathbits.TrailingZeros64(m)&63] & 63
				v := b[u]
				undo[u] = v
				touched |= uint64(bit) << u
				sign := v >> 31
				b[u] = v + ((delta^sign)-sign)&-int32(bit)
			}
			zig = stop
			if z != 0 {
				u := decUnzig[zig&63] & 63
				undo[u] = b[u]
				touched |= 1 << u
				b[u] = z
				nzm |= 1 << zig
			}
			zig++
		}
	}
	if eobRun > 0 {
		eobRun--
		// refineNonZeroes(b, zig, zigEnd, -1, delta).
		for m := nzm &^ (1<<zig - 1); m != 0; m &= m - 1 {
			if nbits == 0 {
				var ok bool
				if acc, nbits, pos, ok = decLoad8(buf, acc, nbits, pos); !ok {
					if acc, nbits, pos = decRefill(buf, acc, nbits, pos); nbits == 0 {
						goto fail
					}
				}
			}
			bit := uint32(acc >> 63)
			acc <<= 1
			nbits--
			u := decUnzig[mathbits.TrailingZeros64(m)&63] & 63
			v := b[u]
			undo[u] = v
			touched |= uint64(bit) << u
			sign := v >> 31
			b[u] = v + ((delta^sign)-sign)&-int32(bit)
		}
	}
	br.acc, br.nbits, br.pos = acc, nbits, pos
	d.eobRun = eobRun
	if nz != nil {
		*nz |= nzm // Corrections keep coefficients non-zero.
	}
	return true

fail:
	for touched != 0 {
		k := mathbits.TrailingZeros64(touched)
		b[k&63] = undo[k&63]
		touched &= touched - 1
	}
	return false
}
