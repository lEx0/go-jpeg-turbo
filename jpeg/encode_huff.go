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
	encbits "math/bits"

	"github.com/lEx0/go-jpeg-turbo/internal/dct"
)

// Entropy coding. image/jpeg's emit appends the bits of each Huffman code
// and of each value separately to a 32-bit buffer and writes every complete
// byte with WriteByte (plus a stuffed 0x00 after 0xff). The bit stream is
// the same here, but each coefficient is one append of code<<size|value to
// a 64-bit buffer, and the buffer is stored 8 bytes at a time into e.out
// unless one of them is 0xff.

// encHuff holds the Huffman codes of one table class, combined with the
// size of the value that follows them: entry rs (run<<4 | size for AC,
// size for DC) is codeword<<size | (codeword length+size)<<32, so that the
// bits of a coefficient are entry&0xffffffff | value bits, and their count
// entry>>32. Codes absent from the table have length 0, as in image/jpeg's
// lookup table (which emits only the value bits then).
type encHuff struct {
	dc [16]uint64
	ac [256]uint64
}

// encHuffTables are the luminance (0) and chrominance (1) codes.
var encHuffTables = encBuildHuff()

func encBuildHuff() (t [2]encHuff) {
	entry := func(lut huffmanLUT, rs int) uint64 {
		var c uint32
		if rs < len(lut) {
			c = lut[rs]
		}
		s := uint(rs & 15)
		return uint64(c&0xffffff)<<s | uint64(c>>24+uint32(s))<<32
	}
	for c := range t {
		var dc, ac huffmanLUT
		dc.init(theHuffmanSpec[2*c])
		ac.init(theHuffmanSpec[2*c+1])
		for s := range t[c].dc {
			t[c].dc[s] = entry(dc, s)
		}
		for rs := range t[c].ac {
			t[c].ac[rs] = entry(ac, rs)
		}
	}
	return t
}

// encMaxBlockBytes bounds what encCode appends to e.out for a block: a DC
// code and value of at most 11+11 bits, 63 AC codes and values of at most
// 16+16 bits, 3 ZRL codes and an EOB, all possibly doubled by byte
// stuffing, plus the 63 bits that may already be pending. encPutWordFF
// needs 16 bytes at the position it writes to.
const encMaxBlockBytes = 2 * (22 + 63*32 + 4*16 + 63 + 7) / 8

// encOutReserve must leave room for a batch.
const _ = uint(encOutReserve - encBatchMax*encMaxBlockBytes - 16)

// encBatchMax is the number of quantized blocks the encoder holds before
// entropy coding them (an MCU of 6 blocks, or 8 blocks of a gray strip).
const encBatchMax = 8

// encBatch holds quantized blocks waiting for entropy coding, and the DC
// predictors.
type encBatch struct {
	coefs [encBatchMax][blockSize]int16
	nzs   [encBatchMax]uint64
	comps [encBatchMax]uint8 // 0: Y, 1: Cb, 2: Cr
	nb    int
	dc    [3]int32
}

// encQuant quantizes the 8x8 block of samples at src (of component comp)
// with dct.FDCTQuant and queues it. At most encBatchMax blocks can be
// queued before encCode.
func (e *encoder) encQuant(src []byte, stride int, comp uint8) {
	i := e.nb & (encBatchMax - 1)
	e.nzs[i] = dct.FDCTQuant(src, stride, &e.qt[min(comp, 1)], &e.coefs[i])
	e.comps[i] = comp
	e.nb = i + 1
}

// encCode appends the Huffman coding of the queued blocks, with DC
// prediction per component: image/jpeg's writeBlock after the
// quantization. It needs encMaxBlockBytes free bytes in e.out per block.
func (e *encoder) encCode() {
	acc, n := e.bits, e.nBits
	buf := e.out[:cap(e.out)]
	pos := len(e.out)

	// Appending the l (<= 32) low bits of v is
	//
	//	if n += l; n <= 64 {
	//		acc = acc<<l | v
	//	} else {
	//		n -= 64
	//		w := acc<<(l-n) | v>>n // the 64 oldest bits
	//		acc = v
	//		store w, stuffing a 0x00 after every 0xff
	//		(encPutWordFF)
	//	}
	//
	// written out below at every use so that acc, n and pos stay in
	// registers. All the shift counts are below 64 (&63 tells the
	// compiler).
	nb := min(e.nb, encBatchMax)
	for b := 0; b < nb; b++ {
		coef := &e.coefs[b]
		nz := e.nzs[b]
		comp := e.comps[b] % 3
		h := &encHuffTables[(comp+1)>>1]

		// The DC difference.
		dc := int32(coef[0])
		{
			d := dc - e.dc[comp]
			t := d >> 31
			s := uint(encbits.Len32(uint32((d ^ t) - t)))
			c := h.dc[s&15]
			l := uint(c >> 32)
			v := c&0xffffffff | uint64(uint32(d+t))&(1<<(s&63)-1)
			if n += l; n <= 64 {
				acc = acc<<(l&63) | v
			} else {
				n -= 64
				w := acc<<((l-n)&63) | v>>(n&63)
				acc = v
				if (^w-0x0101010101010101)&w&0x8080808080808080 == 0 {
					binary.BigEndian.PutUint64(buf[pos:], w)
					pos += 8
				} else {
					pos = encPutWordFF(buf, pos, w)
				}
			}
		}

		// The AC coefficients: nz<<1 has the bit of zig at bit 63-(zig-1).
		nz <<= 1
		zig := uint(1)
		for nz != 0 {
			run := uint(encbits.LeadingZeros64(nz))
			nz <<= (run + 1) & 63
			zig += run
			for run > 15 {
				c := h.ac[0xf0]
				l := uint(c >> 32)
				v := c & 0xffffffff
				if n += l; n <= 64 {
					acc = acc<<(l&63) | v
				} else {
					n -= 64
					w := acc<<((l-n)&63) | v>>(n&63)
					acc = v
					if (^w-0x0101010101010101)&w&0x8080808080808080 == 0 {
						binary.BigEndian.PutUint64(buf[pos:], w)
						pos += 8
					} else {
						pos = encPutWordFF(buf, pos, w)
					}
				}
				run -= 16
			}
			x := int32(coef[zig&(blockSize-1)])
			zig++
			t := x >> 31
			s := uint(encbits.Len32(uint32((x ^ t) - t)))
			c := h.ac[(run<<4|s)&0xff]
			l := uint(c >> 32)
			v := c&0xffffffff | uint64(uint32(x+t))&(1<<(s&63)-1)
			if n += l; n <= 64 {
				acc = acc<<(l&63) | v
			} else {
				n -= 64
				w := acc<<((l-n)&63) | v>>(n&63)
				acc = v
				if (^w-0x0101010101010101)&w&0x8080808080808080 == 0 {
					binary.BigEndian.PutUint64(buf[pos:], w)
					pos += 8
				} else {
					pos = encPutWordFF(buf, pos, w)
				}
			}
		}
		if zig < blockSize {
			// Trailing zeros: EOB.
			c := h.ac[0x00]
			l := uint(c >> 32)
			v := c & 0xffffffff
			if n += l; n <= 64 {
				acc = acc<<(l&63) | v
			} else {
				n -= 64
				w := acc<<((l-n)&63) | v>>(n&63)
				acc = v
				if (^w-0x0101010101010101)&w&0x8080808080808080 == 0 {
					binary.BigEndian.PutUint64(buf[pos:], w)
					pos += 8
				} else {
					pos = encPutWordFF(buf, pos, w)
				}
			}
		}

		e.dc[comp] = dc
	}
	e.nb = 0
	e.bits, e.nBits = acc, n
	e.out = buf[:pos]
}

// encPutWordFF stores the 8 bytes of w at buf[pos:], big-endian, each
// 0xff followed by a stuffed 0x00, and returns the new position. encCode
// calls it only for a w that has a 0xff byte (and otherwise stores w
// directly).
//
//go:noinline
func encPutWordFF(buf []byte, pos int, w uint64) int {
	buf = buf[pos : pos+16]
	i := 0
	for s := 56; s >= 0; s -= 8 {
		b := byte(w >> uint(s))
		buf[i] = b
		i++
		if b == 0xff {
			buf[i] = 0
			i++
		}
	}
	return pos + i
}

// encFinish writes the pending bits, the last byte padded with 1's, as
// image/jpeg's final emit(0x7f, 7) does.
func (e *encoder) encFinish() {
	acc, n := e.bits, e.nBits
	for n >= 8 {
		n -= 8
		e.encByte(byte(acc >> n))
	}
	if n > 0 {
		e.encByte(byte(acc<<(8-n)) | byte(0xff>>n))
	}
	e.bits, e.nBits = 0, 0
}

func (e *encoder) encByte(b byte) {
	if b == 0xff {
		e.out = append(e.out, b, 0)
	} else {
		e.out = append(e.out, b)
	}
}
