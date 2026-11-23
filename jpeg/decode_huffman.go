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

import "encoding/binary"

// Fast entropy decoding.
//
// The decoder of image/jpeg (huffman.go, reader.go, scan.go) reads the
// entropy-coded data one byte at a time through readByteStuffedByte into a
// 32-bit accumulator and decodes one symbol per function call. This file
// adds a fast path in the style of libjpeg-turbo: a 64-bit, left-aligned bit
// buffer refilled up to 8 bytes at a time straight from d.bytes.buf (with
// 0xFF00 unstuffing), a 2^decLUTBits-entry lookup table that yields the
// code length, the symbol and, when they fit, the sign-extended value bits
// in one load, and a whole block decoded per call. The exact image/jpeg code
// is kept and used whenever the fast path cannot prove that it computes the
// same thing.
//
// Why the results are identical
//
// The state of image/jpeg's entropy decoder is (d.bits.a, d.bits.m,
// d.bits.n, d.bytes.i, d.bytes.nUnreadable), with d.bytes.buf[:d.bytes.j]
// the buffered input. Between two decoding operations (decodeHuffman,
// receiveExtend, decodeBits, decodeBit) it holds 0..15 unread bits: only the
// low n bits of a matter, and m == 1<<(n-1) (0 when n == 0). Whenever
// n >= 8 the low 8 bits of a are the whole byte read last, and nUnreadable
// is its physical length (1, or 2 for a stuffed 0xFF 0x00): the only things
// that clear nUnreadable inside a scan are a failed read and the unread in
// decodeHuffman, and both leave n < 8.
//
// Call a state canonical if n < 8. Every state S with n >= 8 has a canonical
// twin S' that gives the whole byte back to the buffer (i -= nUnreadable,
// n -= 8, nUnreadable = 0). From S', the image/jpeg code behaves exactly as
// from S: the first operation that needs more than n' bits makes
// ensureNBits re-read that byte from the buffer (it is still there, never
// needs fill, and reproduces S exactly, nUnreadable included); an operation
// that needs at most n' bits consumes the same bits from both and keeps the
// twin relation; readFull and ignore at a marker give the byte back
// themselves (S) or have nothing to give back (S'), reaching the same i, and
// the bits are dead afterwards (processSOS and restart handling reset them).
// The same induction covers any number of whole bytes given back, provided
// the bytes are still in the buffer.
//
// The fast reader (decBitReader) keeps a canonical state in another form:
// pos is d.bytes.i plus the bytes it has loaded ahead, nbits the unread
// bits (the partial byte left over at the canonical position followed by
// nbits/8 whole loaded bytes). store converts it back to the canonical
// state (giving back the nbits/8 whole bytes, with their physical lengths),
// load converts the image/jpeg state into it (giving back the byte the
// accumulator may hold). So the two decoders can hand over to each other at
// any block boundary.
//
// What the fast path must never do is decode something for which image/jpeg
// would read a byte it cannot read from the buffer as a plain data byte:
// image/jpeg calls fill when it needs a byte at d.bytes.j, or the byte
// after a 0xFF that is the last buffered byte (fill may then return more
// data, an error, or EOF), and it treats an 0xFF followed by anything but
// 0x00 as a marker (an error, or for decodeHuffman's eager 8-bit read a
// fallback to bit-by-bit decoding). The fast reader therefore loads only
// "safe" bytes (a non-0xFF byte before j, or 0xFF 0x00 with both bytes
// before j) and stops at the first unsafe one, and every operation checks
// that image/jpeg's reads for it stay within the loaded bits:
//
//   - decodeHuffman always reads until it has at least 8 bits (eagerly, even
//     for shorter codes), and longer codes up to their length: it needs
//     max(8, length) bits. The fast path requires nbits >= 8 at every symbol
//     and nbits >= length.
//   - receiveExtend(t) and decodeBits(t) need t more bits, decodeBit one.
//
// (The bits beyond nbits that a lookup also sees do not matter then: an
// entry's result depends only on the first max(8, length+size) bits.)
//
// When a check fails, or the block contains anything unusual (a DC
// magnitude above 16, a code image/jpeg would reject, an EOB run in a
// sequential scan, ...), the block is abandoned: the reader is left at the
// block start,
// converted to the canonical state, and the block is decoded again by the
// exact image/jpeg code, which then calls fill exactly when image/jpeg does
// and returns exactly its errors. After that block the fast path resumes
// from the resulting state. Near the end of each 4096-byte buffer this
// happens for one block; for inputs delivered a few bytes per Read, for
// every block. decode_progressive.go applies the same scheme to the
// progressive scans.
//
// The lookup tables are derived by simulating decodeHuffman (its 8-bit lut,
// then the bit-by-bit maxCodes walk) on every decLUTBits-bit prefix, so that
// malformed tables (over-subscribed, incomplete, duplicate values) decode
// exactly as image/jpeg decodes them. For well-formed tables (no length
// over-subscribed) both stages agree with plain prefix decoding and the table
// is filled code by code instead, which is cheaper; the test suite checks
// the two constructions against each other. Prefixes that do not resolve
// within decLUTBits bits are decoded by decSlowSym, which runs the same
// simulation on the full 16 bits.

// decLUTBits is the number of bits the fast decoder looks up at once.
const decLUTBits = 10

// decLUT is a fast decoding table, indexed by the next decLUTBits bits.
//
// An entry of 0 means that the prefix is not resolved within decLUTBits bits
// (or is invalid): decSlowSym decides. Otherwise the low 8 bits are the
// number of bits the entry consumes, and:
//
//   - in an AC table (class 1), bits 8..15 hold the symbol (run<<4 | size)
//     and bits 16..31 the sign-extended value when the code and its size
//     value bits both fit in decLUTBits (the entry then consumes both; the
//     value is never 0), or 0 when only the code is consumed;
//   - in a DC table (class 0), bits 16..31 hold the sign-extended DC
//     difference and bits 8..15 are 0 when the code and its value bits fit
//     (the entry consumes both), or bits 8..15 hold the size (1..16) of the
//     value still to be read after the code.
type decLUT [1 << decLUTBits]uint32

// decUnzig is unzig as bytes.
var decUnzig = [blockSize]uint8{
	0, 1, 8, 16, 9, 2, 3, 10,
	17, 24, 32, 25, 18, 11, 4, 5,
	12, 19, 26, 33, 40, 48, 41, 34,
	27, 20, 13, 6, 7, 14, 21, 28,
	35, 42, 49, 56, 57, 50, 43, 36,
	29, 22, 15, 23, 30, 37, 44, 51,
	58, 59, 52, 45, 38, 31, 39, 46,
	53, 60, 61, 54, 47, 55, 62, 63,
}

// decExtend is receiveExtend's EXTEND step on the t-bit value x, t in
// [1, 16].
func decExtend(x uint64, t uint) int32 {
	// The masks only tell the compiler that the shifts are below 64.
	v := int32(x)
	return v + (int32(x>>((t-1)&63))-1)&(int32(-1)<<(t&63)+1)
}

// decWellFormed reports whether no code length of h is over-subscribed, in
// which case decodeHuffman's lut and its bit-by-bit walk both decode plain
// canonical prefix codes.
func (h *huffman) decWellFormed() bool {
	for i := range h.maxCodes {
		if h.maxCodes[i] >= 1<<(i+1) {
			return false
		}
	}
	return true
}

// decSimulate returns what decodeHuffman returns for a bit stream starting
// with the decLUTBits bits of p, if that is decided by those bits alone
// (decodeHuffman is called with at least 8 bits available).
func (h *huffman) decSimulate(p uint32) (sym uint8, length uint32, ok bool) {
	if v := h.lut[p>>(decLUTBits-lutSize)]; v != 0 {
		return uint8(v >> 8), uint32(v&0xff) - 1, true
	}
	for i := 0; i < decLUTBits; i++ {
		code := int32(p >> (decLUTBits - 1 - i))
		if code <= h.maxCodes[i] {
			return h.vals[h.valsIndices[i]+code-h.minCodes[i]], uint32(i + 1), true
		}
	}
	return 0, 0, false
}

// decEntry builds the decLUT entry of class tc for the prefix p, whose
// first length bits decode to sym.
func decEntry(tc int, p uint32, sym uint8, length uint32) uint32 {
	s := uint32(sym)
	if tc == dcTable {
		if s > 16 {
			return 0 // "excessive DC component": left to the exact path.
		}
		if length+s > decLUTBits {
			return s<<8 | length
		}
		v := int32(0)
		if s != 0 {
			v = decExtend(uint64(p>>(decLUTBits-length-s))&(1<<s-1), uint(s))
		}
		return uint32(uint16(int16(v)))<<16 | (length + s)
	}
	s &= 15
	if s == 0 || length+s > decLUTBits {
		return uint32(sym)<<8 | length
	}
	v := decExtend(uint64(p>>(decLUTBits-length-s))&(1<<s-1), uint(s))
	return uint32(uint16(int16(v)))<<16 | uint32(sym)<<8 | (length + s)
}

// decBuildLUT (re)builds h.decLUT for use as a table of class tc.
func (h *huffman) decBuildLUT(tc int) {
	if h.decLUT == nil {
		h.decLUT = new(decLUT)
	}
	if h.decWellFormed() {
		h.decFillLUT(tc)
	} else {
		h.decSimulateLUT(tc)
	}
	h.decLUTValid = true
}

// decSimulateLUT builds h.decLUT by simulating decodeHuffman on every
// prefix. It is correct for every table.
func (h *huffman) decSimulateLUT(tc int) {
	t := h.decLUT
	for p := range t {
		t[p] = 0
		if sym, length, ok := h.decSimulate(uint32(p)); ok {
			t[p] = decEntry(tc, uint32(p), sym, length)
		}
	}
}

// decFillLUT builds h.decLUT code by code. It is only correct for
// well-formed tables (see decWellFormed). The entries of a code are equal
// in runs: all of them, or, when they include the value bits, those with
// the same value bits; decEntry is called once per run.
func (h *huffman) decFillLUT(tc int) {
	t := h.decLUT
	clear(t[:])
	for i := 0; i < decLUTBits; i++ {
		if h.maxCodes[i] < 0 {
			continue
		}
		length := uint32(i + 1)
		shift := decLUTBits - length
		for code := h.minCodes[i]; code <= h.maxCodes[i]; code++ {
			sym := h.vals[h.valsIndices[i]+code-h.minCodes[i]]
			s := uint32(sym)
			if tc == acTable {
				s &= 15
			}
			run := uint32(1) << shift
			if s <= 16 && s != 0 && length+s <= decLUTBits {
				run >>= s // The value bits vary within the code's entries.
			}
			lo := uint32(code) << shift
			for p := lo; p < lo+1<<shift; p += run {
				e := decEntry(tc, p, sym, length)
				for k := p; k < p+run; k++ {
					t[k] = e
				}
			}
		}
	}
}

// decSlowSym is decodeHuffman on the bits at the top of acc, of which
// nbits >= 8 are valid. It returns sym<<8 | length, or 0 if decodeHuffman
// would need more than nbits bits or report a bad code.
func decSlowSym(h *huffman, acc uint64, nbits uint) uint32 {
	if v := h.lut[acc>>(64-lutSize)]; v != 0 {
		return uint32(v>>8)<<8 | uint32(v&0xff-1)
	}
	for i := uint(0); i < maxCodeLength; i++ {
		if i >= nbits {
			return 0
		}
		code := int32(acc >> (63 - i))
		if code <= h.maxCodes[i] {
			return uint32(h.vals[h.valsIndices[i]+code-h.minCodes[i]])<<8 | uint32(i+1)
		}
	}
	return 0
}

// decBitReader is the fast reader's state (see the comment at the top of
// this file): the next unread bits are the top nbits bits of acc (bits
// below them are zero or copies of the next stream bits), and the next byte
// to load is d.bytes.buf[pos]. origin is the position of the canonical
// state the reader was loaded from: every whole byte in acc was loaded at
// or after it.
type decBitReader struct {
	acc    uint64
	nbits  uint
	pos    int
	origin int
}

// load sets br from the image/jpeg decoder state. It returns false if that
// state is not one image/jpeg can be in between two operations (then the
// fast path must not be used).
func (br *decBitReader) load(d *decoder) bool {
	a, n, pos := d.bits.a, d.bits.n, d.bytes.i
	if n >= 8 {
		if n >= 16 || d.bytes.nUnreadable == 0 {
			return false
		}
		a >>= 8
		n -= 8
		pos -= d.bytes.nUnreadable
	}
	if n < 0 || pos < 0 || pos > d.bytes.j {
		return false
	}
	br.acc = uint64(a) << (64 - uint(n))
	br.nbits = uint(n)
	br.pos = pos
	br.origin = pos
	return true
}

// store sets the image/jpeg decoder state to the canonical form of br.
func (br *decBitReader) store(d *decoder) {
	pos := br.pos
	for w := br.nbits >> 3; w > 0; w-- {
		// Every loaded byte is a plain byte or 0xFF 0x00, so the last one
		// is a stuffed 0xFF exactly if it ends in 0x00 preceded by 0xFF
		// (a plain 0x00 is preceded by a plain byte or the 0x00 of a
		// stuffed pair, never by 0xFF), unless it is the first byte loaded.
		if pos-2 >= br.origin && d.bytes.buf[pos-1] == 0x00 && d.bytes.buf[pos-2] == 0xff {
			pos -= 2
		} else {
			pos--
		}
	}
	n := br.nbits & 7
	d.bytes.i = pos
	d.bytes.nUnreadable = 0
	d.bits.a = uint32(br.acc >> (64 - n))
	d.bits.n = int32(n)
	d.bits.m = 0
	if n > 0 {
		d.bits.m = 1 << (n - 1)
	}
}

// decRefill loads safe bytes from buf[pos:] into the bit buffer, one at a
// time, until it holds more than 56 bits or the next byte is unsafe (see the
// comment at the top of this file). Its callers first try to load 8 bytes
// at once:
//
//	if pos+8 <= len(buf) {
//		w := binary.BigEndian.Uint64(buf[pos:])
//		if no byte of w is 0xFF {
//			acc |= w >> nbits
//			pos += (63 - nbits) >> 3
//			nbits |= 56
//		}
//	}
//
// which takes as many whole bytes as fit (nbits | 56 == nbits + 8*k for
// nbits < 64). The bits of the next byte that also land in acc below the
// nbits valid ones are the stream's own bits, so the next load (ORing the
// same byte at the same place) and every lookup stay correct.
func decRefill(buf []byte, acc uint64, nbits uint, pos int) (uint64, uint, int) {
	for nbits <= 56 {
		if pos >= len(buf) {
			break
		}
		c := buf[pos]
		if c == 0xff {
			if pos+1 >= len(buf) || buf[pos+1] != 0x00 {
				break
			}
			pos += 2
		} else {
			pos++
		}
		acc |= uint64(c) << ((56 - nbits) & 63)
		nbits += 8
	}
	return acc, nbits, pos
}

// decBlockSeq decodes one block of a sequential scan (spectral selection
// 0..63, no successive approximation) into b, which must be zero, with the
// fast reader. dcH and acH must have valid decLUTs and d.eobRun must be 0.
// On success it advances br, updates *dcPred and returns true. Otherwise it
// returns false and leaves br and *dcPred unchanged (b may have been
// written to); the caller then decodes the block with the exact code.
func (d *decoder) decBlockSeq(br *decBitReader, b *block, dcH, acH *huffman, dcPred *int32) bool {
	buf := d.bytes.buf[:d.bytes.j]
	acc, nbits, pos := br.acc, br.nbits, br.pos

	// DC coefficient: decodeHuffman, then receiveExtend.
	if nbits < 32 && pos+8 <= len(buf) {
		w := binary.BigEndian.Uint64(buf[pos:])
		if x := ^w; (x-0x0101010101010101)&^x&0x8080808080808080 == 0 {
			acc |= w >> (nbits & 63)
			pos += int((63 - nbits) >> 3)
			nbits |= 56
			goto dcRefilled
		}
	}
	if nbits < 32 {
		acc, nbits, pos = decRefill(buf, acc, nbits, pos)
		if nbits < 8 {
			return false
		}
	}
dcRefilled:
	e := dcH.decLUT[acc>>(64-decLUTBits)]
	if e&0xff == 0 {
		if e = decSlowSym(dcH, acc, nbits); e == 0 || e>>8 > 16 {
			return false
		}
	}
	nb := uint(e) & 63 // Entries keep bits 6 and 7 clear.
	if nb > nbits {
		return false
	}
	acc <<= nb & 63
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
	dc := *dcPred + delta
	b[0] = dc

	// AC coefficients, as in processSOS's loop.
	acT := acH.decLUT
	_ = acT[0] // Hoist the nil check out of the loop.
	for zig := uint(1); zig < blockSize; {
		if nbits < 32 && pos+8 <= len(buf) {
			w := binary.BigEndian.Uint64(buf[pos:])
			if x := ^w; (x-0x0101010101010101)&^x&0x8080808080808080 == 0 {
				acc |= w >> (nbits & 63)
				pos += int((63 - nbits) >> 3)
				nbits |= 56
				goto refilled
			}
		}
		if nbits < 32 {
			acc, nbits, pos = decRefill(buf, acc, nbits, pos)
			if nbits < 8 {
				return false
			}
		}
	refilled:
		e := acT[acc>>(64-decLUTBits)]
		if v := int32(e) >> 16; v != 0 {
			// Code and value bits in one entry.
			nb := uint(e) & 63 // Entries keep bits 6 and 7 clear.
			if nb > nbits {
				return false
			}
			zig += uint(e>>12) & 15
			if zig >= blockSize {
				// processSOS stops before reading the value bits.
				nb -= uint(e>>8) & 15
				acc <<= nb & 63
				nbits -= nb
				break
			}
			acc <<= nb & 63
			nbits -= nb
			b[decUnzig[zig]&63] = v
			zig++
			continue
		}
		if e&0xff == 0 {
			if e = decSlowSym(acH, acc, nbits); e == 0 {
				return false
			}
		}
		nb := uint(e) & 63 // Entries keep bits 6 and 7 clear.
		if nb > nbits {
			return false
		}
		acc <<= nb & 63
		nbits -= nb
		r, s := uint(e>>12)&15, uint(e>>8)&15
		if s == 0 {
			if r == 0x0f {
				zig += 0x10
				continue
			}
			if r == 0 {
				break // EOB.
			}
			return false // EOB run.
		}
		zig += r
		if zig >= blockSize {
			break // processSOS stops before reading the value bits.
		}
		if s > nbits {
			return false
		}
		b[decUnzig[zig]&63] = decExtend(acc>>((64-s)&63), s)
		acc <<= s & 63
		nbits -= s
		zig++
	}
	br.acc, br.nbits, br.pos = acc, nbits, pos
	*dcPred = dc
	return true
}
