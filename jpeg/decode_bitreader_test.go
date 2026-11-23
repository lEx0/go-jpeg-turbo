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

// The hand-over between the fast entropy decoder (decBitReader,
// decBlockSeq) and image/jpeg's exact code, from the states that
// TestDecBlockSeqVsExact does not start from: a whole byte held in the
// accumulator (n >= 8, nUnreadable 1, or 2 after a stuffed 0xFF00), as
// image/jpeg leaves it after a block; refills from a reader that returns
// short chunks, so that the buffer shifts between fast blocks; every
// stream prefix of two bytes (three for the Annex K luma tables), for
// small hand-picked tables, malformed ones included; and decBitReader's
// invariant on the accumulator's low bits. They guard the planned changes
// of the entropy decoder (two symbols per table entry, whole MCUs at a
// time). From the verification of the sequential decoder.

import (
	"fmt"
	"io"
	"math/rand"
	"testing"
)

// dbChunkReader returns at most n bytes of b per Read and counts the
// calls.
type dbChunkReader struct {
	b     []byte
	n     int
	calls int
}

func (c *dbChunkReader) Read(p []byte) (int, error) {
	c.calls++
	if len(c.b) == 0 {
		return 0, io.EOF
	}
	k := min(len(p), c.n, len(c.b))
	copy(p, c.b[:k])
	c.b = c.b[k:]
	return k, nil
}

// dbState is a decoder state to start from: the buffered bytes buf, of
// which buf[:i] are consumed, nUnreadable and the bit accumulator.
type dbState struct {
	buf   []byte
	i     int
	nUnr  int
	bits  bits
	state bool // Compare the canonical states after every block.
}

// dbCompare decodes up to nBlocks blocks from s with the exact code alone
// (reading more data from exactR) and with the fast decoder and the scan
// loop's hand-over (from fastR), and then reads two bytes with readFull,
// as at the marker after a scan. It reports the first difference in
// coefficients, DC prediction, EOB run, error, number of Read calls
// (calls returns them, fast and exact) or position.
func dbCompare(dcH, acH *huffman, s dbState, fastR, exactR io.Reader, calls func() (int, int), nBlocks int) string {
	fastD, exactD := &dbScratch[0], &dbScratch[1]
	*fastD, *exactD = decoder{}, decoder{}
	copy(fastD.bytes.buf[:], s.buf)
	fastD.bytes.j = len(s.buf)
	fastD.bytes.i = s.i
	fastD.bytes.nUnreadable = s.nUnr
	fastD.bits = s.bits
	exactD.bytes, exactD.bits = fastD.bytes, fastD.bits
	fastD.r, exactD.r = fastR, exactR
	var br decBitReader
	if !br.load(fastD) {
		return "load failed on a state image/jpeg can be in"
	}
	fast := true
	// current returns the fast decoder's state as image/jpeg's (store and
	// readFull use only the fields it sets).
	scratch := &dbScratch[2]
	current := func() *decoder {
		scratch.bytes, scratch.bits, scratch.r = fastD.bytes, fastD.bits, fastD.r
		if fast {
			br.store(scratch)
		}
		return scratch
	}
	var dcF, dcE int32
	for blk := 0; blk < nBlocks; blk++ {
		var bF, bE block
		err := exactD.decBlockExact(&bE, dcH, acH, &dcE, 0, blockSize-1, 0, 0)
		var errF error
		if !fast || fastD.eobRun != 0 || !fastD.decBlockSeq(&br, &bF, dcH, acH, &dcF) {
			if fast {
				bF = block{}
				br.store(fastD)
			}
			errF = fastD.decBlockExact(&bF, dcH, acH, &dcF, 0, blockSize-1, 0, 0)
			if fast && !br.load(fastD) {
				fast = false
			}
		}
		if gc, wc := calls(); !sameError(errF, err) || gc != wc {
			return fmt.Sprintf("block %d: error %v (%d reads), want %v (%d reads)", blk, errF, gc, err, wc)
		}
		if err != nil {
			break
		}
		if bF != bE || dcF != dcE || fastD.eobRun != exactD.eobRun {
			return fmt.Sprintf("block %d: %v dc %d eob %d, want %v dc %d eob %d", blk, bF, dcF, fastD.eobRun, bE, dcE, exactD.eobRun)
		}
		if s.state {
			gi, gn, ga := decCanonical(current())
			wi, wn, wa := decCanonical(exactD)
			if gi != wi || gn != wn || ga != wa {
				return fmt.Sprintf("block %d: state (%d, %d, %#x), want (%d, %d, %#x)", blk, gi, gn, ga, wi, wn, wa)
			}
		}
	}
	tmp := current()
	var p1, p2 [2]byte
	e1, e2 := tmp.readFull(p1[:]), exactD.readFull(p2[:])
	if !sameError(e1, e2) || p1 != p2 || tmp.bytes.i != exactD.bytes.i || tmp.bytes.j != exactD.bytes.j {
		return fmt.Sprintf("readFull: % x %v at %d/%d, want % x %v at %d/%d", p1, e1, tmp.bytes.i, tmp.bytes.j, p2, e2, exactD.bytes.i, exactD.bytes.j)
	}
	return ""
}

// dbScratch holds the decoders of dbCompare, reused from call to call
// (so dbCompare must not run on two goroutines at once). A decoder is
// about 15 KB: allocated in each call, and for each block it compares,
// they made these tests spend most of their time in the garbage
// collector, tens of thousands of cycles for TestDecBlockSeqExhaustive.
var dbScratch [3]decoder

// dbFailing returns two failing readers (decCountingReader) and their
// call counts.
func dbFailing() (io.Reader, io.Reader, func() (int, int)) {
	f, e := &decCountingReader{}, &decCountingReader{}
	return f, e, func() (int, int) { return f.calls, e.calls }
}

// dbTables returns a pair of random tables, often the Annex K ones, with
// their fast tables built, or nil if processDHT rejects one.
func dbTables(rng *rand.Rand, std int) (dcH, acH *huffman) {
	dcC, dcV := decRandomTable(rng, dcTable)
	acC, acV := decRandomTable(rng, acTable)
	if rng.Intn(std) == 0 {
		s := decStdTables[dcTable][rng.Intn(2)]
		dcC, dcV = s.counts, s.vals
	}
	if rng.Intn(std) == 0 {
		s := decStdTables[acTable][rng.Intn(2)]
		acC, acV = s.counts, s.vals
	}
	dcH, acH = decTestTable(dcTable, dcC, dcV), decTestTable(acTable, acC, acV)
	if dcH == nil || acH == nil {
		return nil, nil
	}
	dcH.decBuildLUT(dcTable)
	acH.decBuildLUT(acTable)
	return dcH, acH
}

// dbHeld returns a random byte held in the accumulator, as the bytes it
// came from (0xFF is stuffed).
func dbHeld(rng *rand.Rand) (x byte, held []byte) {
	x = byte(rng.Intn(256))
	if rng.Intn(4) == 0 {
		x = 0xff
	}
	if x == 0xff {
		return x, []byte{0xff, 0x00}
	}
	return x, []byte{x}
}

// TestDecBlockSeqFromHeldByte starts both decoders from states that hold
// a whole byte in the accumulator (n = 8 to 15, nUnreadable 1 or 2), with
// random tables and streams, and a reader that fails.
func TestDecBlockSeqFromHeldByte(t *testing.T) {
	rng := rand.New(rand.NewSource(77))
	iters := 200000
	if testing.Short() {
		iters = 20000
	}
	for it := 0; it < iters; it++ {
		dcH, acH := dbTables(rng, 3)
		if dcH == nil {
			continue
		}
		stream := decTestStream(rng, 1+rng.Intn(40), rng.Intn(2) == 0)
		x, held := dbHeld(rng)
		var pre []byte
		for k := rng.Intn(3); k > 0; k-- {
			pre = append(pre, byte(rng.Intn(255)))
		}
		buf := append(append(append([]byte(nil), pre...), held...), stream...)
		r := int32(rng.Intn(8))
		s := dbState{
			buf: buf, i: len(pre) + len(held), nUnr: len(held), state: true,
			bits: bits{a: (rng.Uint32()&(1<<r-1))<<8 | uint32(x), n: 8 + r, m: 1 << (7 + r)},
		}
		fastR, exactR, calls := dbFailing()
		if d := dbCompare(dcH, acH, s, fastR, exactR, calls, 24); d != "" {
			t.Fatalf("iteration %d: %s\nbuffer % x, %d consumed, bits %+v", it, d, buf, s.i, s.bits)
		}
	}
}

// TestDecBlockSeqAcrossFill is TestDecBlockSeqFromHeldByte (with or
// without a held byte) with the rest of the stream coming from a reader of
// short chunks, so that the exact path refills and shifts the buffer
// between the fast blocks.
func TestDecBlockSeqAcrossFill(t *testing.T) {
	rng := rand.New(rand.NewSource(78))
	iters := 200000
	if testing.Short() {
		iters = 20000
	}
	for it := 0; it < iters; it++ {
		dcH, acH := dbTables(rng, 2)
		if dcH == nil {
			continue
		}
		stream := decTestStream(rng, 1+rng.Intn(400), rng.Intn(3) == 0)
		x, held := dbHeld(rng)
		r := int32(rng.Intn(16))
		var a uint32
		nUnr := 0
		if r >= 8 {
			a = (rng.Uint32()&(1<<(r-8)-1))<<8 | uint32(x)
			nUnr = len(held)
		} else {
			a = rng.Uint32() & (1<<r - 1)
			held = nil
		}
		pre := []byte{byte(rng.Intn(255)), byte(rng.Intn(255))}
		k := rng.Intn(len(stream) + 1)
		buf := append(append(append([]byte(nil), pre...), held...), stream[:k]...)
		chunk := 1 + rng.Intn(16)
		if rng.Intn(3) == 0 {
			chunk = 1 + rng.Intn(4096)
		}
		s := dbState{buf: buf, i: len(pre) + len(held), nUnr: nUnr, bits: bits{a: a, n: r}}
		if r > 0 {
			s.bits.m = 1 << (r - 1)
		}
		fastR, exactR := &dbChunkReader{b: stream[k:], n: chunk}, &dbChunkReader{b: stream[k:], n: chunk}
		calls := func() (int, int) { return fastR.calls, exactR.calls }
		if d := dbCompare(dcH, acH, s, fastR, exactR, calls, 40); d != "" {
			t.Fatalf("iteration %d: %s\nbuffer % x, then % x in chunks of %d", it, d, buf, stream[k:], chunk)
		}
	}
}

// dbExhaustive compares the decoders on every p-th stream of prefix bytes
// (two, or three if three) followed by each of the tails, from a canonical
// state that depends on the prefix.
func dbExhaustive(t *testing.T, name string, dcH, acH *huffman, three bool, step int, tails [][]byte, shift uint) int {
	limit := 1 << 16
	if three {
		limit = 1 << 22 // First bytes 0x00 to 0x3f.
	}
	n := 0
	for p := 0; p < limit; p += step {
		for ti, tail := range tails {
			stream := []byte{byte(p >> 8), byte(p)}
			if three {
				stream = []byte{byte(p >> 16), byte(p >> 8), byte(p)}
			}
			stream = append(stream, tail...)
			n0 := int32((p ^ ti) & 7)
			s := dbState{buf: stream, bits: bits{a: (uint32(p) * 2654435761) >> shift, n: n0}, state: true}
			if n0 > 0 {
				s.bits.m = 1 << (n0 - 1)
			}
			fastR, exactR, calls := dbFailing()
			if d := dbCompare(dcH, acH, s, fastR, exactR, calls, 16); d != "" {
				t.Fatalf("%s: stream % x, bits %+v: %s", name, stream, s.bits, d)
			}
			n++
		}
	}
	return n
}

// TestDecBlockSeqExhaustive tries every 2-byte stream prefix (every 127th
// with -short), followed by a few tails with stuffing and markers, with
// small hand-picked tables: over-subscribed ones, ones with excessive DC
// sizes, 12- and 16-bit codes only, EOB runs in a sequential scan.
func TestDecBlockSeqExhaustive(t *testing.T) {
	type table struct {
		counts [16]byte
		vals   []byte
	}
	dcs := []table{
		{[16]byte{2}, []byte{0, 1}},
		{[16]byte{2}, []byte{16, 3}},
		{[16]byte{1, 1}, []byte{0, 17}},
		{[16]byte{3}, []byte{1, 2, 9}}, // Over-subscribed.
		{[16]byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2}, []byte{5, 0}},
		{[16]byte{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}, []byte{9, 2}},
	}
	acs := []table{
		{[16]byte{1, 1, 2}, []byte{0x00, 0x01, 0xf0, 0x3f}},
		{[16]byte{1, 1, 2}, []byte{0x11, 0x00, 0xe5, 0x1a}},
		{[16]byte{2, 1}, []byte{0x00, 0x20, 0x01}},          // An EOB run.
		{[16]byte{1, 2, 1}, []byte{0x00, 0xf0, 0xff, 0x0f}}, // Over-subscribed.
		{[16]byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2}, []byte{0x00, 0x05, 0x0f, 0xe1}},
		{[16]byte{1}, []byte{0x00}},
		{[16]byte{1, 0, 0, 0, 0, 0, 0, 0, 1}, []byte{0x3c, 0x00}},
	}
	tails := [][]byte{{0x00}, {0xff}, {0xff, 0x00}, {0x7f, 0xff, 0x00, 0x80}, {0xaa, 0x55, 0xff, 0xd0}, {0x01, 0xfe}, {0xff, 0xff}, {}}
	step := 7
	if testing.Short() {
		step = 127
	}
	for di, dc := range dcs {
		for ai, ac := range acs {
			dcH, acH := decTestTable(dcTable, dc.counts, dc.vals), decTestTable(acTable, ac.counts, ac.vals)
			if dcH == nil || acH == nil {
				t.Fatalf("tables %d/%d rejected", di, ai)
			}
			dcH.decBuildLUT(dcTable)
			acH.decBuildLUT(acTable)
			dbExhaustive(t, fmt.Sprintf("tables %d/%d", di, ai), dcH, acH, false, step, tails, 7)
		}
	}
}

// TestDecBlockSeqExhaustiveStd is TestDecBlockSeqExhaustive with the
// Annex K tables: every 2-byte prefix for the chroma pair, and a sample of
// the 3-byte prefixes for the luma pair.
func TestDecBlockSeqExhaustiveStd(t *testing.T) {
	tails := [][]byte{{0x00}, {0xff, 0x00}, {0xff, 0xd0}, {0x12, 0xff}, {}}
	step := 1
	if testing.Short() {
		step = 13
	}
	for th := 0; th < 2; th++ {
		dcS, acS := decStdTables[dcTable][th], decStdTables[acTable][th]
		dcH, acH := decTestTable(dcTable, dcS.counts, dcS.vals), decTestTable(acTable, acS.counts, acS.vals)
		dcH.decBuildLUT(dcTable)
		acH.decBuildLUT(acTable)
		if th == 0 {
			dbExhaustive(t, "luma", dcH, acH, true, 7*step, tails[:1], 3)
		} else {
			dbExhaustive(t, "chroma", dcH, acH, false, step, tails, 3)
		}
	}
}

// dbAccInvariant checks decBitReader's invariant: the bits of acc below
// the nbits valid ones are zero or the stream's next bits (the unstuffed
// bytes of buf from pos on).
func dbAccInvariant(br *decBitReader, buf []byte) string {
	if br.nbits >= 64 {
		return ""
	}
	var next uint64
	got := uint(0)
	for pos := br.pos; got < 64 && pos < len(buf); got += 8 {
		c := buf[pos]
		if c == 0xff {
			if pos+1 >= len(buf) || buf[pos+1] != 0 {
				break
			}
			pos += 2
		} else {
			pos++
		}
		next |= uint64(c) << (56 - got)
	}
	if low := br.acc << br.nbits; low&^next != 0 {
		return fmt.Sprintf("acc %#x nbits %d: stray bits %#x (the next stream bits are %#x)", br.acc, br.nbits, low&^next, next)
	}
	return ""
}

// TestDecAccInvariant checks dbAccInvariant after every block the fast
// decoder accepts, on random tables and streams.
func TestDecAccInvariant(t *testing.T) {
	rng := rand.New(rand.NewSource(91))
	iters := 200000
	if testing.Short() {
		iters = 20000
	}
	for it := 0; it < iters; it++ {
		dcH, acH := dbTables(rng, 2)
		if dcH == nil {
			continue
		}
		stream := decTestStream(rng, 1+rng.Intn(200), rng.Intn(2) == 0)
		var d decoder
		copy(d.bytes.buf[:], stream)
		d.bytes.j = len(stream)
		d.r = &decCountingReader{}
		var br decBitReader
		br.load(&d)
		var dc int32
		for blk := 0; blk < 64; blk++ {
			var b block
			if !d.decBlockSeq(&br, &b, dcH, acH, &dc) {
				break
			}
			if s := dbAccInvariant(&br, stream); s != "" {
				t.Fatalf("iteration %d block %d: %s\nstream % x", it, blk, s, stream)
			}
		}
	}
}
