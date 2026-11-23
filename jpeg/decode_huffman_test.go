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

// Unit tests of the fast entropy decoder (decode_huffman.go) against the
// exact image/jpeg code of this package (decodeHuffman, receiveExtend,
// decBlockExact), on random and malformed Huffman tables and random
// bit streams with byte stuffing and markers.

import (
	"bytes"
	"fmt"
	"math/rand"
	"testing"
)

// decTestTable builds a Huffman table of class tc with processDHT. It
// returns nil if processDHT rejects it.
func decTestTable(tc int, counts [16]byte, vals []byte) *huffman {
	var d decoder
	seg := []byte{byte(tc << 4)}
	seg = append(seg, counts[:]...)
	seg = append(seg, vals...)
	d.r = bytes.NewReader(seg)
	if err := d.processDHT(len(seg)); err != nil {
		return nil
	}
	h := new(huffman)
	*h = d.huff[tc][0]
	return h
}

// decRandomTable returns a random Huffman table of class tc: well-formed
// (complete or not) or arbitrary (typically over-subscribed).
func decRandomTable(rng *rand.Rand, tc int) (counts [16]byte, vals []byte) {
	total := 0
	switch rng.Intn(4) {
	case 0, 1: // Well-formed: never more codes than a length has room for.
		want := 1 + rng.Intn(256)
		if rng.Intn(2) == 0 {
			want = 1 + rng.Intn(30)
		}
		avail := 2
		for l := 0; l < 16 && total < want; l++ {
			c := min(avail, want-total, 255)
			if l < 15 && rng.Intn(3) != 0 {
				c = rng.Intn(c + 1)
			}
			counts[l] = byte(c)
			total += c
			avail = (avail - c) * 2
		}
	case 2: // Arbitrary counts.
		for total == 0 {
			for l := range counts {
				if rng.Intn(3) == 0 {
					counts[l] = byte(rng.Intn(1 + rng.Intn(256)))
				}
			}
			total = 0
			for _, c := range counts {
				total += int(c)
			}
			if total > 256 {
				counts = [16]byte{}
				total = 0
			}
		}
	case 3: // A few short codes, maybe over-subscribed, and long ones.
		for total == 0 {
			for l := range counts {
				if rng.Intn(4) == 0 {
					counts[l] = byte(rng.Intn(5))
				}
			}
			total = 0
			for _, c := range counts {
				total += int(c)
			}
		}
	}
	vals = make([]byte, total)
	for i := range vals {
		switch r := rng.Intn(10); {
		case tc == dcTable && r < 8:
			vals[i] = byte(rng.Intn(12))
		case tc == dcTable && r < 9:
			vals[i] = byte(12 + rng.Intn(6)) // Around the 16 limit.
		case tc == acTable && r < 7:
			vals[i] = byte(rng.Intn(16)<<4 | (1 + rng.Intn(10)))
		case tc == acTable && r < 9:
			vals[i] = []byte{0x00, 0xf0, 0x10, 0x20, 0xe0, 0x0f, 0xff, 0x1f}[rng.Intn(8)]
		default:
			vals[i] = byte(rng.Intn(256))
		}
	}
	return counts, vals
}

func TestDecLUTConstructions(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	n := 3000
	if testing.Short() {
		n = 500
	}
	wellFormed := 0
	for i := 0; i < n; i++ {
		tc := i & 1
		counts, vals := decRandomTable(rng, tc)
		h := decTestTable(tc, counts, vals)
		if h == nil {
			continue
		}
		h.decLUT = new(decLUT)
		h.decSimulateLUT(tc)
		sim := *h.decLUT
		if !h.decWellFormed() {
			continue
		}
		wellFormed++
		h.decFillLUT(tc)
		if *h.decLUT != sim {
			for p := range sim {
				if h.decLUT[p] != sim[p] {
					t.Fatalf("table %v %v class %d: fill[%#x] = %#x, simulate = %#x", counts, vals, tc, p, h.decLUT[p], sim[p])
				}
			}
		}
	}
	if wellFormed < n/4 {
		t.Errorf("only %d well-formed tables", wellFormed)
	}
}

// decTestStream returns random entropy-coded bytes: mostly random data, with
// stuffed 0xFF bytes and, if markers is set, the odd marker.
func decTestStream(rng *rand.Rand, n int, markers bool) []byte {
	b := make([]byte, 0, n+2)
	zeroish := rng.Intn(3) == 0 // Long runs of short codes.
	for len(b) < n {
		switch r := rng.Intn(64); {
		case r == 0:
			b = append(b, 0xff, 0x00)
		case r == 1 && markers:
			b = append(b, 0xff, []byte{0xd0, 0xd9, 0xff, 0xc4, 0x01}[rng.Intn(5)])
		case zeroish && r < 40:
			b = append(b, byte(rng.Intn(4)))
		default:
			b = append(b, byte(rng.Intn(256)))
		}
	}
	return b
}

// decCountingReader fails every Read and counts the calls.
type decCountingReader struct{ calls int }

func (r *decCountingReader) Read(p []byte) (int, error) {
	r.calls++
	return 0, errTestReader
}

// decCanonical returns image/jpeg's state in canonical form (see
// decode_huffman.go): byte position, unread bit count and unread bits.
func decCanonical(d *decoder) (i int, n int32, a uint32) {
	i, n, a = d.bytes.i, d.bits.n, d.bits.a
	if n >= 8 {
		i -= d.bytes.nUnreadable
		n -= 8
		a >>= 8
	}
	return i, n, a & (1<<n - 1)
}

// TestDecBlockSeqVsExact runs the fast block decoder and the exact one side
// by side over random tables and streams. Whenever the fast decoder accepts
// a block, the exact code must decode the same coefficients and DC
// prediction without reading from the io.Reader and reach the same
// canonical state; when it declines, the scan loop's hand-over (store,
// exact block, load) must keep both in step.
func TestDecBlockSeqVsExact(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	iters := 20000
	if testing.Short() {
		iters = 3000
	}
	var accepted, declined, blocks int
	for it := 0; it < iters; it++ {
		dcC, dcV := decRandomTable(rng, dcTable)
		acC, acV := decRandomTable(rng, acTable)
		if rng.Intn(3) == 0 {
			// A realistic table pair.
			dcC, dcV = [16]byte{0, 1, 5, 1, 1, 1, 1, 1, 1}, []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
		}
		dcH := decTestTable(dcTable, dcC, dcV)
		acH := decTestTable(acTable, acC, acV)
		if dcH == nil || acH == nil {
			continue
		}
		dcH.decBuildLUT(dcTable)
		acH.decBuildLUT(acTable)

		stream := decTestStream(rng, 1+rng.Intn(300), rng.Intn(2) == 0)
		var fastD, exactD decoder
		copy(fastD.bytes.buf[:], stream)
		fastD.bytes.j = len(stream)
		n0 := int32(rng.Intn(8))
		fastD.bits = bits{a: rng.Uint32(), n: n0}
		if n0 > 0 {
			fastD.bits.m = 1 << (n0 - 1)
		}
		exactD.bytes, exactD.bits = fastD.bytes, fastD.bits
		fastR, exactR := &decCountingReader{}, &decCountingReader{}
		fastD.r, exactD.r = fastR, exactR

		var br decBitReader
		if !br.load(&fastD) {
			t.Fatal("load failed on a canonical state")
		}
		var dcF, dcE int32
		for blk := 0; blk < 64; blk++ {
			var bF, bE block
			// The exact decoder alone.
			err := exactD.decBlockExact(&bE, dcH, acH, &dcE, 0, blockSize-1, 0, 0)
			readE := exactR.calls != 0

			// The fast decoder with the scan loop's fallback.
			ok := fastD.eobRun == 0 && fastD.decBlockSeq(&br, &bF, dcH, acH, &dcF)
			var errF error
			if ok {
				accepted++
				if err != nil || readE {
					t.Fatalf("iter %d block %d: fast decoder accepted a block the exact one fails on (err %v, read %v)\ndc %v %v ac %v %v\nstream % x", it, blk, err, readE, dcC, dcV, acC, acV, stream)
				}
			} else {
				declined++
				bF = block{}
				br.store(&fastD)
				errF = fastD.decBlockExact(&bF, dcH, acH, &dcF, 0, blockSize-1, 0, 0)
				if errF == nil && !br.load(&fastD) {
					t.Fatalf("iter %d block %d: load failed after the exact path", it, blk)
				}
			}
			if !sameError(errF, err) || (fastR.calls != 0) != readE {
				t.Fatalf("iter %d block %d: error %v (read %v), want %v (read %v)", it, blk, errF, fastR.calls != 0, err, readE)
			}
			if err != nil || readE {
				break
			}
			blocks++
			if bF != bE || dcF != dcE || fastD.eobRun != exactD.eobRun {
				t.Fatalf("iter %d block %d: fast %v dc %d eob %d, exact %v dc %d eob %d\ndc %v %v ac %v %v\nstream % x", it, blk, bF, dcF, fastD.eobRun, bE, dcE, exactD.eobRun, dcC, dcV, acC, acV, stream)
			}
			var tmp decoder
			tmp.bytes = fastD.bytes
			br.store(&tmp)
			gi, gn, ga := decCanonical(&tmp)
			wi, wn, wa := decCanonical(&exactD)
			if gi != wi || gn != wn || ga != wa {
				t.Fatalf("iter %d block %d: state (%d, %d, %#x), want (%d, %d, %#x)\nstream % x", it, blk, gi, gn, ga, wi, wn, wa, stream)
			}
		}
	}
	t.Logf("%d blocks compared; fast path accepted %d, declined %d", blocks, accepted, declined)
	if accepted < blocks/4 {
		t.Errorf("fast path accepted only %d of %d blocks", accepted, blocks)
	}
}

// TestDecSlowSym checks decSlowSym and the table entries against
// decodeHuffman (and receiveExtend) on every table prefix followed by random
// bits, with every number of valid bits.
func TestDecSlowSym(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	n := 400
	if testing.Short() {
		n = 60
	}
	for i := 0; i < n; i++ {
		tc := i & 1
		counts, vals := decRandomTable(rng, tc)
		h := decTestTable(tc, counts, vals)
		if h == nil {
			continue
		}
		h.decBuildLUT(tc)
		for p := 0; p < 1<<decLUTBits; p += 1 + rng.Intn(4) {
			// 32 bits: the prefix, then random bits without 0xFF bytes.
			x := uint32(p)<<(32-decLUTBits) | rng.Uint32()>>decLUTBits
			var stream [4]byte
			for k := range stream {
				stream[k] = byte(x >> (24 - 8*k))
				if stream[k] == 0xff {
					stream[k] = 0xfe
				}
			}
			x = uint32(stream[0])<<24 | uint32(stream[1])<<16 | uint32(stream[2])<<8 | uint32(stream[3])
			q := x >> (32 - decLUTBits) // p, unless a 0xFF was replaced.
			acc := uint64(x) << 32
			// decodeHuffman with all 32 bits buffered.
			var d decoder
			copy(d.bytes.buf[:], stream[:])
			d.bytes.j = 4
			d.r = &decCountingReader{}
			sym, err := d.decodeHuffman(h)
			length := 32 - d.bits.n - 8*int32(4-d.bytes.i)
			for nbits := uint(8); nbits <= 32; nbits++ {
				got := decSlowSym(h, acc, nbits)
				if err != nil || uint(length) > nbits {
					if got != 0 {
						t.Fatalf("table %v %v: decSlowSym(%#x, %d) = %#x, want 0 (err %v, length %d)", counts, vals, x, nbits, got, err, length)
					}
					continue
				}
				if want := uint32(sym)<<8 | uint32(length); got != want {
					t.Fatalf("table %v %v: decSlowSym(%#x, %d) = %#x, want %#x", counts, vals, x, nbits, got, want)
				}
			}
			e := h.decLUT[q]
			if e == 0 {
				continue
			}
			if err != nil {
				t.Fatalf("table %v %v: entry %#x for prefix %#x, but decodeHuffman fails: %v", counts, vals, e, q, err)
			}
			nb := e & 0xff
			s := uint32(sym)
			if tc == acTable {
				s &= 15
			}
			var val int32
			if s != 0 {
				val, err = d.receiveExtend(uint8(s))
				if err != nil {
					t.Fatal(err)
				}
			}
			switch {
			case tc == dcTable && e>>8&0xff == 0:
				if nb != uint32(length)+s || int32(e)>>16 != val {
					t.Fatalf("table %v %v: DC entry %#x for %#x, want length %d size %d value %d", counts, vals, e, x, length, s, val)
				}
			case tc == dcTable:
				if nb != uint32(length) || e>>8&0xff != s || e>>16 != 0 {
					t.Fatalf("table %v %v: DC entry %#x for %#x, want length %d size %d", counts, vals, e, x, length, s)
				}
			case int32(e)>>16 != 0:
				if nb != uint32(length)+s || int32(e)>>16 != val || e>>8&0xff != uint32(sym) {
					t.Fatalf("table %v %v: AC entry %#x for %#x, want length %d sym %#x value %d", counts, vals, e, x, length, sym, val)
				}
			default:
				if nb != uint32(length) || e>>8&0xff != uint32(sym) || (s != 0 && uint32(length)+s <= decLUTBits) {
					t.Fatalf("table %v %v: AC entry %#x for %#x, want length %d sym %#x", counts, vals, e, x, length, sym)
				}
			}
		}
	}
}

// decBenchScan describes the first scan of a benchmark file.
type decBenchScan struct {
	d       decoder
	data    []byte // The input from the start of the entropy-coded data.
	comps   []decScanComp
	mxx     int
	myy     int
	nBlocks int
}

func decLoadBenchScan(b *testing.B, path string) *decBenchScan {
	data := readFile(b, path)
	s := &decBenchScan{}
	if _, err := s.d.decode(bytes.NewReader(data), false); err != nil {
		b.Fatal(err)
	}
	if s.d.progressive || s.d.ri != 0 {
		b.Skip("not a plain sequential file")
	}
	es := entropyStart(data)
	// Parse the SOS segment just before es.
	for i := es - 1; i > 2; i-- {
		if data[i] == 0xff && data[i+1] == sosMarker {
			nc := int(data[i+4])
			for k := 0; k < nc; k++ {
				cs, t := data[i+5+2*k], data[i+6+2*k]
				for ci := 0; ci < s.d.nComp; ci++ {
					if s.d.comp[ci].c == cs {
						s.comps = append(s.comps, decScanComp{uint8(ci), t >> 4, t & 15})
					}
				}
			}
			break
		}
	}
	s.data = data[es:]
	s.mxx = (s.d.width + 8*s.d.maxH - 1) / (8 * s.d.maxH)
	s.myy = (s.d.height + 8*s.d.maxV - 1) / (8 * s.d.maxV)
	for _, c := range s.comps {
		s.nBlocks += s.mxx * s.myy * s.d.comp[c.compIndex].h * s.d.comp[c.compIndex].v
	}
	return s
}

// run decodes the whole scan without reconstructing it, with the fast path
// or the exact one alone.
func (s *decBenchScan) run(b *testing.B, fast bool) {
	d := &s.d
	d.r = bytes.NewReader(s.data)
	d.bytes.i, d.bytes.j, d.bytes.nUnreadable = 0, 0, 0
	d.bits = bits{}
	d.eobRun = 0
	var br decBitReader
	br.load(d)
	var dc [maxComponents]int32
	var blk block
	for m := 0; m < s.mxx*s.myy; m++ {
		for _, c := range s.comps {
			ci := int(c.compIndex)
			dcH, acH := &d.huff[dcTable][c.td], &d.huff[acTable][c.ta]
			for k := 0; k < d.comp[ci].h*d.comp[ci].v; k++ {
				blk = block{}
				if !fast || d.eobRun != 0 || !d.decBlockSeq(&br, &blk, dcH, acH, &dc[ci]) {
					if fast {
						blk = block{}
						br.store(d)
					}
					if err := d.decBlockExact(&blk, dcH, acH, &dc[ci], 0, blockSize-1, 0, 0); err != nil {
						b.Fatal(err)
					}
					if fast {
						br.load(d)
					}
				}
			}
		}
	}
}

// BenchmarkDecEntropy measures entropy decoding alone (no dequantization,
// IDCT or allocation): impl=exact is image/jpeg's decoder, impl=fast this
// package's. MB/s is megapixels per second.
func BenchmarkDecEntropy(b *testing.B) {
	for _, f := range benchDecodeFiles {
		b.Run("img="+f.name, func(b *testing.B) {
			s := decLoadBenchScan(b, f.path)
			for _, impl := range []string{"exact", "fast"} {
				b.Run("impl="+impl, func(b *testing.B) {
					b.SetBytes(int64(s.d.width * s.d.height))
					for b.Loop() {
						s.run(b, impl == "fast")
					}
				})
			}
		})
	}
}

// TestDecBlockProgVsExact is TestDecBlockSeqVsExact for the progressive
// block decoders (decode_progressive.go): DC first and refinement scans, AC
// first scans with EOB runs, and AC refinement scans over random existing
// coefficients.
func TestDecBlockProgVsExact(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	iters := 20000
	if testing.Short() {
		iters = 3000
	}
	var accepted, declined, blocks [4]int
	for it := 0; it < iters; it++ {
		kind := it % 4
		var zigStart, zigEnd int32
		var ah, al uint32
		switch kind {
		case decDCFirst:
			al = uint32(rng.Intn(16))
		case decDCRefine:
			al = uint32(rng.Intn(15))
			ah = al + 1
		case decACFirst, decACRefine:
			zigStart = int32(1 + rng.Intn(63))
			zigEnd = zigStart + int32(rng.Intn(int(64-zigStart)))
			if rng.Intn(3) == 0 {
				zigStart, zigEnd = 1, 63
			}
			al = uint32(rng.Intn(16))
			if kind == decACRefine {
				al = uint32(rng.Intn(15))
				ah = al + 1
			}
		}
		dcC, dcV := decRandomTable(rng, dcTable)
		acC, acV := decRandomTable(rng, acTable)
		if kind == decACRefine && rng.Intn(2) == 0 {
			// Refinement tables hold sizes 0 and 1 only.
			for i := range acV {
				acV[i] &= 0xf1
			}
		}
		dcH := decTestTable(dcTable, dcC, dcV)
		acH := decTestTable(acTable, acC, acV)
		if dcH == nil || acH == nil {
			continue
		}
		dcH.decBuildLUT(dcTable)
		acH.decBuildLUT(acTable)

		stream := decTestStream(rng, 1+rng.Intn(300), rng.Intn(2) == 0)
		var fastD, exactD decoder
		copy(fastD.bytes.buf[:], stream)
		fastD.bytes.j = len(stream)
		n0 := int32(rng.Intn(8))
		fastD.bits = bits{a: rng.Uint32(), n: n0}
		if n0 > 0 {
			fastD.bits.m = 1 << (n0 - 1)
		}
		if rng.Intn(4) == 0 {
			fastD.eobRun = uint16(rng.Intn(4))
		}
		exactD.bytes, exactD.bits, exactD.eobRun = fastD.bytes, fastD.bits, fastD.eobRun
		fastR, exactR := &decCountingReader{}, &decCountingReader{}
		fastD.r, exactD.r = fastR, exactR

		var br decBitReader
		if !br.load(&fastD) {
			t.Fatal("load failed on a canonical state")
		}
		var dcF, dcE int32
		delta := int32(1) << al
		for blk := 0; blk < 64; blk++ {
			// The block's previous coefficients.
			var bE block
			if kind == decACRefine || kind == decDCRefine || rng.Intn(4) == 0 {
				for k := range bE {
					if rng.Intn(3) == 0 {
						bE[k] = int32(rng.Intn(9)-4) * delta * 2
					}
				}
			}
			bF := bE
			orig := bE
			// The non-zero mask, given or not.
			var nzp *uint64
			nzm := decNonZero(&bF)
			if blk%2 == 0 {
				nzp = &nzm
			}
			err := exactD.decBlockExact(&bE, dcH, acH, &dcE, zigStart, zigEnd, ah, al)
			readE := exactR.calls != 0

			ok := false
			switch kind {
			case decDCFirst:
				ok = fastD.decBlockDCFirst(&br, &bF, dcH, &dcF, uint(al))
			case decDCRefine:
				ok = fastD.decBlockDCRefine(&br, &bF, delta)
			case decACFirst:
				if fastD.eobRun > 0 {
					fastD.eobRun--
					ok = true
				} else {
					ok = fastD.decBlockACFirst(&br, &bF, acH, uint(zigStart), uint(zigEnd), uint(al), nzp)
				}
			case decACRefine:
				ok = fastD.decBlockACRefine(&br, &bF, acH, uint(zigStart), uint(zigEnd), delta, nzp)
			}
			var errF error
			if ok {
				accepted[kind]++
				if err != nil || readE {
					t.Fatalf("kind %d iter %d block %d: fast decoder accepted a block the exact one fails on (err %v, read %v)\nss %d se %d ah %d al %d\nac %v %v\nstream % x", kind, it, blk, err, readE, zigStart, zigEnd, ah, al, acC, acV, stream)
				}
			} else {
				declined[kind]++
				if kind == decACRefine && bF != orig {
					t.Fatalf("kind %d iter %d block %d: block changed by a declined refinement", kind, it, blk)
				}
				if kind == decACFirst || kind == decDCFirst {
					// Their partial writes are rewritten by the exact path.
				} else {
					bF = orig
				}
				br.store(&fastD)
				errF = fastD.decBlockExact(&bF, dcH, acH, &dcF, zigStart, zigEnd, ah, al)
				if errF == nil && !br.load(&fastD) {
					t.Fatalf("iter %d block %d: load failed after the exact path", it, blk)
				}
			}
			if !sameError(errF, err) || (fastR.calls != 0) != readE {
				t.Fatalf("kind %d iter %d block %d: error %v (read %v), want %v (read %v)", kind, it, blk, errF, fastR.calls != 0, err, readE)
			}
			if err != nil || readE {
				break
			}
			blocks[kind]++
			if ok && nzp != nil && nzm != decNonZero(&bF) {
				t.Fatalf("kind %d iter %d block %d: non-zero mask %#x, want %#x", kind, it, blk, nzm, decNonZero(&bF))
			}
			if bF != bE || dcF != dcE || fastD.eobRun != exactD.eobRun {
				t.Fatalf("kind %d iter %d block %d: fast %v dc %d eob %d, exact %v dc %d eob %d\nss %d se %d ah %d al %d\nac %v %v\nstream % x", kind, it, blk, bF, dcF, fastD.eobRun, bE, dcE, exactD.eobRun, zigStart, zigEnd, ah, al, acC, acV, stream)
			}
			var tmp decoder
			tmp.bytes = fastD.bytes
			br.store(&tmp)
			gi, gn, ga := decCanonical(&tmp)
			wi, wn, wa := decCanonical(&exactD)
			if gi != wi || gn != wn || ga != wa {
				t.Fatalf("kind %d iter %d block %d: state (%d, %d, %#x), want (%d, %d, %#x)\nstream % x", kind, it, blk, gi, gn, ga, wi, wn, wa, stream)
			}
		}
	}
	for k := range blocks {
		t.Logf("kind %d: %d blocks compared; fast path accepted %d, declined %d", k, blocks[k], accepted[k], declined[k])
		if accepted[k] < blocks[k]/4 {
			t.Errorf("kind %d: fast path accepted only %d of %d blocks", k, accepted[k], blocks[k])
		}
	}
}

func BenchmarkDecBuildLUT(b *testing.B) {
	for _, tc := range []int{dcTable, acTable} {
		s := decStdTables[tc][0]
		h := decTestTable(tc, s.counts, s.vals)
		b.Run(fmt.Sprintf("class=%d/fill", tc), func(b *testing.B) {
			for b.Loop() {
				h.decBuildLUT(tc)
			}
		})
		b.Run(fmt.Sprintf("class=%d/simulate", tc), func(b *testing.B) {
			for b.Loop() {
				h.decSimulateLUT(tc)
			}
		})
	}
}

// BenchmarkDecSmall decodes small images, where the fixed costs (tables,
// allocation) matter.
func BenchmarkDecSmall(b *testing.B) {
	for _, f := range []string{"photo16x16_420_base.jpeg", "photo33x31_420_base.jpeg", "video150x103_221122_base.jpeg", "photo257x131_rst8.jpeg"} {
		data := readFile(b, "../testdata/corpus/"+f)
		for _, impl := range benchImpls {
			b.Run("img="+f+"/impl="+impl.name, func(b *testing.B) {
				for b.Loop() {
					if _, err := impl.decode(bytes.NewReader(data)); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
