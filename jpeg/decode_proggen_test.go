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

// Generated progressive JPEG streams against image/jpeg. The progressive
// fast path (decode_progressive.go, decScanProg) is the decoder's most
// intricate code: an undo log for refinements it gives up, per-block
// non-zero masks that stay valid only while every scan goes through it,
// and EOB runs that span blocks, scans and restart intervals. The corpus's
// progressive files all use cjpeg's default script (Al at most 2, the
// standard bands), and TestDecBlockProgVsExact works block by block; the
// generator here covers the scan level: random scan scripts (DC first and
// refinement scans, interleaved or not, in any component order; AC bands
// with Al up to 15 and Ah = Al+1 chains), refinements without a first
// scan, repeated, overlapping and invalid scans, tables (DHT, DQT, DRI)
// redefined or left undefined between scans, and restart intervals. The
// entropy-coded data comes from a model of image/jpeg's decoder (the
// non-zero history of every coefficient, the EOB run), so that most
// streams decode far enough to reach those paths; a few are mutated or
// truncated. From the verification of the progressive decoder.

import (
	"bytes"
	"encoding/binary"
	"fmt"
	std "image/jpeg"
	"io"
	"math/rand"
	"testing"
)

// pgTable is a Huffman table of the generator, with the code of every
// value (the first one, if a value appears twice).
type pgTable struct {
	counts  [16]byte
	vals    []byte
	codes   []uint32
	lengths []uint
	idx     map[byte]int
}

func pgTableOf(counts [16]byte, vals []byte) *pgTable {
	t := &pgTable{counts: counts, vals: vals}
	t.codes, t.lengths = decCodes(t.counts)
	t.idx = map[byte]int{}
	for i, v := range t.vals {
		if _, ok := t.idx[v]; !ok {
			t.idx[v] = i
		}
	}
	return t
}

// pgSplit assigns code lengths to n symbols: a random complete prefix code
// no deeper than 16 bits, skewed now and then to give long codes.
func pgSplit(rng *rand.Rand, n, depth int, out []int) {
	if n == 1 {
		out[0] = max(depth, 1)
		return
	}
	lim := 1 << (15 - depth) // The most symbols a subtree below can hold.
	lo, hi := max(1, n-lim), min(n-1, lim)
	k := lo + rng.Intn(hi-lo+1)
	if rng.Intn(3) == 0 {
		if rng.Intn(2) == 0 {
			k = lo
		} else {
			k = hi
		}
	}
	pgSplit(rng, k, depth+1, out[:k])
	pgSplit(rng, n-k, depth+1, out[k:])
}

// pgMakeTable returns a random table of the symbols syms: a complete code,
// or (unless clean) now and then one with a code missing.
func pgMakeTable(rng *rand.Rand, syms []byte, clean bool) *pgTable {
	s := append([]byte(nil), syms...)
	rng.Shuffle(len(s), func(i, j int) { s[i], s[j] = s[j], s[i] })
	ls := make([]int, len(s))
	pgSplit(rng, len(s), 0, ls)
	var counts [16]byte
	var vals []byte
	for l := 1; l <= 16; l++ {
		for i, v := range s {
			if ls[i] == l {
				vals = append(vals, v)
				counts[l-1]++
			}
		}
	}
	if rng.Intn(8) == 0 && counts[15] == 0 && !clean {
		// Drop the last code of the longest length that has two: an
		// incomplete code.
		for l := 15; l >= 0; l-- {
			if counts[l] > 1 {
				counts[l]--
				pos := 0
				for k := 0; k <= l; k++ {
					pos += int(counts[k])
				}
				vals = append(vals[:pos], vals[pos+1:]...)
				break
			}
		}
	}
	return pgTableOf(counts, vals)
}

func (t *pgTable) has(v byte) bool { _, ok := t.idx[v]; return ok }

func (t *pgTable) emit(w *decJPEGBuilder, v byte) {
	i := t.idx[v]
	w.bits(t.codes[i], t.lengths[i])
}

func (t *pgTable) payload(tc, th int) []byte {
	p := []byte{byte(tc<<4 | th)}
	p = append(p, t.counts[:]...)
	return append(p, t.vals...)
}

// pgDCSyms returns the symbols of a DC table: sizes 0 to 11 (or 16), and
// (unless clean) now and then an excessive one.
func pgDCSyms(rng *rand.Rand, clean bool) []byte {
	n := 12
	if rng.Intn(4) == 0 {
		n = 17
	}
	var s []byte
	for i := 0; i < n; i++ {
		s = append(s, byte(i))
	}
	if rng.Intn(40) == 0 && !clean {
		s = append(s, byte(17+rng.Intn(100)))
	}
	return s
}

// pgACFirstSyms returns the symbols of an AC first-scan table: most
// run/size pairs, some EOB runs, and ZRL.
func pgACFirstSyms(rng *rand.Rand) []byte {
	maxS := 10
	if rng.Intn(4) == 0 {
		maxS = 15
	}
	var s []byte
	for r := 0; r < 16; r++ {
		for sz := 1; sz <= maxS; sz++ {
			if rng.Intn(6) == 0 && !(r == 0 && sz <= 2) {
				continue
			}
			s = append(s, byte(r<<4|sz))
		}
	}
	for r := 0; r < 15; r++ {
		if r == 0 || rng.Intn(3) != 0 {
			s = append(s, byte(r<<4))
		}
	}
	s = append(s, 0xf0)
	if len(s) > 250 {
		s = s[:250]
	}
	return s
}

// pgRefineSyms returns the symbols of an AC refinement table: the runs
// with size 1, some EOB runs, ZRL and (unless clean) now and then an
// invalid size.
func pgRefineSyms(rng *rand.Rand, clean bool) []byte {
	var s []byte
	for r := 0; r < 16; r++ {
		s = append(s, byte(r<<4|1))
	}
	for r := 0; r < 15; r++ {
		if r == 0 || rng.Intn(3) != 0 {
			s = append(s, byte(r<<4))
		}
	}
	s = append(s, 0xf0)
	if rng.Intn(10) == 0 && !clean {
		s = append(s, byte(rng.Intn(16)<<4|(2+rng.Intn(14))))
	}
	return s
}

type pgComp struct{ h, v, tq int }

// pgGen holds the state of one generated stream: the frame, the tables in
// force, and the model of image/jpeg's decoder (the non-zero history of
// every block's coefficients, and the EOB run).
type pgGen struct {
	rng     *rand.Rand
	w       *decJPEGBuilder
	clean   bool // No deliberate invalidity.
	extreme bool // Huge coefficients and quantization values.
	inject  bool // One deliberate error in the entropy-coded data.
	eobP    int  // The chance, in percent, of an EOB at each step.
	comps   []pgComp
	width   int
	height  int
	mxx     int
	myy     int
	ri      int
	nz      [4][]uint64
	eobRun  int
	tables  [2][4]*pgTable
}

// emitEOB writes an EOB-run symbol of t and its run bits, and returns the
// run length; it returns false if t has no EOB symbol.
func (g *pgGen) emitEOB(t *pgTable) (run int, ok bool) {
	var cands []int
	for r := 0; r < 15; r++ {
		if t.has(byte(r << 4)) {
			cands = append(cands, r)
		}
	}
	if len(cands) == 0 {
		return 0, false
	}
	r := cands[0]
	if g.rng.Intn(20) >= 12 {
		r = cands[g.rng.Intn(len(cands))]
	}
	if g.rng.Intn(50) == 0 {
		r = cands[len(cands)-1]
	}
	t.emit(g.w, byte(r<<4))
	run = 1 << r
	if r > 0 {
		x := g.rng.Uint32() & (1<<r - 1)
		g.w.bits(x, uint(r))
		run |= int(x)
	}
	return run, true
}

func (g *pgGen) dcFirstBlock(t *pgTable) {
	// Mostly small sizes.
	var v byte
	for k := 0; k < 10; k++ {
		v = t.vals[g.rng.Intn(len(t.vals))]
		if v <= 11 || g.rng.Intn(30) == 0 || (g.extreme && v >= 12 && v <= 16) {
			break
		}
	}
	t.emit(g.w, v)
	g.w.bits(g.rng.Uint32(), uint(min(v, 16)))
}

func (g *pgGen) acFirstBlock(t *pgTable, nz *uint64, ss, se int) {
	if g.eobRun > 0 {
		g.eobRun--
		return
	}
	for zig := ss; zig <= se; {
		if g.rng.Intn(100) < g.eobP || (zig > ss+20 && g.rng.Intn(3) == 0) {
			if run, ok := g.emitEOB(t); ok {
				g.eobRun = run - 1
				return
			}
		}
		if zig+15 <= se && g.rng.Intn(12) == 0 && t.has(0xf0) {
			t.emit(g.w, 0xf0)
			zig += 16
			continue
		}
		// A coefficient symbol that fits, with a small size mostly; or
		// any symbol, with its image/jpeg meaning.
		var v byte
		found := false
		for k := 0; k < 12 && !found; k++ {
			v = t.vals[g.rng.Intn(len(t.vals))]
			r, s := int(v>>4), int(v&15)
			found = s != 0 && (zig+r <= se || g.rng.Intn(40) == 0) &&
				(s <= 3 || g.rng.Intn(4) == 0 || (g.extreme && s >= 9))
		}
		if !found {
			v = t.vals[g.rng.Intn(len(t.vals))]
		}
		t.emit(g.w, v)
		r, s := int(v>>4), int(v&15)
		if s == 0 {
			if r != 15 {
				run := 1 << r
				if r > 0 {
					x := g.rng.Uint32() & (1<<r - 1)
					g.w.bits(x, uint(r))
					run |= int(x)
				}
				g.eobRun = run - 1
				return
			}
			zig += 16
			continue
		}
		zig += r
		if zig > se {
			return
		}
		g.w.bits(g.rng.Uint32(), uint(s))
		*nz |= 1 << uint(zig)
		zig++
	}
}

// corrections writes the correction bits of the non-zero coefficients of
// nz in [from, to).
func (g *pgGen) corrections(nz uint64, from, to int) {
	for p := from; p < to; p++ {
		if nz>>uint(p)&1 != 0 {
			g.w.bits(g.rng.Uint32(), 1)
		}
	}
}

func (g *pgGen) acRefineBlock(t *pgTable, nz *uint64, ss, se int) {
	if g.eobRun > 0 {
		g.eobRun--
		g.corrections(*nz, ss, se+1)
		return
	}
	for zig := ss; zig <= se; {
		var zeros []int
		for p := zig; p <= se; p++ {
			if *nz>>uint(p)&1 == 0 {
				zeros = append(zeros, p)
			}
		}
		if len(zeros) == 0 || g.rng.Intn(100) < g.eobP || (zig > ss+20 && g.rng.Intn(3) == 0) {
			if run, ok := g.emitEOB(t); ok {
				g.corrections(*nz, zig, se+1)
				g.eobRun = run - 1
				return
			}
		}
		if g.inject && g.rng.Intn(200) == 0 {
			// An arbitrary symbol: errors and odd paths.
			g.inject = false
			t.emit(g.w, t.vals[g.rng.Intn(len(t.vals))])
			g.w.bits(g.rng.Uint32(), 16)
			return
		}
		if len(zeros) >= 16 && g.rng.Intn(10) == 0 && t.has(0xf0) {
			t.emit(g.w, 0xf0)
			stop := zeros[15]
			g.corrections(*nz, zig, stop)
			zig = stop + 1
			continue
		}
		if len(zeros) == 0 {
			// No EOB symbol: a new coefficient anyway ("too many
			// coefficients").
			if t.has(0x01) {
				t.emit(g.w, 0x01)
				g.w.bits(g.rng.Uint32(), 1)
				g.corrections(*nz, zig, se+1)
			}
			return
		}
		r := g.rng.Intn(min(16, len(zeros)))
		if g.rng.Intn(2) == 0 {
			r = min(r, g.rng.Intn(3))
		}
		v := byte(r<<4 | 1)
		if !t.has(v) {
			found := false
			for rr := 0; rr < min(16, len(zeros)) && !found; rr++ {
				if t.has(byte(rr<<4 | 1)) {
					r, v, found = rr, byte(rr<<4|1), true
				}
			}
			if !found {
				return
			}
		}
		t.emit(g.w, v)
		g.w.bits(g.rng.Uint32(), 1) // The sign.
		stop := zeros[r]
		g.corrections(*nz, zig, stop)
		*nz |= 1 << uint(stop)
		zig = stop + 1
	}
}

// pgScan is a scan of the script: its components (indexes into the
// frame's), the band and the successive approximation.
type pgScan struct {
	comps          []int
	ss, se, ah, al int
}

// scanData writes the entropy-coded data of scan sc, whose components use
// the tables sel (DC, AC), in image/jpeg's block order, with restart
// markers (some of them odd, if g.inject).
func (g *pgGen) scanData(sc pgScan, sel [][2]int) {
	kind := decDCFirst
	switch {
	case sc.ss == 0 && sc.ah != 0:
		kind = decDCRefine
	case sc.ss != 0 && sc.ah == 0:
		kind = decACFirst
	case sc.ss != 0:
		kind = decACRefine
	}
	for _, ci := range sc.comps {
		if g.nz[ci] == nil {
			g.nz[ci] = make([]uint64, g.mxx*g.myy*g.comps[ci].h*g.comps[ci].v)
		}
	}
	se := min(sc.se, 63)
	blockCount, mcu, rst := 0, 0, 0
	for my := 0; my < g.myy; my++ {
		for mx := 0; mx < g.mxx; mx++ {
			for i, ci := range sc.comps {
				hi, vi := g.comps[ci].h, g.comps[ci].v
				for j := 0; j < hi*vi; j++ {
					bx, by := hi*mx+j%hi, vi*my+j/hi
					if len(sc.comps) == 1 {
						q := g.mxx * hi
						bx, by = blockCount%q, blockCount/q
						blockCount++
						if bx*8 >= g.width || by*8 >= g.height {
							continue
						}
					}
					k := by*g.mxx*hi + bx
					dcT, acT := g.tables[0][sel[i][0]], g.tables[1][sel[i][1]]
					switch kind {
					case decDCFirst:
						if dcT != nil {
							g.dcFirstBlock(dcT)
						}
					case decDCRefine:
						g.w.bits(g.rng.Uint32(), 1)
					case decACFirst:
						if acT != nil && k < len(g.nz[ci]) {
							g.acFirstBlock(acT, &g.nz[ci][k], sc.ss, se)
						} else if g.eobRun > 0 {
							g.eobRun--
						}
					case decACRefine:
						if acT != nil && k < len(g.nz[ci]) {
							g.acRefineBlock(acT, &g.nz[ci][k], sc.ss, se)
						} else if g.eobRun > 0 && k < len(g.nz[ci]) {
							g.eobRun--
							g.corrections(g.nz[ci][k], sc.ss, se+1)
						}
					}
				}
			}
			mcu++
			if g.ri > 0 && mcu%g.ri == 0 && mcu < g.mxx*g.myy {
				g.w.flush()
				if g.inject && g.rng.Intn(30) == 0 {
					// Garbage before the marker, or a wrong one.
					g.inject = false
					switch g.rng.Intn(4) {
					case 0:
						g.w.b = append(g.w.b, 0x12, 0x34)
					case 1:
						g.w.b = append(g.w.b, 0xff, 0x00)
					case 2:
						rst++
					case 3:
						g.w.b = append(g.w.b, 0xff)
					}
				}
				if g.rng.Intn(40) == 0 {
					g.w.b = append(g.w.b, 0xff, 0xff) // Fill bytes.
				}
				g.w.b = append(g.w.b, 0xff, byte(rst0Marker+rst%8))
				rst++
				g.eobRun = 0
			}
		}
	}
	if g.ri > 0 && g.rng.Intn(10) == 0 {
		// A restart marker after the last MCU, as some encoders write.
		g.w.flush()
		g.w.b = append(g.w.b, 0xff, byte(rst0Marker+rst%8))
	}
	g.w.flush()
}

func (g *pgGen) dqt() {
	rng := g.rng
	var p []byte
	for tq := 0; tq < 4; tq++ {
		if rng.Intn(2) == 0 && tq > 0 {
			continue
		}
		if rng.Intn(8) == 0 || g.extreme {
			p = append(p, byte(0x10|tq))
			for i := 0; i < 64; i++ {
				hi := byte(rng.Intn(3))
				if g.extreme && rng.Intn(2) == 0 {
					hi = byte(rng.Intn(256))
				}
				p = append(p, hi, byte(rng.Intn(256)))
			}
			continue
		}
		p = append(p, byte(tq))
		for i := 0; i < 64; i++ {
			p = append(p, byte(1+rng.Intn(1+rng.Intn(255))))
		}
	}
	g.w.segment(dqtMarker, p)
}

// pgRandomJPEG builds a random progressive JPEG of at most maxDim x maxDim
// pixels. With clean, it has no deliberate invalidity (and image/jpeg
// decodes most such streams); with extreme, it uses 16-bit quantization
// tables of any value, large coefficients and large point transforms.
func pgRandomJPEG(rng *rand.Rand, maxDim int, clean, extreme bool) []byte {
	w := &decJPEGBuilder{}
	g := &pgGen{rng: rng, w: w, clean: clean}
	g.eobP = []int{5, 10, 20, 35, 60}[rng.Intn(5)]
	g.inject = rng.Intn(6) == 0 && !clean
	g.extreme = extreme || rng.Intn(10) == 0
	w.b = append(w.b, 0xff, soiMarker)
	g.dqt()

	nComp := []int{1, 1, 3, 3, 3, 4}[rng.Intn(6)]
	width, height := 1+rng.Intn(maxDim), 1+rng.Intn(maxDim)
	if rng.Intn(3) == 0 {
		width, height = maxDim/2+rng.Intn(maxDim/2+1), maxDim/2+rng.Intn(maxDim/2+1)
	}
	g.width, g.height = width, height
	comps := make([]pgComp, nComp)
	hvs := []int{1, 1, 1, 2, 2, 4}
	for i := range comps {
		comps[i] = pgComp{hvs[rng.Intn(len(hvs))], hvs[rng.Intn(len(hvs))], rng.Intn(2)}
	}
	if nComp == 3 && rng.Intn(4) != 0 {
		comps[0].h, comps[0].v = []int{1, 2, 2, 4, 1}[rng.Intn(5)], []int{1, 1, 2, 2, 2}[rng.Intn(5)]
		comps[1].h, comps[1].v, comps[2].h, comps[2].v = 1, 1, 1, 1
	}
	if nComp == 4 {
		hv := []int{1, 2}[rng.Intn(2)]
		comps[0].h, comps[0].v, comps[3].h, comps[3].v = hv, hv, hv, hv
		comps[1].h, comps[1].v, comps[2].h, comps[2].v = 1, 1, 1, 1
		w.segment(app14Marker, []byte{'A', 'd', 'o', 'b', 'e', 0, 100, 0, 0, 0, 0, byte(rng.Intn(3))})
	}
	sof := []byte{8, byte(height >> 8), byte(height), byte(width >> 8), byte(width), byte(nComp)}
	maxH, maxV := 1, 1
	for i, c := range comps {
		sof = append(sof, byte(i+1), byte(c.h<<4|c.v), byte(c.tq))
		maxH, maxV = max(maxH, c.h), max(maxV, c.v)
	}
	if nComp == 1 {
		// How the decoder treats a single component.
		maxH, maxV = 1, 1
		comps[0].h, comps[0].v = 1, 1
	}
	w.segment(sof2Marker, sof)
	g.comps = comps
	g.mxx, g.myy = (width+8*maxH-1)/(8*maxH), (height+8*maxV-1)/(8*maxV)

	if rng.Intn(3) == 0 {
		g.ri = 1 + rng.Intn(1+rng.Intn(12))
		if rng.Intn(10) == 0 {
			g.ri = 0
		}
		w.segment(driMarker, []byte{byte(g.ri >> 8), byte(g.ri)})
	}

	scans := pgScript(rng, nComp, clean, g.extreme)

	for si, sc := range scans {
		// The tables of this scan: defined, redefined, or (unless clean)
		// now and then left undefined.
		kindDC := sc.ss == 0 && sc.ah == 0
		kindRef := sc.ah != 0
		undefined := rng.Intn(25) == 0 && !clean
		var sel [][2]int
		for range sc.comps {
			s := [2]int{rng.Intn(4), rng.Intn(4)}
			if rng.Intn(3) != 0 {
				s = [2]int{0, 0}
			}
			if undefined {
				for x := 0; x < 2; x++ {
					for th := 3; th >= 0; th-- {
						if g.tables[x][th] == nil {
							s[x] = th
							break
						}
					}
				}
			}
			sel = append(sel, s)
		}
		var p []byte
		for _, s := range sel {
			if undefined {
				break
			}
			switch {
			case kindDC:
				if g.tables[0][s[0]] == nil || rng.Intn(3) == 0 {
					var t *pgTable
					if rng.Intn(12) == 0 && !clean {
						t = pgTableOf(decRandomTable(rng, dcTable))
					} else {
						t = pgMakeTable(rng, pgDCSyms(rng, clean), clean)
					}
					g.tables[0][s[0]] = t
					p = append(p, t.payload(0, s[0])...)
				}
			case sc.ss != 0:
				if g.tables[1][s[1]] == nil || rng.Intn(4) != 0 {
					var t *pgTable
					switch {
					case rng.Intn(14) == 0 && !clean:
						cnt, vals := decRandomTable(rng, acTable)
						if kindRef && rng.Intn(2) == 0 {
							for i := range vals {
								vals[i] &= 0xf1
							}
						}
						t = pgTableOf(cnt, vals)
					case kindRef:
						t = pgMakeTable(rng, pgRefineSyms(rng, clean), clean)
					default:
						t = pgMakeTable(rng, pgACFirstSyms(rng), clean)
					}
					g.tables[1][s[1]] = t
					p = append(p, t.payload(1, s[1])...)
				}
			}
		}
		if len(p) > 0 {
			w.segment(dhtMarker, p)
		}
		if si > 0 && rng.Intn(10) == 0 {
			g.dqt()
		}
		if si > 0 && rng.Intn(20) == 0 {
			g.ri = rng.Intn(5)
			w.segment(driMarker, []byte{0, byte(g.ri)})
		}
		hdr := []byte{byte(len(sc.comps))}
		for k, ci := range sc.comps {
			hdr = append(hdr, byte(ci+1), byte(sel[k][0]<<4|sel[k][1]))
		}
		hdr = append(hdr, byte(sc.ss), byte(sc.se), byte(sc.ah<<4|sc.al))
		w.segment(sosMarker, hdr)
		valid := sc.ss <= sc.se && sc.se < 64 && !(sc.ss == 0 && sc.se != 0) &&
			!(sc.ss != 0 && len(sc.comps) != 1) && (sc.ah == 0 || sc.ah == sc.al+1)
		if !valid {
			// The decoder stops at the header; some data anyway.
			w.bits(rng.Uint32(), 32)
			w.flush()
			continue
		}
		g.scanData(sc, sel)
		if rng.Intn(30) == 0 && !clean {
			w.b = append(w.b, 0x00, 0x11) // Extraneous data between scans.
		}
	}
	w.b = append(w.b, 0xff, eoiMarker)
	b := w.b
	if !clean {
		switch rng.Intn(8) {
		case 0:
			b = mutate(rng, b)
		case 1:
			b = b[:rng.Intn(len(b)+1)]
		}
	}
	return b
}

// pgScript returns a random scan script for nComp components: the DC
// first scans (interleaved, one per component, or two groups), AC bands
// per component with successive approximation chains (some split into
// sub-bands, some stopping early), DC refinements, all interleaved at
// random; a component may be left out, and (unless clean) odd scans are
// inserted: refinements without a first scan, repeated first scans,
// overlapping bands and invalid parameters.
func pgScript(rng *rand.Rand, nComp int, clean, extreme bool) []pgScan {
	var scans []pgScan
	all := make([]int, nComp)
	for i := range all {
		all[i] = i
	}
	dcAl := []int{0, 0, 0, 1, 1, 2, 3, 5, 8, 13, 15}[rng.Intn(11)]
	if extreme && rng.Intn(2) == 0 {
		dcAl = 13 + rng.Intn(3)
	}
	switch rng.Intn(4) {
	case 0, 1:
		scans = append(scans, pgScan{comps: all, al: dcAl})
	case 2:
		for _, c := range rng.Perm(nComp) {
			scans = append(scans, pgScan{comps: []int{c}, al: dcAl})
		}
	default:
		perm := rng.Perm(nComp)
		k := 1 + rng.Intn(nComp)
		scans = append(scans, pgScan{comps: perm[:k], al: dcAl})
		if k < nComp {
			scans = append(scans, pgScan{comps: perm[k:], al: dcAl})
		}
	}
	// AC bands per component, with successive approximation.
	var queues [][]pgScan
	omit := -1
	if nComp > 1 && rng.Intn(15) == 0 {
		omit = rng.Intn(nComp)
	}
	for c := 0; c < nComp; c++ {
		if rng.Intn(8) == 0 || c == omit {
			continue
		}
		bounds := []int{1, 64}
		for nb := []int{1, 1, 2, 3, 5}[rng.Intn(5)]; nb > 1; nb-- {
			bounds = append(bounds, 2+rng.Intn(62))
		}
		for i := 1; i < len(bounds); i++ {
			for j := i; j > 0 && bounds[j] < bounds[j-1]; j-- {
				bounds[j], bounds[j-1] = bounds[j-1], bounds[j]
			}
		}
		for b := 0; b+1 < len(bounds); b++ {
			ss, se := bounds[b], bounds[b+1]-1
			if ss > se {
				continue
			}
			al := []int{0, 0, 1, 1, 2, 3, 5, 9, 13, 14, 15}[rng.Intn(11)]
			if extreme && rng.Intn(2) == 0 {
				al = 12 + rng.Intn(4)
			}
			q := []pgScan{{comps: []int{c}, ss: ss, se: se, al: al}}
			stop := 0
			if rng.Intn(4) == 0 {
				stop = rng.Intn(al + 1)
			}
			for a := al; a > stop; a-- {
				if a > 14 {
					continue // Ah would be 16.
				}
				if se > ss && rng.Intn(4) == 0 {
					m := ss + rng.Intn(se-ss)
					q = append(q, pgScan{comps: []int{c}, ss: ss, se: m, ah: a, al: a - 1})
					q = append(q, pgScan{comps: []int{c}, ss: m + 1, se: se, ah: a, al: a - 1})
				} else {
					q = append(q, pgScan{comps: []int{c}, ss: ss, se: se, ah: a, al: a - 1})
				}
			}
			queues = append(queues, q)
		}
	}
	// DC refinements.
	var dcq []pgScan
	for a := dcAl; a > 0; a-- {
		if a > 14 {
			continue
		}
		if rng.Intn(2) == 0 {
			dcq = append(dcq, pgScan{comps: rng.Perm(nComp), ah: a, al: a - 1})
		} else {
			for _, c := range rng.Perm(nComp) {
				dcq = append(dcq, pgScan{comps: []int{c}, ah: a, al: a - 1})
			}
		}
	}
	queues = append(queues, dcq)
	for {
		var live []int
		for i, q := range queues {
			if len(q) > 0 {
				live = append(live, i)
			}
		}
		if len(live) == 0 {
			break
		}
		i := live[rng.Intn(len(live))]
		scans = append(scans, queues[i][0])
		queues[i] = queues[i][1:]
	}
	if omit >= 0 {
		var kept []pgScan
		for _, sc := range scans {
			var cs []int
			for _, c := range sc.comps {
				if c != omit {
					cs = append(cs, c)
				}
			}
			if len(cs) > 0 {
				sc.comps = cs
				kept = append(kept, sc)
			}
		}
		scans = kept
	}
	nOdd := 0
	if rng.Intn(2) == 0 && !clean {
		nOdd = 1 + rng.Intn(4)
	}
	for k := 0; k < nOdd; k++ {
		c := rng.Intn(nComp)
		ss := 1 + rng.Intn(63)
		se := ss + rng.Intn(64-ss)
		al := rng.Intn(15)
		var sc pgScan
		switch rng.Intn(9) {
		case 0, 1, 7, 8: // An AC refinement, maybe without a first scan.
			sc = pgScan{comps: []int{c}, ss: ss, se: se, ah: al + 1, al: al}
		case 2, 3: // An AC first scan, maybe a repeated one.
			sc = pgScan{comps: []int{c}, ss: ss, se: se, al: al}
		case 4: // A DC refinement.
			sc = pgScan{comps: rng.Perm(nComp), ah: al + 1, al: al}
		case 5: // A DC first scan.
			sc = pgScan{comps: rng.Perm(nComp)[:1+rng.Intn(nComp)], al: rng.Intn(16)}
		case 6: // Invalid parameters.
			switch rng.Intn(6) {
			case 0:
				sc = pgScan{comps: []int{c}, ss: 0, se: se, al: al}
			case 1:
				sc = pgScan{comps: []int{c}, ss: se, se: ss - 1, al: al}
			case 2:
				sc = pgScan{comps: []int{c}, ss: ss, se: 64 + rng.Intn(3), al: al}
			case 3:
				sc = pgScan{comps: rng.Perm(nComp), ss: ss, se: se, al: al}
			case 4:
				sc = pgScan{comps: []int{c}, ss: ss, se: se, ah: al + 2, al: al}
			case 5:
				sc = pgScan{comps: []int{c}, ss: ss, se: se, ah: 15, al: 14}
			}
		}
		pos := rng.Intn(len(scans) + 1)
		if pos == 0 && rng.Intn(4) != 0 {
			pos = 1
		}
		scans = append(scans[:pos], append([]pgScan{sc}, scans[pos:]...)...)
	}
	if len(scans) > 60 {
		scans = scans[:60]
	}
	return scans
}

// pgCheck decodes data with the fast path forced on (and with it off for
// odd i), through bytes.Reader, and for some i through readers that
// return a byte, a few bytes or half the bytes at a time.
func pgCheck(t *testing.T, i int, data []byte) {
	t.Helper()
	old := decMinFastBlocks
	defer func() { decMinFastBlocks = old }()
	decMinFastBlocks = 0
	if i%2 == 1 {
		decMinFastBlocks = 64
	}
	name := fmt.Sprintf("gen%d/min%d", i, decMinFastBlocks)
	compareDecode(t, name, data, readers[0].mk)
	if i%5 == 0 {
		compareDecode(t, name+"/onebyte", data, readers[1].mk)
	}
	if i%7 == 0 {
		compareDecode(t, name+"/chunk", data, func(b []byte) io.Reader { return &decChunkReader{bytes.NewReader(b), 1 + i%13} })
		compareDecode(t, name+"/half", data, readers[2].mk)
	}
}

// TestDecProgGen compares generated progressive streams (and, for some,
// streams with large coefficients and point transforms) with image/jpeg.
func TestDecProgGen(t *testing.T) {
	n := 2000
	if testing.Short() {
		n = 300
	}
	rng := rand.New(rand.NewSource(1))
	ok := 0
	for i := 0; i < n; i++ {
		data := pgRandomJPEG(rng, []int{16, 40, 100, 200}[rng.Intn(4)], false, i%5 == 4)
		pgCheck(t, i, data)
		if t.Failed() {
			t.Logf("gen%d: % x", i, data)
			return
		}
		if _, err := std.Decode(bytes.NewReader(data)); err == nil {
			ok++
		}
	}
	// The deliberate invalidities stop some streams early; most must
	// decode, or the generator does not reach the deep paths.
	t.Logf("%d of %d generated streams decode", ok, n)
	if ok < n/5 {
		t.Errorf("only %d of %d generated streams decode", ok, n)
	}
}

// TestDecProgGenModel checks the generator's model of image/jpeg's
// decoder: without deliberate invalidities, image/jpeg decodes every
// stream whose frame it accepts (the random sampling factors exceed the
// limit of 10 blocks per MCU now and then, on purpose).
func TestDecProgGenModel(t *testing.T) {
	n := 500
	if testing.Short() {
		n = 100
	}
	rng := rand.New(rand.NewSource(2))
	decoded := 0
	for i := 0; i < n; i++ {
		data := pgRandomJPEG(rng, []int{16, 40, 100}[rng.Intn(3)], true, i%4 == 3)
		_, err := std.Decode(bytes.NewReader(data))
		switch {
		case err == nil:
			decoded++
		case err.Error() != "invalid JPEG format: total sampling factors too large":
			t.Errorf("gen%d: a clean stream fails to decode: %v", i, err)
		}
	}
	if decoded < n*9/10 {
		t.Errorf("only %d of %d clean streams decode", decoded, n)
	}
}

// pgCompareCoeffs decodes data without its EOI marker with the fast path
// forced on and off (Decode then stops before reconstructing the image,
// with io.ErrUnexpectedEOF) and compares the coefficients left in
// d.progCoeffs, and the non-zero masks of the fast path where they are
// valid. That is a stronger check than the pixels, which the IDCT's
// clamping and rounding can make equal. It returns false if the stream
// stopped for another reason.
func pgCompareCoeffs(t *testing.T, name string, data []byte, mk func([]byte) io.Reader) bool {
	t.Helper()
	if n := len(data); n >= 2 && data[n-2] == 0xff && data[n-1] == eoiMarker {
		data = data[:n-2]
	}
	old := decMinFastBlocks
	defer func() { decMinFastBlocks = old }()
	run := func(min int) (*decoder, error) {
		decMinFastBlocks = min
		d := new(decoder)
		_, err := d.decode(mk(data), false)
		return d, err
	}
	dF, errF := run(0)
	dE, errE := run(1 << 62)
	if !sameError(errF, errE) {
		t.Errorf("%s: error %v, want %v", name, errF, errE)
		return false
	}
	if errE != io.ErrUnexpectedEOF {
		return false
	}
	for c := range dE.progCoeffs {
		pf, pe := dF.progCoeffs[c], dE.progCoeffs[c]
		if len(pf) != len(pe) {
			t.Errorf("%s: component %d: %d blocks, want %d", name, c, len(pf), len(pe))
			return false
		}
		for k := range pf {
			if pf[k] != pe[k] {
				t.Errorf("%s: component %d block %d:\n got %v\nwant %v", name, c, k, pf[k], pe[k])
				return false
			}
			if dF.decNZValid[c] && dF.decNZ[c] != nil {
				if m := decNonZero(&pf[k]); m != dF.decNZ[c][k] {
					t.Errorf("%s: component %d block %d: non-zero mask %#x, want %#x", name, c, k, dF.decNZ[c][k], m)
					return false
				}
			}
		}
	}
	return true
}

// TestDecProgGenCoeffs compares the coefficients of generated streams
// (pgCompareCoeffs), through bytes.Reader and a one-byte reader.
func TestDecProgGenCoeffs(t *testing.T) {
	n := 2000
	if testing.Short() {
		n = 300
	}
	rng := rand.New(rand.NewSource(3))
	compared := 0
	for i := 0; i < n; i++ {
		data := pgRandomJPEG(rng, []int{16, 40, 100, 200}[rng.Intn(4)], false, i%5 == 4)
		mk := readers[0].mk
		if i%3 == 1 {
			mk = readers[1].mk
		}
		if pgCompareCoeffs(t, fmt.Sprintf("gen%d", i), data, mk) {
			compared++
		}
		if t.Failed() {
			t.Logf("gen%d: % x", i, data)
			return
		}
	}
	t.Logf("%d of %d streams compared", compared, n)
	if compared < n/5 {
		t.Errorf("only %d of %d streams compared", compared, n)
	}
}

// pgByteSource is a rand.Source that returns the fuzzer's bytes and then a
// sequence seeded by them, so that mutations of the input change the
// generator's choices one by one.
type pgByteSource struct {
	b []byte
	x uint64
}

func (s *pgByteSource) Uint64() uint64 {
	if len(s.b) >= 8 {
		v := binary.LittleEndian.Uint64(s.b)
		s.b = s.b[8:]
		s.x ^= v
		return v
	}
	if len(s.b) > 0 {
		var t [8]byte
		copy(t[:], s.b)
		s.b = nil
		s.x ^= binary.LittleEndian.Uint64(t[:])
	}
	// splitmix64.
	s.x += 0x9e3779b97f4a7c15
	z := s.x
	z = (z ^ z>>30) * 0xbf58476d1ce4e5b9
	z = (z ^ z>>27) * 0x94d049bb133111eb
	return z ^ z>>31
}

func (s *pgByteSource) Int63() int64 { return int64(s.Uint64() >> 1) }
func (s *pgByteSource) Seed(int64)   {}

// FuzzDecProgGen generates a progressive stream from the fuzzer's bytes
// and compares it with image/jpeg (pixels and errors, the fast path forced
// on and off) and coefficient by coefficient (pgCompareCoeffs).
func FuzzDecProgGen(f *testing.F) {
	for i := 0; i < 32; i++ {
		b := make([]byte, 8+i%24)
		rand.New(rand.NewSource(int64(i))).Read(b)
		f.Add(b)
	}
	f.Fuzz(func(t *testing.T, seed []byte) {
		rng := rand.New(&pgByteSource{b: seed})
		data := pgRandomJPEG(rng, []int{16, 40, 100, 160}[rng.Intn(4)], false, len(seed)%5 == 4)
		pgCheck(t, len(seed), data)
		pgCheck(t, len(seed)+1, data)
		pgCompareCoeffs(t, "coefficients", data, readers[0].mk)
	})
}

// pgNZInvalidStream builds a progressive gray stream for the scan-level
// path where the fast path's non-zero masks become invalid: an AC first
// scan whose EOB run outlasts it, an AC refinement scan with an undefined
// table that the run covers entirely (decoded by the exact path, which
// does not keep the masks), and then refinement scans that the fast path
// decodes without masks (nz == nil); withNew adds new coefficients in
// them.
func pgNZInvalidStream(withNew bool, w, h int) []byte {
	b := &decJPEGBuilder{}
	b.b = append(b.b, 0xff, soiMarker)
	q := []byte{0}
	for i := 0; i < 64; i++ {
		q = append(q, byte(1+i%7))
	}
	b.segment(dqtMarker, q)
	b.segment(sof2Marker, []byte{8, byte(h >> 8), byte(h), byte(w >> 8), byte(w), 1, 1, 0x11, 0})
	nb := ((w + 7) / 8) * ((h + 7) / 8)
	run := 2 * nb
	r := 0
	for 1<<(r+1) <= run {
		r++
	}
	dc := pgTableOf([16]byte{1}, []byte{0})
	ac0 := pgTableOf([16]byte{0, 3}, []byte{0x01, byte(r << 4), 0x00})
	ac1 := pgTableOf([16]byte{0, 3}, []byte{0x01, 0x00, 0x11})
	var p []byte
	p = append(p, dc.payload(0, 0)...)
	p = append(p, ac0.payload(1, 0)...)
	p = append(p, ac1.payload(1, 1)...)
	b.segment(dhtMarker, p)

	// DC first.
	b.segment(sosMarker, []byte{1, 1, 0x00, 0, 0, 0x00})
	for i := 0; i < nb; i++ {
		dc.emit(b, 0)
	}
	b.flush()
	// AC first, Al 1: block 0 gets 1<<1 at zig 1, then an EOB run of 2*nb
	// blocks, longer than the rest of the scan.
	b.segment(sosMarker, []byte{1, 1, 0x00, 1, 63, 0x01})
	ac0.emit(b, 0x01)
	b.bits(1, 1)
	ac0.emit(b, byte(r<<4))
	b.bits(uint32(run-1<<r), uint(r))
	b.flush()
	// AC refinement with an undefined table (3), all of it within the
	// run: only block 0's correction bit.
	b.segment(sosMarker, []byte{1, 1, 0x03, 1, 63, 0x10})
	b.bits(1, 1)
	b.flush()
	// AC refinement with a defined table: every block ends with an EOB.
	b.segment(sosMarker, []byte{1, 1, 0x01, 1, 63, 0x10})
	for i := 0; i < nb; i++ {
		if withNew && i%5 == 3 {
			ac1.emit(b, 0x11) // A new coefficient after one zero.
			b.bits(uint32(i&1), 1)
		}
		ac1.emit(b, 0x00)
		if i == 0 {
			b.bits(1, 1)
		}
	}
	b.flush()
	// One more refinement: the corrections of the coefficients now
	// non-zero.
	b.segment(sosMarker, []byte{1, 1, 0x01, 1, 63, 0x10})
	for i := 0; i < nb; i++ {
		ac1.emit(b, 0x00)
		if i == 0 {
			b.bits(1, 1)
		}
		if withNew && i%5 == 3 {
			b.bits(1, 1)
		}
	}
	b.flush()
	return append(b.b, 0xff, eoiMarker)
}

// TestDecProgNZInvalid decodes pgNZInvalidStream streams through every
// reader, with the fast path forced and not, checks that the masks are
// invalid after the third scan, and compares the coefficients with those
// of the exact path alone.
func TestDecProgNZInvalid(t *testing.T) {
	for _, withNew := range []bool{false, true} {
		for _, sz := range [][2]int{{128, 128}, {64, 72}, {200, 40}} {
			name := fmt.Sprintf("new=%v/%dx%d", withNew, sz[0], sz[1])
			data := pgNZInvalidStream(withNew, sz[0], sz[1])
			if r := decodeStd(bytes.NewReader(data)); r.err != nil || r.panic != nil {
				t.Fatalf("%s: image/jpeg rejects the stream: %v", name, r)
			}
			for _, r := range readers {
				compareDecode(t, name+"/"+r.name, data, r.mk)
			}
			old := decMinFastBlocks
			decMinFastBlocks = 0
			compareDecode(t, name+"/force", data, readers[0].mk)
			d := new(decoder)
			d.decode(bytes.NewReader(data[:len(data)-2]), false)
			decMinFastBlocks = old
			if d.decNZValid[0] {
				t.Errorf("%s: the non-zero masks are still valid after the refinement with an undefined table", name)
			}
			pgCompareCoeffs(t, name, data, readers[0].mk)
		}
	}
}
