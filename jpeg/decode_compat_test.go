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

// Differential tests of the decoder against image/jpeg aimed at the fast
// entropy decoder (decode_huffman.go): byte stuffing, markers, truncation and
// reader errors at and around the 4096-byte buffer boundaries, odd io.Reader
// chunk sizes, and hand-built streams with malformed Huffman tables, EOB runs
// in baseline scans, excessive DC sizes, non-interleaved baseline scans and
// tables redefined between scans.

import (
	"bytes"
	"fmt"
	"image"
	std "image/jpeg"
	"io"
	"math/rand"
	"path/filepath"
	"testing"
)

// decBufBoundaries returns the stream offsets at which the decoder's
// 4096-byte buffer ends when the input arrives in full 4096-byte reads:
// the first fill reads 4096 bytes, every later one 4094 (two bytes are kept).
func decBufBoundaries(n int) []int {
	var b []int
	for p := 4096; p < n+4096; p += 4094 {
		b = append(b, p)
	}
	return b
}

// decSomeBoundaries returns the first few buffer boundaries within n bytes,
// and the last one.
func decSomeBoundaries(n int) []int {
	b := decBufBoundaries(n)
	if len(b) > 6 {
		b = append(b[:5], b[len(b)-1])
	}
	return b
}

// decPad inserts p bytes after the SOI marker without changing the decoded
// image: a COM segment, or for p < 4 extraneous zero bytes (which both
// decoders skip before the next marker).
func decPad(data []byte, p int) []byte {
	var pad []byte
	switch {
	case p >= 4:
		pad = make([]byte, p)
		pad[0], pad[1], pad[2], pad[3] = 0xff, comMarker, byte((p-2)>>8), byte(p-2)
		for i := 4; i < p; i++ {
			pad[i] = byte(i)
		}
	case p > 0:
		pad = make([]byte, p)
	}
	out := append([]byte(nil), data[:2]...)
	out = append(out, pad...)
	return append(out, data[2:]...)
}

// decPaddings returns paddings (see decPad) that move each of the given
// stream positions to every offset in [-span, span] around a buffer
// boundary.
func decPaddings(n int, positions []int, span int) []int {
	seen := map[int]bool{}
	var ps []int
	for _, pos := range positions {
		for o := -span; o <= span; o++ {
			for _, b := range decBufBoundaries(n + 2*4096) {
				p := b + o - pos
				if p < 0 {
					continue
				}
				if p > 0xfff0 {
					break
				}
				if !seen[p] {
					seen[p] = true
					ps = append(ps, p)
				}
				break
			}
		}
	}
	return ps
}

// decFFPositions returns the positions of the 0xFF bytes in data[from:],
// and of every marker, at most max of them.
func decFFPositions(data []byte, from, max int) []int {
	var ps []int
	for i := from; i < len(data) && len(ps) < max; i++ {
		if data[i] == 0xff {
			ps = append(ps, i)
		}
	}
	return ps
}

// decNoiseJPEG encodes noise at quality 100 with image/jpeg: entropy-coded
// data with many 0xFF bytes, spanning several buffers.
func decNoiseJPEG(tb testing.TB, w, h int, gray bool, seed int64) []byte {
	tb.Helper()
	rng := rand.New(rand.NewSource(seed))
	var m image.Image
	if gray {
		g := image.NewGray(image.Rect(0, 0, w, h))
		rng.Read(g.Pix)
		m = g
	} else {
		c := image.NewRGBA(image.Rect(0, 0, w, h))
		rng.Read(c.Pix)
		for i := 3; i < len(c.Pix); i += 4 {
			c.Pix[i] = 0xff
		}
		m = c
	}
	var buf bytes.Buffer
	if err := std.Encode(&buf, m, &std.Options{Quality: 100}); err != nil {
		tb.Fatal(err)
	}
	return buf.Bytes()
}

// decChunkReader returns at most n bytes per Read.
type decChunkReader struct {
	r io.Reader
	n int
}

func (r *decChunkReader) Read(p []byte) (int, error) {
	if len(p) > r.n {
		p = p[:r.n]
	}
	return r.r.Read(p)
}

// decBoundaryInputs returns the inputs of the buffer-boundary tests: noise
// at q100 (0xFF00 everywhere), and restart-interval files.
func decBoundaryInputs(tb testing.TB) []struct {
	name string
	data []byte
} {
	in := []struct {
		name string
		data []byte
	}{
		{"noise420", decNoiseJPEG(tb, 96, 64, false, 1)},
		{"noisegray", decNoiseJPEG(tb, 128, 96, true, 2)},
	}
	files := []string{"video150x103_221122_rst1.jpeg", "photo257x131_rst8.jpeg", "video150x103_221122_progrst.jpeg", "photo640x480_rst8.jpeg", "photo640x480_q100.jpeg", "photo640x480_q100prog.jpeg"}
	if testing.Short() {
		in = in[:1]
		files = files[:3]
	}
	for _, f := range files {
		in = append(in, struct {
			name string
			data []byte
		}{f, readFile(tb, filepath.Join("../testdata/corpus", f))})
	}
	return in
}

// TestDecCompatBufferBoundaries moves every 0xFF byte (stuffed bytes, RST
// markers, the EOI marker) of several files to every offset around the
// buffer boundaries, and decodes the results through several readers.
func TestDecCompatBufferBoundaries(t *testing.T) {
	maxFF, span := 80, 9
	if testing.Short() {
		maxFF, span = 12, 3
	}
	for _, in := range decBoundaryInputs(t) {
		es := entropyStart(in.data)
		ps := decPaddings(len(in.data), decFFPositions(in.data, es, maxFF), span)
		// Also the end of the data, and the last bytes of the scan.
		ps = append(ps, decPaddings(len(in.data), []int{len(in.data) - 2, len(in.data) - 3}, 2*span)...)
		for k, p := range ps {
			data := decPad(in.data, p)
			name := fmt.Sprintf("%s/pad%d", in.name, p)
			compareDecode(t, name, data, readers[0].mk)
			if k%31 == 0 && len(data) < 64<<10 {
				for _, r := range readers[1:] {
					compareDecode(t, name+"/"+r.name, data, r.mk)
				}
			}
		}
	}
}

// TestDecCompatChunkedReaders decodes through readers that return odd
// numbers of bytes per call, which put the buffer boundaries everywhere.
func TestDecCompatChunkedReaders(t *testing.T) {
	sizes := []int{2, 3, 5, 7, 13, 64, 255, 511, 1000, 4093, 4095}
	for _, in := range decBoundaryInputs(t) {
		for _, n := range sizes {
			if testing.Short() && n < 64 && len(in.data) > 50000 {
				continue
			}
			compareDecode(t, fmt.Sprintf("%s/chunk%d", in.name, n), in.data, func(b []byte) io.Reader {
				return &decChunkReader{bytes.NewReader(b), n}
			})
		}
	}
}

// TestDecCompatTruncatedBoundaries truncates inputs at every byte around the
// buffer boundaries (and, for small files, at every byte).
func TestDecCompatTruncatedBoundaries(t *testing.T) {
	span := 24
	if testing.Short() {
		span = 6
	}
	pads := []int{0, 1, 2, 5, 1000}
	if testing.Short() {
		span, pads = 3, pads[2:4]
	}
	for _, in := range decBoundaryInputs(t) {
		for _, p := range pads {
			data := decPad(in.data, p)
			seen := map[int]bool{}
			try := func(n int) {
				if n < 0 || n > len(data) || seen[n] {
					return
				}
				seen[n] = true
				compareDecode(t, fmt.Sprintf("%s/pad%d[:%d]", in.name, p, n), data[:n], readers[0].mk)
			}
			for _, b := range decSomeBoundaries(len(data)) {
				for o := -span; o <= span; o++ {
					try(b + o)
				}
			}
			for o := 0; o <= span; o++ {
				try(len(data) - o)
			}
		}
	}
}

// TestDecCompatTruncatedEverywhere truncates small files at every byte.
func TestDecCompatTruncatedEverywhere(t *testing.T) {
	files := []string{
		"video150x103_221122_rst1.jpeg", "video150x103_211211_rst3.jpeg", "photo257x131_gray.jpeg",
		"video150x103_221122_base.jpeg", "video150x103_121121_opt.jpeg", "photo33x31_420_prog.jpeg",
	}
	step := 1
	if testing.Short() {
		step = 7
	}
	for _, f := range files {
		data := readFile(t, filepath.Join("../testdata/corpus", f))
		for n := 0; n <= len(data); n += step {
			compareDecode(t, fmt.Sprintf("%s[:%d]", f, n), data[:n], readers[0].mk)
		}
	}
}

// TestDecCompatReaderErrorBoundaries makes the reader fail at every offset
// around the buffer boundaries.
func TestDecCompatReaderErrorBoundaries(t *testing.T) {
	span := 20
	if testing.Short() {
		span = 5
	}
	pads := []int{0, 3, 777}
	if testing.Short() {
		span, pads = 3, pads[1:2]
	}
	for _, in := range decBoundaryInputs(t) {
		for _, p := range pads {
			data := decPad(in.data, p)
			for _, b := range decSomeBoundaries(len(data)) {
				for o := -span; o <= span; o++ {
					n := b + o
					compareDecode(t, fmt.Sprintf("%s/pad%d/errafter%d", in.name, p, n), data, func(b []byte) io.Reader {
						return &errAfterReader{data: b, n: n}
					})
				}
			}
		}
	}
}

// decFileFFs checks that the inputs really put 0xFF bytes in the
// entropy-coded data, so that the tests above test byte stuffing.
func TestDecCompatInputsHaveStuffing(t *testing.T) {
	for _, in := range decBoundaryInputs(t)[:1] {
		es := entropyStart(in.data)
		if n := len(decFFPositions(in.data, es, 1<<20)); n < 40 || len(in.data) < 3*4096 {
			t.Errorf("%s: %d bytes, %d 0xFF bytes in the entropy-coded data", in.name, len(in.data), n)
		}
	}
}

// Hand-built streams.

// decJPEGBuilder writes JPEG segments and entropy-coded data.
type decJPEGBuilder struct {
	b    []byte
	acc  uint64
	nacc uint
}

func (w *decJPEGBuilder) segment(marker byte, payload []byte) {
	n := len(payload) + 2
	w.b = append(w.b, 0xff, marker, byte(n>>8), byte(n))
	w.b = append(w.b, payload...)
}

// bits writes the low n bits of v, MSB first, with byte stuffing.
func (w *decJPEGBuilder) bits(v uint32, n uint) {
	for n > 0 {
		k := min(n, 8)
		n -= k
		w.acc = w.acc<<k | uint64(v>>n)&(1<<k-1)
		w.nacc += k
		for w.nacc >= 8 {
			c := byte(w.acc >> (w.nacc - 8))
			w.nacc -= 8
			w.b = append(w.b, c)
			if c == 0xff {
				w.b = append(w.b, 0x00)
			}
		}
	}
}

// flush pads to a byte boundary with 1 bits.
func (w *decJPEGBuilder) flush() {
	if w.nacc > 0 {
		w.bits(0xff, 8-w.nacc)
	}
	w.acc, w.nacc = 0, 0
}

// decCodes returns the canonical code (and its length) of every value
// index of a DHT table, as processDHT assigns them.
func decCodes(counts [16]byte) (codes []uint32, lengths []uint) {
	code := uint32(0)
	for l := 0; l < 16; l++ {
		for i := 0; i < int(counts[l]); i++ {
			codes = append(codes, code)
			lengths = append(lengths, uint(l+1))
			code++
		}
		code <<= 1
	}
	return codes, lengths
}

type decGenTable struct {
	counts  [16]byte
	vals    []byte
	codes   []uint32
	lengths []uint
}

func (t *decGenTable) payload(tc, th int) []byte {
	p := []byte{byte(tc<<4 | th)}
	p = append(p, t.counts[:]...)
	return append(p, t.vals...)
}

// sym writes a random symbol of t, preferring the ones in want, and
// returns it.
func (t *decGenTable) sym(rng *rand.Rand, w *decJPEGBuilder, want func(byte) bool) byte {
	i := rng.Intn(len(t.vals))
	for k := 0; k < 8 && !want(t.vals[i]); k++ {
		i = rng.Intn(len(t.vals))
	}
	w.bits(t.codes[i], t.lengths[i])
	return t.vals[i]
}

// decStdTables have the code lengths of the Annex K tables, which most
// files use, with the AC symbols in a similar order (the common ones
// first).
var decStdTables = [2][2]struct {
	counts [16]byte
	vals   []byte
}{
	{
		{[16]byte{0, 1, 5, 1, 1, 1, 1, 1, 1}, []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}},
		{[16]byte{0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1}, []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}},
	},
	{
		{[16]byte{0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d}, decACSymbols()},
		{[16]byte{0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77}, decACSymbols()},
	},
}

// decACSymbols returns the 162 baseline AC symbols, roughly from the most
// to the least common.
func decACSymbols() []byte {
	var v []byte
	for key := 0; key <= 2*15+10; key++ {
		for r := 0; r < 16; r++ {
			for s := 1; s <= 10; s++ {
				if 2*r+s == key {
					v = append(v, byte(r<<4|s))
				}
			}
		}
		if key == 2 {
			v = append(v, 0x00) // EOB.
		}
	}
	return append(v, 0xf0) // ZRL.
}

func decGenTableFor(rng *rand.Rand, tc, th int) *decGenTable {
	t := &decGenTable{}
	if rng.Intn(3) != 0 {
		s := decStdTables[tc][th%2]
		t.counts, t.vals = s.counts, s.vals
	} else {
		t.counts, t.vals = decRandomTable(rng, tc)
	}
	t.codes, t.lengths = decCodes(t.counts)
	return t
}

// decRandomJPEG builds a random sequential JPEG: random sampling factors,
// quantization and Huffman tables (well-formed or not), restart intervals,
// interleaved or non-interleaved scans with tables redefined in between,
// and entropy-coded data that mostly follows the tables (EOB runs and
// excessive DC sizes included when the tables contain such values), with
// the odd corruption.
func decRandomJPEG(rng *rand.Rand, maxDim int) []byte {
	w := &decJPEGBuilder{}
	w.b = append(w.b, 0xff, soiMarker)
	extended := rng.Intn(3) == 0
	nTh := 2
	if extended {
		nTh = 4
	}
	dqt := func() {
		var p []byte
		for tq := 0; tq < 4; tq++ {
			if rng.Intn(2) == 0 && tq > 0 {
				continue
			}
			if rng.Intn(8) == 0 {
				p = append(p, byte(0x10|tq))
				for i := 0; i < 64; i++ {
					p = append(p, byte(rng.Intn(3)), byte(rng.Intn(256)))
				}
				continue
			}
			p = append(p, byte(tq))
			for i := 0; i < 64; i++ {
				p = append(p, byte(1+rng.Intn(1+rng.Intn(255))))
			}
		}
		w.segment(dqtMarker, p)
	}
	dqt()

	nComp := []int{1, 3, 3, 3, 4}[rng.Intn(5)]
	width, height := 1+rng.Intn(maxDim), 1+rng.Intn(maxDim)
	type comp struct{ h, v, tq int }
	comps := make([]comp, nComp)
	hvs := []int{1, 1, 1, 2, 2, 4}
	for i := range comps {
		comps[i] = comp{hvs[rng.Intn(len(hvs))], hvs[rng.Intn(len(hvs))], rng.Intn(2)}
	}
	if nComp == 3 && rng.Intn(2) == 0 {
		comps[0].h, comps[0].v = []int{1, 2, 2, 4}[rng.Intn(4)], []int{1, 1, 2, 2}[rng.Intn(4)]
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
		maxH, maxV = 1, 1
		comps[0].h, comps[0].v = 1, 1 // How the decoder treats a single component.
	}
	marker := byte(sof0Marker)
	if extended {
		marker = sof1Marker
	}
	w.segment(marker, sof)

	tables := [2][4]*decGenTable{}
	dht := func() {
		var p []byte
		for tc := 0; tc < 2; tc++ {
			for th := 0; th < nTh; th++ {
				if tables[tc][th] != nil && rng.Intn(2) == 0 {
					continue
				}
				tables[tc][th] = decGenTableFor(rng, tc, th)
				p = append(p, tables[tc][th].payload(tc, th)...)
			}
		}
		if len(p) > 0 {
			w.segment(dhtMarker, p)
		}
	}
	dht()
	ri := 0
	if rng.Intn(2) == 0 {
		ri = rng.Intn(6)
		w.segment(driMarker, []byte{0, byte(ri)})
	}

	// Scans: one interleaved scan, or one per component, or a mix.
	var scans [][]int
	switch rng.Intn(3) {
	case 0:
		all := make([]int, nComp)
		for i := range all {
			all[i] = i
		}
		scans = [][]int{all}
	case 1:
		for i := 0; i < nComp; i++ {
			scans = append(scans, []int{i})
		}
	default:
		scans = [][]int{{0}}
		if nComp > 1 {
			rest := []int{}
			for i := 1; i < nComp; i++ {
				rest = append(rest, i)
			}
			scans = append(scans, rest)
		}
	}
	mxx, myy := (width+8*maxH-1)/(8*maxH), (height+8*maxV-1)/(8*maxV)
	for si, sc := range scans {
		if si > 0 {
			if rng.Intn(2) == 0 {
				dht()
			}
			if rng.Intn(3) == 0 {
				dqt()
			}
		}
		sel := make([][2]int, len(sc))
		p := []byte{byte(len(sc))}
		for k, ci := range sc {
			sel[k] = [2]int{rng.Intn(nTh), rng.Intn(nTh)}
			p = append(p, byte(ci+1), byte(sel[k][0]<<4|sel[k][1]))
		}
		p = append(p, 0, 63, 0)
		w.segment(sosMarker, p)

		// The MCU count of this scan, as processSOS counts them.
		nMCU := mxx * myy
		rst := 0
		for m := 0; m < nMCU; m++ {
			for k, ci := range sc {
				nb := comps[ci].h * comps[ci].v
				for j := 0; j < nb; j++ {
					dcT, acT := tables[dcTable][sel[k][0]], tables[acTable][sel[k][1]]
					s := dcT.sym(rng, w, func(v byte) bool { return v <= 11 || rng.Intn(20) == 0 })
					w.bits(rng.Uint32(), uint(min(s, 16)))
					for zig := 1; zig < 64; zig++ {
						v := acT.sym(rng, w, func(v byte) bool {
							if v == 0 {
								return rng.Intn(4) == 0 || zig > 20
							}
							return v&15 != 0 || v == 0xf0 || rng.Intn(10) == 0
						})
						r, s := int(v>>4), uint(v&15)
						if s == 0 {
							if r == 15 {
								zig += 15
								continue
							}
							if r != 0 {
								w.bits(rng.Uint32(), uint(r))
							}
							break
						}
						zig += r
						if zig > 63 {
							break
						}
						w.bits(rng.Uint32(), s)
					}
				}
			}
			if ri > 0 && (m+1)%ri == 0 && m+1 < nMCU {
				w.flush()
				w.b = append(w.b, 0xff, byte(rst0Marker+rst%8))
				rst++
			}
		}
		w.flush()
	}
	w.b = append(w.b, 0xff, eoiMarker)

	// The odd corruption.
	b := w.b
	if rng.Intn(4) == 0 {
		b = mutate(rng, b)
	}
	return b
}

func TestDecCompatHandBuilt(t *testing.T) {
	n := 4000
	if testing.Short() {
		n = 500
	}
	rng := rand.New(rand.NewSource(5))
	ok := 0
	for i := 0; i < n; i++ {
		data := decRandomJPEG(rng, 80)
		compareDecode(t, fmt.Sprintf("handbuilt%d", i), data, readers[0].mk)
		if i%16 == 0 {
			compareDecode(t, fmt.Sprintf("handbuilt%d/onebyte", i), data, readers[1].mk)
		}
		if _, err := std.Decode(bytes.NewReader(data)); err == nil {
			ok++
		}
		if t.Failed() {
			t.Logf("input: % x", data)
			break
		}
	}
	t.Logf("%d of %d hand-built files decode without error", ok, n)
	if ok < n/10 {
		t.Errorf("only %d of %d hand-built files decode", ok, n)
	}
}

// TestDecCompatHandBuiltLarge builds bigger hand-built images (spanning
// several buffers) with the standard tables, so that most of the data goes
// through the fast path, and checks every block via the image.
func TestDecCompatHandBuiltLarge(t *testing.T) {
	n := 60
	if testing.Short() {
		n = 10
	}
	rng := rand.New(rand.NewSource(6))
	for i := 0; i < n; i++ {
		data := decRandomJPEG(rng, 400)
		compareDecode(t, fmt.Sprintf("large%d", i), data, readers[0].mk)
		if i%4 == 0 {
			compareDecode(t, fmt.Sprintf("large%d/half", i), data, readers[2].mk)
		}
	}
}

// decForceFast makes every scan use the fast path, however small, for the
// rest of the test.
func decForceFast(tb testing.TB) {
	old := decMinFastBlocks
	decMinFastBlocks = 0
	tb.Cleanup(func() { decMinFastBlocks = old })
}

// TestDecCompatCorpusForceFast runs the corpus (whole, truncated and
// mutated) with the fast path forced on even for the smallest images.
func TestDecCompatCorpusForceFast(t *testing.T) {
	decForceFast(t)
	rng := rand.New(rand.NewSource(7))
	for i, f := range corpusFiles(t) {
		data := readFile(t, f)
		name := filepath.Base(f)
		compareDecode(t, name, data, readers[0].mk)
		if len(data) < 16<<10 {
			compareDecode(t, name+"/onebyte", data, readers[1].mk)
		}
		if testing.Short() && i%4 != 0 {
			continue
		}
		es := entropyStart(data)
		for _, n := range []int{es + 1, es + (len(data)-es)/2, len(data) - 3} {
			if n > 0 && n < len(data) {
				compareDecode(t, fmt.Sprintf("%s[:%d]", name, n), data[:n], readers[0].mk)
			}
		}
		for k := 0; k < 4; k++ {
			compareDecode(t, fmt.Sprintf("%s/mut%d", name, k), mutate(rng, data), readers[0].mk)
		}
	}
}

// FuzzDecCompatFast is FuzzCompatDecode with the fast path forced on for
// every scan (most fuzz inputs are too small to use it otherwise).
func FuzzDecCompatFast(f *testing.F) {
	for i, file := range corpusFiles(f) {
		if i%4 == 1 {
			if b := readFile(f, file); len(b) < 64<<10 {
				f.Add(b)
			}
		}
	}
	decForceFast(f)
	f.Fuzz(func(t *testing.T, b []byte) {
		if cfg, err := std.DecodeConfig(bytes.NewReader(b)); err == nil && cfg.Width*cfg.Height > 4e6 {
			return
		}
		compareDecode(t, "fuzz", b, readers[0].mk)
	})
}
