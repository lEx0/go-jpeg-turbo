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

package ycc

import (
	"bytes"
	"fmt"
	"image/color"
	"math/rand/v2"
	"runtime"
	"slices"
	"sync"
	"testing"
)

// The exported kernels (NEON on arm64; on amd64 the portable Go code and
// the SSE4.1, AVX2 and AVX-512 kernels, each test running at every level
// the machine supports; Go with -tags purego and on other architectures)
// are checked against their *Ref specifications: the arithmetic identities
// they rely on exhaustively, every RGB color and (with -short: a subset of
// the alphas) every NRGBA color through the kernels (under the race
// detector a sample of them, see exhaustiveChunks), and random rows of
// every length 0..100 and some long ones at unaligned offsets with guard
// bytes around the outputs.

// ---- arithmetic identities ----

// TestPixelExhaustive proves that pixel (the portable kernels' per-pixel
// conversion: color.RGBToYCbCr with only the upper chroma clamp) and the
// claim that color.RGBToYCbCr's chroma is clamp(sum>>16, 0, 255) of the
// pre-shift sum hold for every RGB triple.
func TestPixelExhaustive(t *testing.T) {
	for c := 0; c < 1<<24; c++ {
		r, g, b := uint8(c>>16), uint8(c>>8), uint8(c)
		wy, wcb, wcr := color.RGBToYCbCr(r, g, b)
		y, cb, cr := pixel(int32(r), int32(g), int32(b))
		if y != wy || cb != wcb || cr != wcr {
			t.Fatalf("pixel(%d, %d, %d) = %d %d %d, want %d %d %d", r, g, b, y, cb, cr, wy, wcb, wcr)
		}
		r1, g1, b1 := int32(r), int32(g), int32(b)
		scb := -11056*r1 - 21712*g1 + 32768*b1 + 257<<15
		scr := 32768*r1 - 27440*g1 - 5328*b1 + 257<<15
		if scb < 1<<16 || scb > 1<<24 || scr < 1<<16 || scr > 1<<24 {
			t.Fatalf("(%d, %d, %d): chroma sums %d %d outside [1<<16, 1<<24]", r, g, b, scb, scr)
		}
		if uint8(max(0, min(scb>>16, 255))) != wcb || uint8(max(0, min(scr>>16, 255))) != wcr {
			t.Fatalf("(%d, %d, %d): clamp(sum>>16) differs from color.RGBToYCbCr", r, g, b)
		}
	}
}

// TestSplitExhaustive proves the 16-bit scheme of the NEON kernels (see
// ycc_arm64.s): every 16-bit coefficient split into a high and a low byte,
// all sums in 16 bits, for every RGB triple.
func TestSplitExhaustive(t *testing.T) {
	const max16 = 1<<16 - 1
	for c := 0; c < 1<<24; c++ {
		r, g, b := uint32(c>>16), uint32(c>>8&0xff), uint32(c&0xff)
		nr, ng, nb := 255-r, 255-g, 255-b
		wy, wcb, wcr := color.RGBToYCbCr(uint8(r), uint8(g), uint8(b))

		h, l := 76*r+150*g+29*b, 139*r+70*g+47*b
		x := h + l>>8
		if h > max16 || l > max16 || x+128 > max16 {
			t.Fatalf("(%d, %d, %d): luma overflows 16 bits", r, g, b)
		}
		y := (x + 128) >> 8

		h, l = 128*b+43*nr+84*ng, 48*nr+208*ng
		x = h + l>>8
		if h > max16 || l > max16 || x > max16 {
			t.Fatalf("(%d, %d, %d): Cb overflows 16 bits", r, g, b)
		}
		cb := min(x>>8+1, 255)

		h, l = 128*r+107*ng+20*nb, 48*ng+208*nb
		x = h + l>>8
		if h > max16 || l > max16 || x > max16 {
			t.Fatalf("(%d, %d, %d): Cr overflows 16 bits", r, g, b)
		}
		cr := min(x>>8+1, 255)

		if uint8(y) != wy || uint8(cb) != wcb || uint8(cr) != wcr {
			t.Fatalf("split(%d, %d, %d) = %d %d %d, want %d %d %d", r, g, b, y, cb, cr, wy, wcb, wcr)
		}
	}
}

// madd is PMADDWD on one 32-bit lane: the signed words of v times the
// int16 pair (k0, k1), summed.
func madd(v uint32, k0, k1 int16) int32 {
	return int32(int16(v))*int32(k0) + int32(int16(v>>16))*int32(k1)
}

// TestMaddExhaustive proves the 32-bit scheme of the x86 kernels (see
// ycc_amd64.s) for every RGB triple: P1 = (r, g) and P2 = (b, 256) as
// 16-bit words, PMADDWD by int16 coefficient pairs, luma through P1 read
// as r + 65536 g, the halved chroma sums, and the narrowing by PACKSSDW
// (signed saturation to int16) and PACKUSWB (unsigned saturation to a
// byte; VPMOVUSDB and VPACKUSDW saturate the same way); and that the
// AVX512 kernels' sums (P2 = (b, g), the biases added) are the same.
func TestMaddExhaustive(t *testing.T) {
	sat16 := func(v int32) int32 { return max(-32768, min(v, 32767)) }
	satU8 := func(v int32) uint8 { return uint8(max(0, min(v, 255))) }
	for c := 0; c < 1<<24; c++ {
		r, g, b := uint32(c>>16), uint32(c>>8&0xff), uint32(c&0xff)
		wy, wcb, wcr := color.RGBToYCbCr(uint8(r), uint8(g), uint8(b))
		p1, p2 := r|g<<16, b|256<<16

		sy := int32(p1) + madd(p1, 19594, -27066) + madd(p2, 7471, 128)
		scb := madd(p1, -5528, -10856) + madd(p2, 16384, 16448)
		scr := madd(p1, 16384, -13720) + madd(p2, -2664, 16448)
		if sy < 0 || sy >= 1<<24 || scb < 1<<15 || scb > 1<<23 || scr < 1<<15 || scr > 1<<23 {
			t.Fatalf("(%d, %d, %d): sums %d %d %d out of range", r, g, b, sy, scb, scr)
		}
		// The AVX512 kernels: P2 = (b, g), the biases in the
		// accumulators, the same sums.
		p2bg := b | g<<16
		zy := 32768 + madd(p1, 19595, 19235) + madd(p2bg, 7471, 19235)
		zcb := 4210688 + madd(p1, -5528, -10856) + madd(p2bg, 16384, 0)
		zcr := 4210688 + madd(p1, 16384, -13720) + madd(p2bg, -2664, 0)
		if zy != sy || zcb != scb || zcr != scr {
			t.Fatalf("(%d, %d, %d): AVX512 sums %d %d %d, want %d %d %d", r, g, b, zy, zcb, zcr, sy, scb, scr)
		}
		y := satU8(sat16(int32(uint32(sy) >> 16)))
		cb := satU8(sat16(int32(uint32(scb) >> 15)))
		cr := satU8(sat16(int32(uint32(scr) >> 15)))
		if y != wy || cb != wcb || cr != wcr {
			t.Fatalf("madd(%d, %d, %d) = %d %d %d, want %d %d %d", r, g, b, y, cb, cr, wy, wcb, wcr)
		}
	}
}

// TestRoundShift proves that PMULHRSW by 8192, as the x86 kernels use it,
// is (x + 2) >> 2 for every 2x2 sum x of 8-bit samples.
func TestRoundShift(t *testing.T) {
	pmulhrsw := func(a, b int16) int16 { return int16(((int32(a)*int32(b))>>14 + 1) >> 1) }
	for x := 0; x <= 4*255; x++ {
		if got := pmulhrsw(int16(x), 8192); int(got) != (x+2)>>2 {
			t.Fatalf("pmulhrsw(%d, 8192) = %d, want %d", x, got, (x+2)>>2)
		}
	}
}

// premulStd is what image/jpeg computes for one NRGBA channel.
func premulStd(c, a uint8) uint8 {
	r, _, _, _ := color.NRGBA{c, 0, 0, a}.RGBA()
	return uint8(r >> 8)
}

// TestPremulExhaustive proves the division-free forms of
// color.NRGBA.RGBA() >> 8 for every (channel, alpha) pair: the portable
// (c*a*33025)>>23, the NEON kernels' 16-bit (y + (y>>8) + 1) >> 8 with
// y = c*a + (c*a)>>8, and the x86 kernels' 16-bit
// ((c*a * 33025) >> 16) >> 7.
func TestPremulExhaustive(t *testing.T) {
	for a := 0; a < 256; a++ {
		m := premulMul(uint8(a))
		for c := 0; c < 256; c++ {
			want := premulStd(uint8(c), uint8(a))
			if got := premul8(uint8(c), m); got != int32(want) {
				t.Fatalf("premul8(%d, %d) = %d, want %d", c, a, got, want)
			}
			if uint64(c)*uint64(m) >= 1<<32 {
				t.Fatalf("premul8(%d, %d) overflows 32 bits", c, a)
			}
			x := uint32(c * a)
			y := x + x>>8
			z := y + y>>8
			if z+1 > 1<<16-1 {
				t.Fatalf("(%d, %d): 16-bit premultiply overflows", c, a)
			}
			if got := (z + 1) >> 8; got != uint32(want) {
				t.Fatalf("16-bit premultiply(%d, %d) = %d, want %d", c, a, got, want)
			}
			// x86: PMULLW (exact: x < 1<<16), PMULHUW by 33025, >> 7.
			if hi := x * 33025 >> 16; x >= 1<<16 || hi>>7 != uint32(want) {
				t.Fatalf("x86 premultiply(%d, %d) = %d, want %d", c, a, hi>>7, want)
			}
		}
	}
}

// ---- reference tables for the exhaustive kernel tests ----

var (
	tabOnce sync.Once
	yccTab  *[3][1 << 24]byte // color.RGBToYCbCr of every (r<<16 | g<<8 | b)
	pmTab   *[256][256]byte   // pmTab[a][c] = premulStd(c, a)
)

func tables() (*[3][1 << 24]byte, *[256][256]byte) {
	tabOnce.Do(func() {
		yccTab = new([3][1 << 24]byte)
		parallel(256, func(hi int) {
			for c := hi << 16; c < (hi+1)<<16; c++ {
				yccTab[0][c], yccTab[1][c], yccTab[2][c] = color.RGBToYCbCr(uint8(c>>16), uint8(c>>8), uint8(c))
			}
		})
		pmTab = new([256][256]byte)
		for a := 0; a < 256; a++ {
			for c := 0; c < 256; c++ {
				pmTab[a][c] = premulStd(uint8(c), uint8(a))
			}
		}
	})
	return yccTab, pmTab
}

// parallel runs f(0..n-1) on GOMAXPROCS goroutines.
func parallel(n int, f func(int)) {
	var wg sync.WaitGroup
	next := make(chan int)
	for range runtime.GOMAXPROCS(0) {
		wg.Go(func() {
			for i := range next {
				f(i)
			}
		})
	}
	for i := range n {
		next <- i
	}
	close(next)
	wg.Wait()
}

// firstErr collects the first error reported by concurrent workers.
type firstErr struct {
	mu  sync.Mutex
	err error
}

func (e *firstErr) set(format string, args ...any) {
	e.mu.Lock()
	if e.err == nil {
		e.err = fmt.Errorf(format, args...)
	}
	e.mu.Unlock()
}

func (e *firstErr) failed() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.err != nil
}

// TestTablesMatchRef checks the table-driven expectations used below
// against the NRGBA *Ref functions on random pixels.
func TestTablesMatchRef(t *testing.T) {
	tab, pm := tables()
	rng := rand.New(rand.NewPCG(1, 2))
	const n = 1 << 16
	src := make([]byte, 4*n)
	for i := range src {
		src[i] = byte(rng.Uint32())
	}
	y, cb, cr := make([]byte, n), make([]byte, n), make([]byte, n)
	NRGBAToYCbCrRef(src, y, cb, cr, n)
	for i := range n {
		p := src[4*i : 4*i+4]
		a := p[3]
		c := int(pm[a][p[0]])<<16 | int(pm[a][p[1]])<<8 | int(pm[a][p[2]])
		if y[i] != tab[0][c] || cb[i] != tab[1][c] || cr[i] != tab[2][c] {
			t.Fatalf("table expectation differs from NRGBAToYCbCrRef for %v", p)
		}
	}
}

// ---- exhaustive kernel tests ----

// exhaustiveChunk is the row length of the exhaustive tests: even (for
// the 4:2:0 kernels) and not a multiple of 16, 32 or 64 (the vector
// blocks), so that every call also ends with an overlapping block.
const exhaustiveChunk = 49998

// TestRGBAToYCbCrExhaustive converts every RGB color (with random alpha
// bytes, which must be ignored; under the race detector a sample of the
// colors, as in the other exhaustive kernel tests).
func TestRGBAToYCbCrExhaustive(t *testing.T) { forEachLevel(t, testRGBAToYCbCrExhaustive) }

func testRGBAToYCbCrExhaustive(t *testing.T) {
	tab, _ := tables()
	var fe firstErr
	chunks := exhaustiveChunks()
	parallel(len(chunks), func(j int) {
		k := chunks[j]
		start := k * exhaustiveChunk
		n := min(exhaustiveChunk, 1<<24-start)
		rng := rand.New(rand.NewPCG(uint64(k), 3))
		src := make([]byte, 4*n)
		for i := range n {
			c := start + i
			src[4*i], src[4*i+1], src[4*i+2], src[4*i+3] = byte(c>>16), byte(c>>8), byte(c), byte(rng.Uint32())
		}
		y, cb, cr := make([]byte, n), make([]byte, n), make([]byte, n)
		RGBAToYCbCr(src, y, cb, cr, n)
		for i := range n {
			c := start + i
			if y[i] != tab[0][c] || cb[i] != tab[1][c] || cr[i] != tab[2][c] {
				fe.set("RGBAToYCbCr(%06x) = %d %d %d, want %d %d %d", c, y[i], cb[i], cr[i], tab[0][c], tab[1][c], tab[2][c])
				return
			}
		}
	})
	if fe.err != nil {
		t.Fatalf("%s: %v", implName(), fe.err)
	}
}

// perm is a bijection of the 24-bit colors, used for the second row of
// the 4:2:0 tests so that both rows see every color.
func perm(c int) int { return (c * 0x9e3779) & (1<<24 - 1) }

// TestRGBAToYCbCr420Exhaustive converts rows holding every RGB color
// (row 0 in order, row 1 permuted) and checks the luma of both rows and
// the averaged chroma against tables built with color.RGBToYCbCr.
func TestRGBAToYCbCr420Exhaustive(t *testing.T) { forEachLevel(t, testRGBAToYCbCr420Exhaustive) }

func testRGBAToYCbCr420Exhaustive(t *testing.T) {
	tab, _ := tables()
	var fe firstErr
	const n = exhaustiveChunk // even
	chunks := exhaustiveChunks()
	parallel(len(chunks), func(j int) {
		start := chunks[j] * n
		m := min(n, 1<<24-start)
		s0, s1 := make([]byte, 4*m), make([]byte, 4*m)
		for i := range m {
			c, d := start+i, perm(start+i)
			s0[4*i], s0[4*i+1], s0[4*i+2], s0[4*i+3] = byte(c>>16), byte(c>>8), byte(c), byte(i)
			s1[4*i], s1[4*i+1], s1[4*i+2], s1[4*i+3] = byte(d>>16), byte(d>>8), byte(d), byte(i>>8)
		}
		y0, y1, cb, cr := make([]byte, m), make([]byte, m), make([]byte, m/2), make([]byte, m/2)
		RGBAToYCbCr420(s0, s1, y0, y1, cb, cr, m)
		check420(&fe, "RGBAToYCbCr420", tab, start, m, y0, y1, cb, cr, func(i int) (int, int) {
			return start + i, perm(start + i)
		})
	})
	if fe.err != nil {
		t.Fatalf("%s: %v", implName(), fe.err)
	}
}

// check420 compares 4:2:0 output with the tables; idx(i) returns the
// premultiplied colors of pixel i of both rows.
func check420(fe *firstErr, name string, tab *[3][1 << 24]byte, start, m int, y0, y1, cb, cr []byte, idx func(int) (int, int)) {
	for i := 0; i < m; i += 2 {
		a, c := idx(i)
		b, d := idx(i + 1)
		if y0[i] != tab[0][a] || y0[i+1] != tab[0][b] || y1[i] != tab[0][c] || y1[i+1] != tab[0][d] {
			fe.set("%s: luma of pixels %d, %d (row start %d) differs", name, i, i+1, start)
			return
		}
		wcb := avg4(tab[1][a], tab[1][b], tab[1][c], tab[1][d])
		wcr := avg4(tab[2][a], tab[2][b], tab[2][c], tab[2][d])
		if cb[i/2] != wcb || cr[i/2] != wcr {
			fe.set("%s: chroma %d (row start %d) = %d %d, want %d %d", name, i/2, start, cb[i/2], cr[i/2], wcb, wcr)
			return
		}
	}
}

// Under the race detector the exhaustive kernel tests convert a sample:
// the detector does not see inside the assembly kernels, the normal runs
// prove the kernels exact on every input, and the instrumented loops over
// all of them took about 40 minutes. The sample still runs every kernel at
// every level on long rows (ending with an overlapping block), with the
// concurrent workers, for the extreme alphas and some others.

// raceChunkStep is the stride of the chunks the exhaustive kernel tests
// convert under the race detector.
const raceChunkStep = 16

// exhaustiveChunks returns the chunks k (the colors from k*exhaustiveChunk
// up to 1<<24) the exhaustive kernel tests convert: all of them, or under
// the race detector every raceChunkStep-th and the last (short) one.
func exhaustiveChunks() []int {
	n := (1<<24 + exhaustiveChunk - 1) / exhaustiveChunk
	var ks []int
	for k := range n {
		if !raceEnabled || k%raceChunkStep == 0 || k == n-1 {
			ks = append(ks, k)
		}
	}
	return ks
}

// TestRegressRaceSample: the exhaustive kernel tests ran in full under the
// race detector, which made this package take 2356 s with -race against
// 190 s without it. Under race they convert a sample, still from both ends
// of the color range and with the short last chunk; without race every
// chunk and (without -short) every alpha.
func TestRegressRaceSample(t *testing.T) {
	chunks, alphas := exhaustiveChunks(), exhaustiveAlphas()
	all := (1<<24 + exhaustiveChunk - 1) / exhaustiveChunk
	if len(chunks) == 0 || chunks[0] != 0 || chunks[len(chunks)-1] != all-1 || !slices.IsSorted(chunks) {
		t.Fatalf("exhaustiveChunks() = %v, want sorted chunks of 0..%d including both ends", chunks, all-1)
	}
	if !slices.Contains(alphas, 0) || !slices.Contains(alphas, 255) {
		t.Errorf("exhaustiveAlphas() = %v, want 0 and 255 among them", alphas)
	}
	if !raceEnabled {
		if len(chunks) != all {
			t.Errorf("%d of %d chunks without the race detector, want all", len(chunks), all)
		}
		if !testing.Short() && len(alphas) != 256 {
			t.Errorf("%d alphas without -short, want 256", len(alphas))
		}
		return
	}
	// TestNRGBAToYCbCrExhaustive and its 4:2:0 counterpart convert the
	// most pixels; one pass over the colors per level each is enough.
	if px := len(chunks) * exhaustiveChunk * len(alphas); px > 1<<24 {
		t.Errorf("under the race detector the NRGBA exhaustive tests convert %d pixels per level (%d chunks, %d alphas), want at most 1<<24",
			px, len(chunks), len(alphas))
	}
}

// exhaustiveAlphas returns the alpha values the NRGBA exhaustive tests
// cover: all of them, or with -short and under the race detector the
// extremes and a few others.
func exhaustiveAlphas() []int {
	if testing.Short() || raceEnabled {
		return []int{0, 1, 2, 3, 64, 127, 128, 129, 200, 253, 254, 255}
	}
	a := make([]int, 256)
	for i := range a {
		a[i] = i
	}
	return a
}

// TestNRGBAToYCbCrExhaustive converts every (r, g, b, a) quadruple (with
// -short every RGB color for a dozen alpha values; under the race detector
// a sample of the colors for those).
func TestNRGBAToYCbCrExhaustive(t *testing.T) { forEachLevel(t, testNRGBAToYCbCrExhaustive) }

func testNRGBAToYCbCrExhaustive(t *testing.T) {
	tab, pm := tables()
	alphas, chunks := exhaustiveAlphas(), exhaustiveChunks()
	var fe firstErr
	parallel(len(alphas)*len(chunks), func(job int) {
		if fe.failed() {
			return
		}
		a := alphas[job/len(chunks)]
		start := chunks[job%len(chunks)] * exhaustiveChunk
		n := min(exhaustiveChunk, 1<<24-start)
		src := make([]byte, 4*n)
		for i := range n {
			c := start + i
			src[4*i], src[4*i+1], src[4*i+2], src[4*i+3] = byte(c>>16), byte(c>>8), byte(c), byte(a)
		}
		y, cb, cr := make([]byte, n), make([]byte, n), make([]byte, n)
		NRGBAToYCbCr(src, y, cb, cr, n)
		p := &pm[a]
		for i := range n {
			c := start + i
			e := int(p[c>>16])<<16 | int(p[c>>8&0xff])<<8 | int(p[c&0xff])
			if y[i] != tab[0][e] || cb[i] != tab[1][e] || cr[i] != tab[2][e] {
				fe.set("NRGBAToYCbCr(%06x, alpha %d) = %d %d %d, want %d %d %d", c, a, y[i], cb[i], cr[i], tab[0][e], tab[1][e], tab[2][e])
				return
			}
		}
	})
	if fe.err != nil {
		t.Fatalf("%s: %v", implName(), fe.err)
	}
}

// TestNRGBAToYCbCr420Exhaustive is the 4:2:0 counterpart: row 0 holds
// every color with alpha a, row 1 every (permuted) color with alpha 255-a.
func TestNRGBAToYCbCr420Exhaustive(t *testing.T) { forEachLevel(t, testNRGBAToYCbCr420Exhaustive) }

func testNRGBAToYCbCr420Exhaustive(t *testing.T) {
	tab, pm := tables()
	alphas, chunks := exhaustiveAlphas(), exhaustiveChunks()
	const n = exhaustiveChunk
	var fe firstErr
	parallel(len(alphas)*len(chunks), func(job int) {
		if fe.failed() {
			return
		}
		a0 := alphas[job/len(chunks)]
		a1 := 255 - a0
		start := chunks[job%len(chunks)] * n
		m := min(n, 1<<24-start)
		s0, s1 := make([]byte, 4*m), make([]byte, 4*m)
		for i := range m {
			c, d := start+i, perm(start+i)
			s0[4*i], s0[4*i+1], s0[4*i+2], s0[4*i+3] = byte(c>>16), byte(c>>8), byte(c), byte(a0)
			s1[4*i], s1[4*i+1], s1[4*i+2], s1[4*i+3] = byte(d>>16), byte(d>>8), byte(d), byte(a1)
		}
		y0, y1, cb, cr := make([]byte, m), make([]byte, m), make([]byte, m/2), make([]byte, m/2)
		NRGBAToYCbCr420(s0, s1, y0, y1, cb, cr, m)
		p0, p1 := &pm[a0], &pm[a1]
		pre := func(p *[256]byte, c int) int {
			return int(p[c>>16])<<16 | int(p[c>>8&0xff])<<8 | int(p[c&0xff])
		}
		check420(&fe, fmt.Sprintf("NRGBAToYCbCr420 alpha %d/%d", a0, a1), tab, start, m, y0, y1, cb, cr, func(i int) (int, int) {
			return pre(p0, start+i), pre(p1, perm(start+i))
		})
	})
	if fe.err != nil {
		t.Fatalf("%s: %v", implName(), fe.err)
	}
}

// ---- random rows ----

// fill fills b with test samples: mode 0 uniform random, 1 extremes,
// 2 a ramp with noise, 3 constant. For NRGBA rows (alpha) every fourth
// byte is drawn from typical alpha patterns instead.
func fill(r *rand.Rand, b []byte, mode int, alpha bool) {
	c := byte(r.IntN(256))
	for i := range b {
		switch mode {
		case 0:
			b[i] = byte(r.IntN(256))
		case 1:
			b[i] = [6]byte{0, 1, 127, 128, 254, 255}[r.IntN(6)]
		case 2:
			b[i] = byte(i*3 + r.IntN(5))
		default:
			b[i] = c
		}
		if alpha && i&3 == 3 {
			switch r.IntN(4) {
			case 0:
				b[i] = 255
			case 1:
				b[i] = 0
			case 2:
				b[i] = byte(r.IntN(256))
			}
		}
	}
}

const guardLen = 64

// guarded returns an n-byte slice starting off bytes into a buffer whose
// other bytes hold a pattern, and a func reporting whether the pattern
// survived.
func guarded(n, off int) ([]byte, func() bool) {
	buf := make([]byte, guardLen+off+n+guardLen)
	for i := range buf {
		buf[i] = 0xa5
	}
	lo, hi := guardLen+off, guardLen+off+n
	return buf[lo:hi:hi], func() bool {
		for _, v := range buf[:lo] {
			if v != 0xa5 {
				return false
			}
		}
		for _, v := range buf[hi:] {
			if v != 0xa5 {
				return false
			}
		}
		return true
	}
}

// input returns an n-byte input slice starting off bytes into a buffer,
// followed by junk the kernels must not use.
func input(r *rand.Rand, n, off, mode int, alpha bool) []byte {
	buf := make([]byte, off+n+64)
	for i := range buf {
		buf[i] = byte(r.Uint32())
	}
	s := buf[off : off+n]
	fill(r, s, mode, alpha)
	return s
}

func rowLengths() []int {
	ns := make([]int, 0, 120)
	for n := range 101 {
		ns = append(ns, n)
	}
	return append(ns, 127, 128, 129, 255, 256, 257, 1000, 1023, 1024, 1025, 1918, 1920, 4097)
}

var offsets = []int{0, 1, 3, 8, 13}

type rowKernel struct {
	name     string
	alpha    bool
	fn, ref  func(src, y, cb, cr []byte, n int)
	portable func(src, y, cb, cr []byte, n int)
}

var rowKernels = []rowKernel{
	{"RGBAToYCbCr", false, RGBAToYCbCr, RGBAToYCbCrRef, rgbaToYCbCrGo},
	{"NRGBAToYCbCr", true, NRGBAToYCbCr, NRGBAToYCbCrRef, nrgbaToYCbCrGo},
}

type rows420Kernel struct {
	name     string
	alpha    bool
	fn, ref  func(src0, src1, y0, y1, cb, cr []byte, n int)
	portable func(src0, src1, y0, y1, cb, cr []byte, n int)
}

var rows420Kernels = []rows420Kernel{
	{"RGBAToYCbCr420", false, RGBAToYCbCr420, RGBAToYCbCr420Ref, rgbaToYCbCr420Go},
	{"NRGBAToYCbCr420", true, NRGBAToYCbCr420, NRGBAToYCbCr420Ref, nrgbaToYCbCr420Go},
}

func TestRows(t *testing.T) { forEachLevel(t, testRows) }

func testRows(t *testing.T) {
	r := rand.New(rand.NewPCG(4, 5))
	for _, k := range rowKernels {
		for _, impl := range []struct {
			name string
			fn   func(src, y, cb, cr []byte, n int)
		}{{implName(), k.fn}, {"portable", k.portable}} {
			for _, n := range rowLengths() {
				for _, off := range offsets {
					for mode := range 4 {
						src := input(r, 4*n, off, mode, k.alpha)
						wy, wcb, wcr := make([]byte, n), make([]byte, n), make([]byte, n)
						k.ref(src, wy, wcb, wcr, n)
						y, okY := guarded(n, off)
						cb, okCb := guarded(n, (off+5)%16)
						cr, okCr := guarded(n, (off+11)%16)
						impl.fn(src, y, cb, cr, n)
						if !okY() || !okCb() || !okCr() {
							t.Fatalf("%s/%s n=%d off=%d: write outside the outputs", k.name, impl.name, n, off)
						}
						if !bytes.Equal(y, wy) || !bytes.Equal(cb, wcb) || !bytes.Equal(cr, wcr) {
							t.Fatalf("%s/%s n=%d off=%d mode=%d: output differs from %sRef\nsrc %v\ny  %v\nwy %v\ncb %v\nwcb %v\ncr %v\nwcr %v",
								k.name, impl.name, n, off, mode, k.name, src, y, wy, cb, wcb, cr, wcr)
						}
					}
				}
			}
		}
	}
}

func TestRows420(t *testing.T) { forEachLevel(t, testRows420) }

func testRows420(t *testing.T) {
	r := rand.New(rand.NewPCG(6, 7))
	for _, k := range rows420Kernels {
		for _, impl := range []struct {
			name string
			fn   func(src0, src1, y0, y1, cb, cr []byte, n int)
		}{{implName(), k.fn}, {"portable", k.portable}} {
			for _, n := range rowLengths() {
				for _, off := range offsets {
					for mode := range 4 {
						s0 := input(r, 4*n, off, mode, k.alpha)
						s1 := input(r, 4*n, (off+7)%16, (mode+1)%4, k.alpha)
						wy0, wy1, wcb, wcr := make([]byte, n), make([]byte, n), make([]byte, n/2), make([]byte, n/2)
						k.ref(s0, s1, wy0, wy1, wcb, wcr, n)
						y0, ok0 := guarded(n, off)
						y1, ok1 := guarded(n, (off+3)%16)
						cb, okCb := guarded(n/2, (off+5)%16)
						cr, okCr := guarded(n/2, (off+11)%16)
						impl.fn(s0, s1, y0, y1, cb, cr, n)
						if !ok0() || !ok1() || !okCb() || !okCr() {
							t.Fatalf("%s/%s n=%d off=%d: write outside the outputs", k.name, impl.name, n, off)
						}
						if !bytes.Equal(y0, wy0) || !bytes.Equal(y1, wy1) || !bytes.Equal(cb, wcb) || !bytes.Equal(cr, wcr) {
							t.Fatalf("%s/%s n=%d off=%d mode=%d: output differs from %sRef\ny0 %v\nw  %v\ny1 %v\nw  %v\ncb %v\nw  %v\ncr %v\nw  %v",
								k.name, impl.name, n, off, mode, k.name, y0, wy0, y1, wy1, cb, wcb, cr, wcr)
						}
					}
				}
			}
		}
	}
}

func TestDownsample2x2Rows(t *testing.T) { forEachLevel(t, testDownsample2x2Rows) }

func testDownsample2x2Rows(t *testing.T) {
	r := rand.New(rand.NewPCG(8, 9))
	for _, impl := range []struct {
		name string
		fn   func(r0, r1, out []byte, n int)
	}{{implName(), Downsample2x2}, {"portable", downsample2x2Go}} {
		for _, n := range rowLengths() {
			for _, off := range offsets {
				for mode := range 4 {
					r0 := input(r, 2*n, off, mode, false)
					r1 := input(r, 2*n, (off+9)%16, (mode+2)%4, false)
					want := make([]byte, n)
					Downsample2x2Ref(r0, r1, want, n)
					out, ok := guarded(n, (off+1)%16)
					impl.fn(r0, r1, out, n)
					if !ok() {
						t.Fatalf("Downsample2x2/%s n=%d off=%d: write outside the output", impl.name, n, off)
					}
					if !bytes.Equal(out, want) {
						t.Fatalf("Downsample2x2/%s n=%d off=%d mode=%d: got %v, want %v", impl.name, n, off, mode, out, want)
					}
				}
			}
		}
	}
}

// TestDownsample2x2Exhaustive averages every 2x2 square whose samples
// take 22 values (a grid of 17s plus the extremes) in every position, and
// squares with every sum 0..1020, through the vector path (n >= 16).
func TestDownsample2x2Exhaustive(t *testing.T) { forEachLevel(t, testDownsample2x2Exhaustive) }

func testDownsample2x2Exhaustive(t *testing.T) {
	vals := []byte{0, 1, 2, 3, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 252, 253, 254, 255}
	var r0, r1 []byte
	for _, a := range vals {
		for _, b := range vals {
			for _, c := range vals {
				for _, d := range vals {
					r0 = append(r0, a, b)
					r1 = append(r1, c, d)
				}
			}
		}
	}
	// Every sum s in 0..1020 at least once.
	for s := 0; s <= 1020; s++ {
		a := min(s, 255)
		b := min(s-a, 255)
		c := min(s-a-b, 255)
		r0 = append(r0, byte(a), byte(b))
		r1 = append(r1, byte(c), byte(s-a-b-c))
	}
	n := len(r0) / 2
	want, got := make([]byte, n), make([]byte, n)
	Downsample2x2Ref(r0, r1, want, n)
	Downsample2x2(r0, r1, got, n)
	if !bytes.Equal(got, want) {
		t.Fatalf("%s: Downsample2x2 differs from Downsample2x2Ref", implName())
	}
}

// TestNRGBAOpaqueBlocks checks the NRGBA kernels on rows whose pixels are
// all opaque (alpha 255, which the vector kernels convert without
// premultiplying, a block at a time) but one, at every position, with
// any other alpha.
func TestNRGBAOpaqueBlocks(t *testing.T) { forEachLevel(t, testNRGBAOpaqueBlocks) }

func testNRGBAOpaqueBlocks(t *testing.T) {
	r := rand.New(rand.NewPCG(14, 15))
	opaque := func(n int) []byte {
		b := make([]byte, 4*n)
		fill(r, b, 0, false)
		for i := 3; i < len(b); i += 4 {
			b[i] = 255
		}
		return b
	}
	for _, n := range []int{16, 17, 31, 32, 33, 63, 64, 65, 100, 129, 130, 257} {
		for p := -1; p < n; p++ {
			s0, s1 := opaque(n), opaque(n)
			a := byte(r.IntN(255))
			if p >= 0 && p&1 == 0 {
				s0[4*p+3] = a
			} else if p >= 0 {
				s1[4*p+3] = a
			}
			for _, src := range [][]byte{s0, s1} {
				wy, wcb, wcr := make([]byte, n), make([]byte, n), make([]byte, n)
				y, cb, cr := make([]byte, n), make([]byte, n), make([]byte, n)
				NRGBAToYCbCrRef(src, wy, wcb, wcr, n)
				NRGBAToYCbCr(src, y, cb, cr, n)
				if !bytes.Equal(y, wy) || !bytes.Equal(cb, wcb) || !bytes.Equal(cr, wcr) {
					t.Fatalf("NRGBAToYCbCr/%s n=%d: pixel %d alpha %d: output differs from the Ref", implName(), n, p, a)
				}
			}
			wy0, wy1, wcb, wcr := make([]byte, n), make([]byte, n), make([]byte, n/2), make([]byte, n/2)
			y0, y1, cb, cr := make([]byte, n), make([]byte, n), make([]byte, n/2), make([]byte, n/2)
			NRGBAToYCbCr420Ref(s0, s1, wy0, wy1, wcb, wcr, n)
			NRGBAToYCbCr420(s0, s1, y0, y1, cb, cr, n)
			if !bytes.Equal(y0, wy0) || !bytes.Equal(y1, wy1) || !bytes.Equal(cb, wcb) || !bytes.Equal(cr, wcr) {
				t.Fatalf("NRGBAToYCbCr420/%s n=%d: pixel %d alpha %d: output differs from the Ref", implName(), n, p, a)
			}
		}
	}
}

// TestNoAlloc checks that the exported kernels do not allocate.
func TestNoAlloc(t *testing.T) { forEachLevel(t, testNoAlloc) }

func testNoAlloc(t *testing.T) {
	const n = 100
	src0, src1 := make([]byte, 4*n), make([]byte, 4*n)
	y0, y1, cb, cr := make([]byte, n), make([]byte, n), make([]byte, n), make([]byte, n)
	allocs := testing.AllocsPerRun(10, func() {
		RGBAToYCbCr(src0, y0, cb, cr, n)
		NRGBAToYCbCr(src0, y0, cb, cr, n)
		Downsample2x2(src0, src1, y0, n)
		RGBAToYCbCr420(src0, src1, y0, y1, cb, cr, n)
		NRGBAToYCbCr420(src0, src1, y0, y1, cb, cr, n)
	})
	if allocs != 0 {
		t.Fatalf("kernels allocate: %v allocs per run", allocs)
	}
}

// TestShortSlices checks that a slice shorter than n makes the kernels
// panic before they write anything (and that n <= 0 does nothing).
func TestShortSlices(t *testing.T) { forEachLevel(t, testShortSlices) }

func testShortSlices(t *testing.T) {
	mustPanic := func(name string, f func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Errorf("%s: no panic", name)
			}
		}()
		f()
	}
	for _, n := range []int{5, 16, 17, 64} {
		long := make([]byte, 4*n)
		short := make([]byte, n-1)
		short4 := make([]byte, 4*n-1)
		untouched := func(name string, bufs ...[]byte) {
			for _, b := range bufs {
				for _, v := range b {
					if v != 0 {
						t.Errorf("%s n=%d: wrote before panicking", name, n)
						return
					}
				}
			}
		}
		for _, k := range rowKernels {
			y, cb, cr := make([]byte, n), make([]byte, n), make([]byte, n)
			mustPanic(k.name+" short src", func() { k.fn(short4, y, cb, cr, n) })
			untouched(k.name, y, cb, cr)
			mustPanic(k.name+" short y", func() { k.fn(long, short, cb, cr, n) })
			mustPanic(k.name+" short cb", func() { k.fn(long, y, short, cr, n) })
			mustPanic(k.name+" short cr", func() { k.fn(long, y, cb, short, n) })
			untouched(k.name, y, cb, cr)
		}
		for _, k := range rows420Kernels {
			y0, y1, cb, cr := make([]byte, n), make([]byte, n), make([]byte, n/2), make([]byte, n/2)
			mustPanic(k.name+" short src0", func() { k.fn(short4, long, y0, y1, cb, cr, n) })
			mustPanic(k.name+" short src1", func() { k.fn(long, short4, y0, y1, cb, cr, n) })
			mustPanic(k.name+" short y1", func() { k.fn(long, long, y0, short, cb, cr, n) })
			mustPanic(k.name+" short cr", func() { k.fn(long, long, y0, y1, cb, make([]byte, n/2-1), n) })
			untouched(k.name, y0, y1, cb, cr)
		}
		out := make([]byte, n)
		mustPanic("Downsample2x2 short r1", func() { Downsample2x2(long, short, out, n) })
		mustPanic("Downsample2x2 short out", func() { Downsample2x2(long, long, short, n) })
		untouched("Downsample2x2", out)
	}
	for _, n := range []int{0, -1, -100} {
		RGBAToYCbCr(nil, nil, nil, nil, n)
		NRGBAToYCbCr(nil, nil, nil, nil, n)
		Downsample2x2(nil, nil, nil, n)
		RGBAToYCbCr420(nil, nil, nil, nil, nil, nil, n)
		NRGBAToYCbCr420(nil, nil, nil, nil, nil, nil, n)
	}
}

// ---- fuzzing ----

// FuzzKernels runs every kernel on fuzzer-chosen rows against its *Ref.
func FuzzKernels(f *testing.F) {
	f.Add([]byte{0, 0, 0, 0, 255, 255, 255, 255}, uint8(0))
	f.Add(bytes.Repeat([]byte{255, 0, 0, 255, 0, 0, 255, 128}, 40), uint8(3))
	f.Fuzz(func(t *testing.T, data []byte, off uint8) {
		data = data[min(int(off&15), len(data)):]
		n := len(data) / 8 // two rows of n pixels
		s0, s1 := data[:4*n], data[4*n:8*n]
		for _, l := range testLevels() {
			func() {
				defer SetLevel(SetLevel(l))
				fuzzKernels(t, s0, s1, n)
			}()
		}
	})
}

// fuzzKernels checks every kernel at the current level on the rows s0
// and s1 of n pixels.
func fuzzKernels(t *testing.T, s0, s1 []byte, n int) {
	for _, k := range rowKernels {
		wy, wcb, wcr := make([]byte, n), make([]byte, n), make([]byte, n)
		y, cb, cr := make([]byte, n), make([]byte, n), make([]byte, n)
		k.ref(s0, wy, wcb, wcr, n)
		k.fn(s0, y, cb, cr, n)
		if !bytes.Equal(y, wy) || !bytes.Equal(cb, wcb) || !bytes.Equal(cr, wcr) {
			t.Fatalf("%s/%s differs from its Ref", k.name, implName())
		}
	}
	for _, k := range rows420Kernels {
		wy0, wy1, wcb, wcr := make([]byte, n), make([]byte, n), make([]byte, n/2), make([]byte, n/2)
		y0, y1, cb, cr := make([]byte, n), make([]byte, n), make([]byte, n/2), make([]byte, n/2)
		k.ref(s0, s1, wy0, wy1, wcb, wcr, n)
		k.fn(s0, s1, y0, y1, cb, cr, n)
		if !bytes.Equal(y0, wy0) || !bytes.Equal(y1, wy1) || !bytes.Equal(cb, wcb) || !bytes.Equal(cr, wcr) {
			t.Fatalf("%s/%s differs from its Ref", k.name, implName())
		}
	}
	want, got := make([]byte, 2*n), make([]byte, 2*n)
	Downsample2x2Ref(s0, s1, want, 2*n)
	Downsample2x2(s0, s1, got, 2*n)
	if !bytes.Equal(got, want) {
		t.Fatalf("%s: Downsample2x2 differs from its Ref", implName())
	}
}
