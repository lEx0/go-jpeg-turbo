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

import (
	"bytes"
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"sync"
	"testing"

	"github.com/lEx0/go-jpeg-turbo/internal/cpuinfo"
)

// forLevels runs f as a subtest per level of cpuinfo.Levels (named after
// the level), with SetLevel(level); the previous level is restored after
// the subtest and its parallel subtests. With -tags purego, which has no
// kernels per level, only the Go level runs.
func forLevels(t *testing.T, f func(t *testing.T)) {
	t.Helper()
	levels := cpuinfo.Levels()
	if purego {
		levels = levels[:1]
	}
	for _, l := range levels {
		t.Run(l.String(), func(t *testing.T) {
			prev := SetLevel(l)
			t.Cleanup(func() { SetLevel(prev) })
			f(t)
		})
	}
}

// atLevel returns impls, without the portable Go kernels (which do not
// depend on the level) except at level Go.
func atLevel[T any](impls []T) []T {
	if Level() == cpuinfo.Go {
		return impls
	}
	return impls[:1]
}

// Differential tests of the kernels (the exported functions, which run NEON
// on arm64, the SSE4.1, AVX2 or AVX-512 kernels on amd64 depending on the
// level, and the portable Go of fast.go with -tags purego or elsewhere, and
// the portable Go functions directly on every build) against their *Ref
// specifications. Tests of the exported functions run at every level the
// machine has (forLevels).

// --- input generators ------------------------------------------------------

// Coefficient block kinds.
const (
	kindReal     = iota // decoded-like: DC and a few low-frequency ACs
	kindFull            // every coefficient a random int32
	kindSparse          // a few random coefficients of random size
	kindDC              // DC only, random int32
	kindExtreme         // values from a set of int32 extremes
	kindRow0            // only row 0 (or only column 0) non-zero
	kindWrapZero        // non-zero coefficients whose products wrap to 0
	kindTop             // random int32 in rows 0-3 only (the sparse NEON path)
	kindBottom          // random int32 in rows 4-7 (and maybe the DC) only
	numKinds
)

var kindNames = [numKinds]string{"real", "full", "sparse", "dc", "extreme", "row0", "wrapzero", "top", "bottom"}

var extremes = []int32{
	0, 1, -1, 2, -2, 127, -128, 255, 1023, -1024, 2047, -2048,
	1 << 13, -1 << 13, 1<<14 - 1, 1 << 14, -1 << 14, 1<<15 - 1, -1 << 15,
	1 << 16, -1 << 16, 1<<17 - 1, 1 << 17, 1 << 18, 1 << 19, 1 << 20,
	1 << 24, 1 << 28, 1 << 30, -1 << 30, math.MaxInt32, math.MinInt32,
	math.MaxInt32 - 1, math.MinInt32 + 1, 0x55555555, -0x55555555, 65535, -65535,
}

// randInt32 returns a random int32 of random magnitude (uniform bit length).
func randInt32(r *rand.Rand) int32 {
	v := int32(r.Uint32())
	return v >> r.IntN(32)
}

func genBlock(r *rand.Rand, kind int, b *Block) {
	*b = Block{}
	switch kind {
	case kindReal:
		b[0] = int32(r.IntN(4096) - 2048)
		n := r.IntN(20)
		for i := 0; i < n; i++ {
			zig := 1 + int(float64(r.IntN(63))*r.Float64())
			mag := int32(1 + r.IntN(1+256>>(zig/8)))
			if r.IntN(2) == 0 {
				mag = -mag
			}
			b[Unzig[zig]] = mag
		}
	case kindFull:
		for i := range b {
			b[i] = int32(r.Uint32())
		}
	case kindSparse:
		n := 1 + r.IntN(6)
		for i := 0; i < n; i++ {
			b[r.IntN(BlockSize)] = randInt32(r)
		}
	case kindDC:
		b[0] = randInt32(r)
	case kindExtreme:
		for i := range b {
			if r.IntN(3) != 0 {
				b[i] = extremes[r.IntN(len(extremes))]
			}
		}
	case kindRow0:
		col := r.IntN(2) == 0
		for i := 0; i < 8; i++ {
			v := randInt32(r)
			if r.IntN(3) == 0 {
				v = 0
			}
			if col {
				b[8*i] = v
			} else {
				b[i] = v
			}
		}
	case kindWrapZero:
		// Filled together with the table by genWrapZero.
	case kindTop, kindBottom:
		lo := 0
		if kind == kindBottom {
			lo = 32
			b[0] = randInt32(r)
		}
		for i := lo; i < lo+32; i++ {
			if r.IntN(4) != 0 {
				b[i] = int32(r.Uint32())
				if r.IntN(2) == 0 {
					b[i] = randInt32(r)
				}
			}
		}
		if r.IntN(8) == 0 {
			b[lo+r.IntN(32)] = extremes[r.IntN(len(extremes))]
		}
	}
}

// Quantization table kinds (natural order, int32 like the decoder's).
const (
	qtReal  = iota // a scaled standard table, entries in [1, 255]
	qt8            // random [1, 255]
	qt16           // random [1, 65535]
	qtOne          // all 1
	qtMax16        // all 65535
	qtFull         // random int32
	qtExtreme
	numQT
)

func genQT(r *rand.Rand, kind int, qt *Block) {
	switch kind {
	case qtReal:
		q := qualityTable(1+r.IntN(100), r.IntN(2))
		for zig, v := range q {
			qt[Unzig[zig]] = int32(v)
		}
	case qt8:
		for i := range qt {
			qt[i] = int32(1 + r.IntN(255))
		}
	case qt16:
		for i := range qt {
			qt[i] = int32(1 + r.IntN(65535))
		}
	case qtOne:
		for i := range qt {
			qt[i] = 1
		}
	case qtMax16:
		for i := range qt {
			qt[i] = 65535
		}
	case qtFull:
		for i := range qt {
			qt[i] = int32(r.Uint32())
		}
	case qtExtreme:
		for i := range qt {
			qt[i] = extremes[r.IntN(len(extremes))]
		}
	}
}

// genWrapZero fills b and qt so that some AC products wrap to exactly 0.
func genWrapZero(r *rand.Rand, b, qt *Block) {
	*b = Block{}
	for i := range qt {
		qt[i] = int32(1 + r.IntN(255))
	}
	b[0] = int32(r.IntN(4096) - 2048)
	n := 1 + r.IntN(8)
	for i := 0; i < n; i++ {
		k := 1 + r.IntN(BlockSize-1)
		s := 1 + r.IntN(31)
		b[k] = int32(r.Uint32()) << s
		qt[k] = int32(r.Uint32()|1) << (32 - s)
	}
	if r.IntN(4) == 0 {
		b[Unzig[1+r.IntN(5)]] = int32(r.IntN(9) - 4)
	}
}

// qualityTable is image/jpeg's quantization table for a quality (Encode,
// writer.go), in zig-zag order; i selects luminance (0) or chrominance.
func qualityTable(quality, i int) [BlockSize]byte {
	var scale int
	if quality < 50 {
		scale = 5000 / quality
	} else {
		scale = 200 - quality*2
	}
	var q [BlockSize]byte
	for j := range q {
		x := int(unscaledQuant[i][j])
		x = (x*scale + 50) / 100
		if x < 1 {
			x = 1
		} else if x > 255 {
			x = 255
		}
		q[j] = uint8(x)
	}
	return q
}

// unscaledQuant is image/jpeg's (writer.go), in zig-zag order.
var unscaledQuant = [2][BlockSize]byte{
	{
		16, 11, 12, 14, 12, 10, 16, 14,
		13, 14, 18, 17, 16, 19, 24, 40,
		26, 24, 22, 22, 24, 49, 35, 37,
		29, 40, 58, 51, 61, 60, 57, 51,
		56, 55, 64, 72, 92, 78, 64, 68,
		87, 69, 55, 56, 80, 109, 81, 87,
		95, 98, 103, 104, 103, 62, 77, 113,
		121, 112, 100, 120, 92, 101, 103, 99,
	},
	{
		17, 18, 18, 24, 21, 24, 47, 26,
		26, 47, 99, 66, 56, 66, 99, 99,
		99, 99, 99, 99, 99, 99, 99, 99,
		99, 99, 99, 99, 99, 99, 99, 99,
		99, 99, 99, 99, 99, 99, 99, 99,
		99, 99, 99, 99, 99, 99, 99, 99,
		99, 99, 99, 99, 99, 99, 99, 99,
		99, 99, 99, 99, 99, 99, 99, 99,
	},
}

// iterations scales a test's case count (-short divides it by 10).
func iterations(n int) int {
	if testing.Short() {
		return n / 10
	}
	return n
}

// parallel runs f(seed, n) on every CPU, splitting n cases.
func parallel(t *testing.T, n int, f func(r *rand.Rand, n int)) {
	t.Helper()
	w := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	for i := 0; i < w; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			f(rand.New(rand.NewPCG(uint64(i), 0x5eed)), n/w+1)
		}(i)
	}
	wg.Wait()
}

// --- inverse DCT -----------------------------------------------------------

// storeFn is the signature of DequantIDCTStore.
type storeFn func(b, qt *Block, dst []byte, stride int)

var storeImpls = []struct {
	name string
	f    storeFn
}{
	{"exported", DequantIDCTStore},
	{"go", dequantIDCTStoreGo},
}

var idctImpls = []struct {
	name string
	f    func(b, qt *Block)
}{
	{"exported", DequantIDCT},
	{"go", dequantIDCTGo},
}

// checkStore runs f and DequantIDCTStoreRef on copies of the same
// destination (random bytes around the 8x8 area, slice capacity ending right
// after it) and reports any difference anywhere in the buffer.
func checkStore(t testing.TB, r *rand.Rand, name string, f storeFn, b, qt *Block, stride, off int) bool {
	n := off + 7*stride + 8
	const guard = 32
	want := make([]byte, n+guard)
	pattern := byte(n)
	if r != nil {
		pattern = byte(r.Uint32())
	}
	for i := range want {
		want[i] = byte(i*131) ^ pattern
	}
	got := bytes.Clone(want)
	b1, b2 := *b, *b
	DequantIDCTStoreRef(&b1, qt, want[off:n:n], stride)
	f(&b2, qt, got[off:n:n], stride)
	if !bytes.Equal(got, want) {
		for i := range got {
			if got[i] != want[i] {
				y, x := -1, -1
				if stride > 0 {
					y, x = (i-off)/stride, (i-off)%stride
				}
				t.Errorf("%s: stride %d off %d: byte %d (y %d x %d) = %d, want %d\nb = %v\nqt = %v",
					name, stride, off, i, y, x, got[i], want[i], *b, *qt)
				return false
			}
		}
	}
	return true
}

// randStride returns a destination stride: mostly small, sometimes large.
func randStride(r *rand.Rand) int {
	if r.IntN(8) == 0 {
		return []int{64, 100, 1023, 4096}[r.IntN(4)]
	}
	return 8 + r.IntN(33)
}

func checkIDCT(t testing.TB, name string, f func(b, qt *Block), b, qt *Block) bool {
	want, got := *b, *b
	DequantIDCTRef(&want, qt)
	f(&got, qt)
	if got != want {
		t.Errorf("%s: got\n%v\nwant\n%v\nb = %v\nqt = %v", name, got, want, *b, *qt)
		return false
	}
	return true
}

func TestDequantIDCTStore(t *testing.T) {
	forLevels(t, testDequantIDCTStore)
}

func testDequantIDCTStore(t *testing.T) {
	storeImpls, idctImpls := atLevel(storeImpls), atLevel(idctImpls)
	for kind := 0; kind < numKinds; kind++ {
		t.Run(kindNames[kind], func(t *testing.T) {
			t.Parallel()
			var mu sync.Mutex
			fails := 0
			parallel(t, iterations(300000), func(r *rand.Rand, n int) {
				var b, qt Block
				for i := 0; i < n; i++ {
					if kind == kindWrapZero {
						genWrapZero(r, &b, &qt)
					} else {
						genBlock(r, kind, &b)
						genQT(r, r.IntN(numQT), &qt)
					}
					stride := randStride(r)
					off := r.IntN(16)
					for _, impl := range storeImpls {
						if !checkStore(t, r, impl.name, impl.f, &b, &qt, stride, off) {
							mu.Lock()
							fails++
							mu.Unlock()
						}
					}
					for _, impl := range idctImpls {
						if !checkIDCT(t, impl.name, impl.f, &b, &qt) {
							mu.Lock()
							fails++
							mu.Unlock()
						}
					}
					if fails > 5 {
						return
					}
				}
			})
		})
	}
}

// TestDequantIDCTStoreArgs covers the argument checks of DequantIDCTStore:
// strides below 8 (overlapping rows, handled by the Ref) and destinations of
// exactly 7*stride+8 bytes.
func TestDequantIDCTStoreArgs(t *testing.T) {
	forLevels(t, testDequantIDCTStoreArgs)
}

func testDequantIDCTStoreArgs(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	var b, qt Block
	for i := 0; i < 2000; i++ {
		genBlock(r, kindReal, &b)
		genQT(r, qtReal, &qt)
		for stride := 0; stride < 40; stride++ {
			checkStore(t, r, "exported", DequantIDCTStore, &b, &qt, stride, r.IntN(9))
		}
	}
	// Too short: both panic.
	for _, f := range []storeFn{DequantIDCTStoreRef, DequantIDCTStore} {
		func() {
			defer func() {
				if recover() == nil {
					t.Error("no panic for a short destination")
				}
			}()
			f(&b, &qt, make([]byte, 7*16+7), 16)
		}()
	}
}

// dcValue is the output of the inverse DCT of a block whose only non-zero
// dequantized coefficient is d = b[0]*qt[0] (all 64 values are the same):
// idctRows turns row 0 into eight copies of d<<17 and the other rows into
// zeros, and idctCols turns each column into eight copies of
// ((d<<17) + 1<<19) >> 20, every other term being a product, shift or box
// of zeros. The NEON kernel's DC path computes exactly this.
func dcValue(d int32) int32 {
	return (d<<17 + 1<<19) >> 20
}

func TestDCOnly(t *testing.T) {
	forLevels(t, testDCOnly)
}

func testDCOnly(t *testing.T) {
	storeImpls, idctImpls := atLevel(storeImpls), atLevel(idctImpls)
	r := rand.New(rand.NewPCG(3, 4))
	var ds []int32
	for _, e := range extremes {
		for k := int32(-3); k <= 3; k++ {
			ds = append(ds, e+k)
		}
	}
	// The wrap points of d<<17 (bit 14 of d) and of + 1<<19.
	for _, base := range []int32{1 << 14, 1 << 15, 3 << 14, 1<<14 - 4, 1<<15 - 4} {
		for k := int32(-8); k <= 8; k++ {
			ds = append(ds, base+k, -base+k)
		}
	}
	for i := 0; i < 20000; i++ {
		ds = append(ds, int32(r.Uint32()))
	}
	for _, d := range ds {
		var b, qt Block
		genQT(r, r.IntN(numQT), &qt)
		qt[0] = 1
		b[0] = d
		want := b
		DequantIDCTRef(&want, &qt)
		for i, v := range want {
			if v != dcValue(d) {
				t.Fatalf("d %d: Ref output %d = %d, dcValue = %d", d, i, v, dcValue(d))
			}
		}
		// And with the product split between b[0] and qt[0].
		b[0], qt[0] = d|1, int32(r.Uint32())
		for _, impl := range storeImpls {
			checkStore(t, r, impl.name, impl.f, &b, &qt, 8+r.IntN(30), r.IntN(16))
		}
		for _, impl := range idctImpls {
			checkIDCT(t, impl.name, impl.f, &b, &qt)
		}
		b[0], qt[0] = d, 1
		for _, impl := range storeImpls {
			checkStore(t, r, impl.name, impl.f, &b, &qt, 8+r.IntN(30), r.IntN(16))
		}
		for _, impl := range idctImpls {
			checkIDCT(t, impl.name, impl.f, &b, &qt)
		}
	}
}

// TestIDCTSingleCoefficient tries every position with a set of extreme
// values and tables.
func TestIDCTSingleCoefficient(t *testing.T) {
	forLevels(t, testIDCTSingleCoefficient)
}

func testIDCTSingleCoefficient(t *testing.T) {
	storeImpls, idctImpls := atLevel(storeImpls), atLevel(idctImpls)
	r := rand.New(rand.NewPCG(5, 6))
	for pos := 0; pos < BlockSize; pos++ {
		for _, v := range extremes {
			for _, q := range []int32{1, 2, 255, 65535, math.MaxInt32, -1} {
				var b, qt Block
				for i := range qt {
					qt[i] = q
				}
				b[pos] = v
				for _, impl := range storeImpls {
					checkStore(t, r, impl.name, impl.f, &b, &qt, 8, 0)
				}
				for _, impl := range idctImpls {
					checkIDCT(t, impl.name, impl.f, &b, &qt)
				}
			}
		}
	}
}

// --- forward DCT -------------------------------------------------------------

// Sample block kinds.
const (
	smpRandom = iota
	smpFlat
	smpExtreme // 0 or 255
	smpChecker
	smpImpulse
	smpSmooth
	smpNoise // small noise around a level
	numSmp
)

func genSamples(r *rand.Rand, kind int, s *[BlockSize]byte) {
	switch kind {
	case smpRandom:
		for i := range s {
			s[i] = byte(r.Uint32())
		}
	case smpFlat:
		v := [4]byte{0, 255, 128, byte(r.Uint32())}[r.IntN(4)]
		for i := range s {
			s[i] = v
		}
	case smpExtreme:
		for i := range s {
			s[i] = byte(-(r.Uint32() & 1))
		}
	case smpChecker:
		a, b := byte(0), byte(255)
		if r.IntN(2) == 0 {
			a, b = b, a
		}
		mode := r.IntN(4)
		for y := 0; y < 8; y++ {
			for x := 0; x < 8; x++ {
				var on bool
				switch mode {
				case 0:
					on = (x+y)&1 == 0
				case 1:
					on = x&1 == 0
				case 2:
					on = y&1 == 0
				default:
					on = (x/2+y/2)&1 == 0
				}
				if on {
					s[8*y+x] = a
				} else {
					s[8*y+x] = b
				}
			}
		}
	case smpImpulse:
		bg, fg := byte(0), byte(255)
		if r.IntN(2) == 0 {
			bg, fg = fg, bg
		}
		for i := range s {
			s[i] = bg
		}
		s[r.IntN(BlockSize)] = fg
	case smpSmooth:
		a, bx, by := r.Float64()*255, r.Float64()*60-30, r.Float64()*60-30
		for y := 0; y < 8; y++ {
			for x := 0; x < 8; x++ {
				v := a + bx*float64(x) + by*float64(y)
				s[8*y+x] = byte(max(0, min(255, int(v))))
			}
		}
	case smpNoise:
		base := r.IntN(256)
		for i := range s {
			s[i] = byte(max(0, min(255, base+r.IntN(9)-4)))
		}
	}
}

func newQuant(q *[BlockSize]byte) *Quant {
	t := new(Quant)
	t.Init(q)
	return t
}

// genTable returns a random encoder table: a quality table, random [1, 255],
// all 1, all 255 or all equal.
func genTable(r *rand.Rand) [BlockSize]byte {
	var q [BlockSize]byte
	switch r.IntN(5) {
	case 0, 1:
		q = qualityTable(1+r.IntN(100), r.IntN(2))
	case 2:
		for i := range q {
			q[i] = byte(1 + r.IntN(255))
		}
	case 3:
		v := byte([2]int{1, 255}[r.IntN(2)])
		for i := range q {
			q[i] = v
		}
	default:
		v := byte(1 + r.IntN(255))
		for i := range q {
			q[i] = v
		}
	}
	return q
}

type fdctFn func(src []byte, stride int, t *Quant, out *[BlockSize]int16) uint64

var fdctImpls = []struct {
	name string
	f    fdctFn
}{
	{"exported", FDCTQuant},
	{"go", fdctQuantGo},
}

func checkFDCT(t testing.TB, name string, f fdctFn, s *[BlockSize]byte, tab *Quant, stride, off int) bool {
	n := off + 7*stride + 8
	src := make([]byte, n+16)
	for i := range src {
		src[i] = byte(i*37 + 11)
	}
	for y := 0; y < 8; y++ {
		copy(src[off+y*stride:off+y*stride+8], s[8*y:8*y+8])
	}
	var want, got [BlockSize]int16
	for i := range got {
		got[i] = 0x5555
	}
	wantNZ := FDCTQuantRef(src[off:n:n], stride, tab, &want)
	gotNZ := f(src[off:n:n], stride, tab, &got)
	if got != want || gotNZ != wantNZ {
		t.Errorf("%s: stride %d off %d: got %v nz %016x\nwant %v nz %016x\nsamples %v\ntable %v",
			name, stride, off, got, gotNZ, want, wantNZ, *s, tab.Q)
		return false
	}
	return true
}

func TestFDCTQuant(t *testing.T) {
	forLevels(t, testFDCTQuant)
}

func testFDCTQuant(t *testing.T) {
	fdctImpls := atLevel(fdctImpls)
	for kind := 0; kind < numSmp; kind++ {
		t.Run(fmt.Sprint(kind), func(t *testing.T) {
			t.Parallel()
			var mu sync.Mutex
			fails := 0
			parallel(t, iterations(200000), func(r *rand.Rand, n int) {
				var s [BlockSize]byte
				for i := 0; i < n; i++ {
					genSamples(r, kind, &s)
					q := genTable(r)
					tab := newQuant(&q)
					stride := randStride(r)
					off := r.IntN(16)
					for _, impl := range fdctImpls {
						if !checkFDCT(t, impl.name, impl.f, &s, tab, stride, off) {
							mu.Lock()
							fails++
							mu.Unlock()
						}
					}
					if fails > 5 {
						return
					}
				}
			})
		})
	}
}

// TestFDCTQuantQualities runs every image/jpeg quality table on a set of
// sample blocks.
func TestFDCTQuantQualities(t *testing.T) {
	forLevels(t, testFDCTQuantQualities)
}

func testFDCTQuantQualities(t *testing.T) {
	fdctImpls := atLevel(fdctImpls)
	r := rand.New(rand.NewPCG(7, 8))
	var blocks [][BlockSize]byte
	for kind := 0; kind < numSmp; kind++ {
		for i := 0; i < 40; i++ {
			var s [BlockSize]byte
			genSamples(r, kind, &s)
			blocks = append(blocks, s)
		}
	}
	for quality := 1; quality <= 100; quality++ {
		for i := 0; i < 2; i++ {
			q := qualityTable(quality, i)
			tab := newQuant(&q)
			for _, s := range blocks {
				for _, impl := range fdctImpls {
					checkFDCT(t, impl.name, impl.f, &s, tab, 8, 0)
				}
			}
		}
	}
}

// TestFDCTQuantArgs covers strides below 8 (overlapping source rows),
// unaligned sources, short sources and unusable tables.
func TestFDCTQuantArgs(t *testing.T) {
	forLevels(t, testFDCTQuantArgs)
}

func testFDCTQuantArgs(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 10))
	for i := 0; i < 3000; i++ {
		var s [BlockSize]byte
		genSamples(r, r.IntN(numSmp), &s)
		q := genTable(r)
		tab := newQuant(&q)
		for stride := 0; stride < 24; stride++ {
			// Build the source directly: rows may overlap.
			off := r.IntN(16)
			n := off + 7*stride + 8
			src := make([]byte, n)
			for j := range src {
				src[j] = byte(r.Uint32())
			}
			var want, got [BlockSize]int16
			wantNZ := FDCTQuantRef(src[off:], stride, tab, &want)
			gotNZ := FDCTQuant(src[off:], stride, tab, &got)
			if got != want || gotNZ != wantNZ {
				t.Fatalf("stride %d: got %v %x, want %v %x", stride, got, gotNZ, want, wantNZ)
			}
		}
	}
	// A short source panics in both.
	q := qualityTable(75, 0)
	tab := newQuant(&q)
	for _, f := range []fdctFn{FDCTQuantRef, FDCTQuant} {
		func() {
			defer func() {
				if recover() == nil {
					t.Error("no panic for a short source")
				}
			}()
			var out [BlockSize]int16
			f(make([]byte, 7*16+7), 16, tab, &out)
		}()
	}
	// A table with a zero entry (or never initialized) divides by zero in
	// both.
	q[5] = 0
	tab = newQuant(&q)
	for _, tb := range []*Quant{tab, new(Quant)} {
		for _, f := range []fdctFn{FDCTQuantRef, FDCTQuant} {
			func() {
				defer func() {
					if recover() == nil {
						t.Error("no panic for a zero table entry")
					}
				}()
				var out [BlockSize]int16
				src := make([]byte, 64)
				for j := range src {
					src[j] = byte(j * 7)
				}
				f(src, 8, tb, &out)
			}()
		}
	}
}

// TestQuantExhaustive checks the rounded division of the portable kernel
// against image/jpeg's div for every divisor 8q (q in [1, 255]) and every
// value a can take ([-8192, 8191]: a = (v + 1<<17) >> 18 for an int32 v),
// and the 16-bit magic numbers of the NEON (magic15) and x86 (magic16)
// kernels for every x below 2^14. TestQuantVecExhaustive runs the vector
// kernels themselves.
func TestQuantExhaustive(t *testing.T) {
	for q := 1; q <= 255; q++ {
		d := int32(8 * q)
		var row [BlockSize]byte
		for i := range row {
			row[i] = byte(q)
		}
		tab := newQuant(&row)
		for a := int32(-8192); a <= 8191; a++ {
			if got, want := quantGo(a, tab.qm[0], tab.qc[0]), div(a, d); got != want {
				t.Fatalf("quantGo(%d, q=%d) = %d, want %d", a, q, got, want)
			}
		}
		// The whole range quantGo is documented for, sampled.
		for a := int64(-16387); a <= 1<<24; a += 1 + a&255 {
			if got, want := quantGo(int32(a), tab.qm[0], tab.qc[0]), div(int32(a), d); got != want {
				t.Fatalf("quantGo(%d, q=%d) = %d, want %d", a, q, got, want)
			}
		}
		mul, k := magic15(uint32(d))
		if mul >= 1<<15 || mul < 1<<14 {
			t.Fatalf("q %d: mul %d out of range", q, mul)
		}
		// mul is the largest such value: the next k gives mul >= 2^15.
		if next := (uint32(1)<<(16+k) + uint32(d) - 1) / uint32(d); next < 1<<15 {
			t.Fatalf("q %d: k %d is not the largest", q, k)
		}
		for x := int32(0); x < 1<<14; x++ {
			// SQDMULH: (2*x*mul) >> 16, then a right shift by k.
			got := ((2 * x * int32(mul)) >> 16) >> k
			if got != x/d {
				t.Fatalf("q %d: magic (%d, %d) gives %d / %d = %d", q, mul, k, x, d, got)
			}
		}
		mul16, s := magic16(uint32(d))
		if mul16 >= 1<<16 || mul16 < 1<<15 || s < 2 || s > 10 {
			t.Fatalf("q %d: magic16 (%d, %d) out of range", q, mul16, s)
		}
		// mul16 is the largest such value: the next s gives mul >= 2^16.
		if next := (uint64(1)<<(17+s) + uint64(d) - 1) / uint64(d); next < 1<<16 {
			t.Fatalf("q %d: s %d is not the largest", q, s)
		}
		scale := uint32(1) << (16 - s)
		for x := uint32(0); x < 1<<14; x++ {
			// PMULHUW by mul, then PMULHUW by scale (SSE4.1, AVX2) or
			// VPSRLVW by s (AVX-512).
			got := (((x * mul16) >> 16) * scale) >> 16
			if got2 := ((x * mul16) >> 16) >> s; got != x/uint32(d) || got2 != got {
				t.Fatalf("q %d: magic16 (%d, %d) gives %d / %d = %d, %d", q, mul16, s, x, d, got, got2)
			}
		}
	}
}

// TestQuantVecExhaustive runs the vector quantization of every level (if
// the build has one) on every value in [-8192, 8191] with the uniform table
// of every q in [1, 255].
func TestQuantVecExhaustive(t *testing.T) {
	forLevels(t, func(t *testing.T) {
		for q := 1; q <= 255; q++ {
			var row [BlockSize]byte
			for i := range row {
				row[i] = byte(q)
			}
			testQuantVec(t, newQuant(&row), q)
		}
	})
}

// testQuantVec runs the vector quantization alone (quantVec, if this build
// has one) on every value in [-8192, 8191] with the uniform table of q.
func testQuantVec(t *testing.T, tab *Quant, q int) {
	var c [BlockSize]int16
	var out [BlockSize]int16
	for base := -8192; base < 8192; base += BlockSize {
		for i := range c {
			c[i] = int16(base + i)
		}
		nz, ok := quantVec(&c, tab, &out)
		if !ok {
			return
		}
		var wantNZ uint64
		for zig := 0; zig < BlockSize; zig++ {
			n := Unzig[zig]
			a := int32(c[8*(n%8)+n/8])
			want := div(a, int32(8*q))
			if int32(out[zig]) != want {
				t.Fatalf("q %d: a %d: got %d, want %d", q, a, out[zig], want)
			}
			if want != 0 {
				wantNZ |= 1 << (63 - zig)
			}
		}
		if nz != wantNZ {
			t.Fatalf("q %d base %d: bitmap %016x, want %016x", q, base, nz, wantNZ)
		}
	}
}

// TestQuantVecLayout checks the table layout of the vector quantization
// with random per-coefficient tables and values.
func TestQuantVecLayout(t *testing.T) {
	forLevels(t, testQuantVecLayout)
}

func testQuantVecLayout(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	var c, out [BlockSize]int16
	q50 := qualityTable(50, 0)
	if _, ok := quantVec(&c, newQuant(&q50), &out); !ok {
		t.Skip("no vector quantization in this build")
	}
	for i := 0; i < iterations(200000); i++ {
		q := genTable(r)
		if i%3 == 0 {
			for j := range q {
				q[j] = byte(1 + r.IntN(255))
			}
		}
		tab := newQuant(&q)
		for j := range c {
			switch r.IntN(4) {
			case 0:
				c[j] = 0
			case 1:
				c[j] = int16(r.IntN(16384) - 8192)
			default:
				c[j] = int16(r.IntN(64) - 32)
			}
		}
		nz, _ := quantVec(&c, tab, &out)
		var wantNZ uint64
		for zig := 0; zig < BlockSize; zig++ {
			n := Unzig[zig]
			want := div(int32(c[8*(n%8)+n/8]), tab.Div[zig])
			if int32(out[zig]) != want {
				t.Fatalf("zig %d: got %d, want %d (table %v)", zig, out[zig], want, q)
			}
			if want != 0 {
				wantNZ |= 1 << (63 - zig)
			}
		}
		if nz != wantNZ {
			t.Fatalf("bitmap %016x, want %016x", nz, wantNZ)
		}
	}
}

// TestQuantInit checks the private fields Init fills.
func TestQuantInit(t *testing.T) {
	r := rand.New(rand.NewPCG(13, 14))
	for i := 0; i < 1000; i++ {
		q := genTable(r)
		tab := newQuant(&q)
		if !tab.ok || tab.Q != q {
			t.Fatal("Init did not accept a valid table")
		}
		for zig, v := range q {
			if tab.Div[zig] != 8*int32(v) {
				t.Fatalf("zig %d: Div %d for q %d", zig, tab.Div[zig], v)
			}
		}
	}
	var q [BlockSize]byte
	if newQuant(&q).ok {
		t.Fatal("Init accepted a zero table")
	}
}

// TestIDCTAliased passes the same block as coefficients and table.
func TestIDCTAliased(t *testing.T) {
	forLevels(t, testIDCTAliased)
}

func testIDCTAliased(t *testing.T) {
	storeImpls, idctImpls := atLevel(storeImpls), atLevel(idctImpls)
	r := rand.New(rand.NewPCG(17, 18))
	for i := 0; i < 20000; i++ {
		var b Block
		genBlock(r, i%numKinds, &b)
		if i%3 == 0 {
			b[0] |= 1 // not DC-only
		}
		for _, impl := range idctImpls {
			want, got := b, b
			DequantIDCTRef(&want, &want)
			impl.f(&got, &got)
			if got != want {
				t.Fatalf("%s: aliased b %v: got %v, want %v", impl.name, b, got, want)
			}
		}
		for _, impl := range storeImpls {
			var want, got [7*8 + 8]byte
			b1, b2 := b, b
			DequantIDCTStoreRef(&b1, &b1, want[:], 8)
			impl.f(&b2, &b2, got[:], 8)
			if got != want {
				t.Fatalf("%s: aliased b %v: got %v, want %v", impl.name, b, got, want)
			}
		}
	}
}
