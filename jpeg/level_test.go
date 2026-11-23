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

// The SIMD level of the kernels under test. The environment variable
// JPEG_TURBO_LEVEL (go, sse41, avx2 or avx512; empty or unset: the
// default, cpuinfo.Max) selects the level of both internal/dct and
// internal/ycc for every test, fuzz target and benchmark of this package:
//
//	JPEG_TURBO_LEVEL=sse41 go test ./jpeg
//	make levels    # go test ./jpeg at every level the machine supports
//
// On arm64 and with -tags purego the kernels are fixed by the build (NEON
// or portable Go) and only go is accepted on arm64; the levels are then
// only recorded.

import (
	"bytes"
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"testing"

	"github.com/lEx0/go-jpeg-turbo/internal/cpuinfo"
	"github.com/lEx0/go-jpeg-turbo/internal/dct"
	"github.com/lEx0/go-jpeg-turbo/internal/ycc"
)

// levelEnv is the environment variable that selects the kernel level.
const levelEnv = "JPEG_TURBO_LEVEL"

// testLevel is the level TestMain selected for both kernel packages.
var testLevel = cpuinfo.Max

// setTestLevel parses s (the value of levelEnv) and selects that level in
// dct and ycc; an empty s keeps the default.
func setTestLevel(s string) error {
	if s == "" {
		testLevel = dct.Level()
		if l := ycc.Level(); l != testLevel {
			return fmt.Errorf("default levels differ: dct %v, ycc %v", testLevel, l)
		}
		return nil
	}
	l, err := cpuinfo.ParseLevel(s)
	if err != nil {
		return fmt.Errorf("%s=%s: %v", levelEnv, s, err)
	}
	if l > cpuinfo.Max {
		return fmt.Errorf("%s=%s: this machine (%s) supports up to %v (%v)", levelEnv, s, runtime.GOARCH, cpuinfo.Max, cpuinfo.Levels())
	}
	dct.SetLevel(l)
	ycc.SetLevel(l)
	if dct.Level() != l || ycc.Level() != l {
		return fmt.Errorf("%s=%s: levels after SetLevel: dct %v, ycc %v", levelEnv, s, dct.Level(), ycc.Level())
	}
	testLevel = l
	return nil
}

func TestMain(m *testing.M) {
	flag.Parse()
	// Under the race detector, -short unless given (race_short_test.go).
	if err := raceShort(); err != nil {
		fmt.Fprintln(os.Stderr, "jpeg tests:", err)
		os.Exit(2)
	}
	if err := setTestLevel(os.Getenv(levelEnv)); err != nil {
		fmt.Fprintln(os.Stderr, "jpeg tests:", err)
		os.Exit(2)
	}
	if testing.Verbose() {
		fmt.Printf("jpeg tests: %s=%q: dct level %v, ycc level %v (%s, cpuinfo.Max %v)\n",
			levelEnv, os.Getenv(levelEnv), dct.Level(), ycc.Level(), runtime.GOARCH, cpuinfo.Max)
	}
	code := m.Run()
	// A test that changed a level without restoring it would have run the
	// tests after it at the wrong level.
	if dct.Level() != testLevel || ycc.Level() != testLevel {
		fmt.Fprintf(os.Stderr, "jpeg tests: levels changed during the run: dct %v, ycc %v, want %v\n",
			dct.Level(), ycc.Level(), testLevel)
		if code == 0 {
			code = 1
		}
	}
	os.Exit(code)
}

// TestLevelEnv checks the parsing of JPEG_TURBO_LEVEL and that the kernels
// run at the level TestMain selected.
func TestLevelEnv(t *testing.T) {
	if dct.Level() != testLevel || ycc.Level() != testLevel {
		t.Fatalf("dct level %v, ycc level %v, want %v", dct.Level(), ycc.Level(), testLevel)
	}
	t.Logf("%s=%q: level %v (cpuinfo.Max %v)", levelEnv, os.Getenv(levelEnv), testLevel, cpuinfo.Max)

	defer func(d, y cpuinfo.Level) {
		dct.SetLevel(d)
		ycc.SetLevel(y)
		testLevel = d
	}(dct.Level(), ycc.Level())
	for _, s := range []string{"bogus", "AVX2", "sse4.1"} {
		if err := setTestLevel(s); err == nil {
			t.Errorf("setTestLevel(%q) succeeded, want an error", s)
		}
	}
	for l := cpuinfo.Go; l <= cpuinfo.AVX512; l++ {
		err := setTestLevel(l.String())
		if l > cpuinfo.Max {
			if err == nil {
				t.Errorf("setTestLevel(%q) above cpuinfo.Max %v succeeded", l, cpuinfo.Max)
			}
			continue
		}
		if err != nil || testLevel != l {
			t.Errorf("setTestLevel(%q): level %v, %v; want %v, nil", l, testLevel, err, l)
		}
	}
}

// ---- the paths inside the kernels ----

// The other fuzz targets of this package compare the codec with
// image/jpeg, and the fuzzer steers by the coverage of Go code, which
// stops at the wrappers of internal/dct and internal/ycc: the paths inside
// their assembly (the DC-only and rows-4-7-zero inverse DCTs, the opaque
// NRGBA blocks, the last block of a row that overlaps the one before)
// give it no signal, so that it rarely builds inputs that reach them.
// FuzzLevelKernels builds the kernels' inputs from its bytes in shapes
// that reach every such path, compares each kernel at the level under test
// with its *Ref, and classifies the input in Go (lvlHit), which makes each
// path a coverage edge the fuzzer can steer by. TestLevelKernelPaths
// checks that its seeds alone reach every path, so that go test ./jpeg
// runs each of them at every JPEG_TURBO_LEVEL.

// The paths lvlHit records, as the kernels see their input.
const (
	lvlIDCTDC       = iota // inverse DCT, only b[0] non-zero (the DC-only path)
	lvlIDCTTop             // rows 4-7 of b zero, an AC coefficient not
	lvlIDCTFull            // a coefficient of rows 4-7 non-zero
	lvlIDCTWrap            // a product b[i]*qt[i] wraps around int32
	lvlIDCTClamp           // DequantIDCTStore clamps an output to 0 or 255
	lvlFDCTStride0         // FDCTQuant of one row repeated (stride 0)
	lvlFDCTOverlap         // stride 1 to 7: the rows overlap
	lvlFDCTStride          // stride 8 or more
	lvlFDCTQ1              // a table entry of 1
	lvlFDCTZero            // the quantized block is zero
	lvlFDCTDC              // only its DC coefficient is non-zero
	lvlYCCShort            // a row shorter than the smallest vector block (16 pixels)
	lvlYCCExact            // n a multiple of a vector block of 16, 32 or 64 pixels
	lvlYCCTail             // n above such a block, not a multiple: the last block overlaps
	lvlYCCOdd              // odd n: the 4:2:0 kernels convert the last pixel apart
	lvlYCCOpaque           // for a block size, every block of row 0 opaque (the NRGBA alpha-255 path)
	lvlYCCMixed            // some of them opaque, some not
	lvlYCCClear            // none of them opaque
	lvlYCC420Opaque        // the same for the blocks of both rows (4:2:0)
	lvlYCC420Mixed
	lvlYCC420Clear
	lvlYCCAlpha0 // an NRGBA pixel of alpha 0
	lvlPaths
)

var lvlPathNames = [lvlPaths]string{
	"idct dc only", "idct rows 4-7 zero", "idct full", "idct int32 wrap", "idct clamp",
	"fdct stride 0", "fdct overlapping rows", "fdct stride >= 8", "fdct table entry 1",
	"fdct zero block", "fdct dc only",
	"ycc short row", "ycc whole blocks", "ycc overlapping last block", "ycc odd n",
	"ycc opaque blocks", "ycc mixed blocks", "ycc no opaque block",
	"ycc 4:2:0 opaque blocks", "ycc 4:2:0 mixed blocks", "ycc 4:2:0 no opaque block",
	"ycc alpha 0",
}

// lvlHits counts the inputs that took each path (fuzz inputs run one at a
// time in each process).
var lvlHits [lvlPaths]int

func lvlHit(p int) { lvlHits[p]++ }

// lvlSeeds are the seed corpus of FuzzLevelKernels.
var lvlSeeds = []struct {
	idct, fdct, conv uint8
	n                uint16
	data             []byte
}{
	{0x00, 0x01, 0x00, 64, lvlBytes(128)},
	{0x01, 0x11, 0x02, 100, lvlBytes(3)},
	{0x02, 0x24, 0x01, 255, lvlBytes(77)},
	{0x2a, 0x38, 0x03, 7, lvlBytes(200)},
	{0x3f, 0x0a, 0x06, 160, lvlBytes(0)},
}

// lvlBytes returns first and 511 pseudo-random bytes.
func lvlBytes(first byte) []byte {
	b := make([]byte, 512)
	b[0] = first
	x := uint32(first)
	for i := 1; i < len(b); i++ {
		x = x*1664525 + 1013904223
		b[i] = byte(x >> 24)
	}
	return b
}

// FuzzLevelKernels checks the kernels of internal/dct and internal/ycc at
// the level under test against their *Ref on inputs shaped by idct, fdct
// and conv (see lvlIDCT, lvlFDCT and lvlYCC) and filled from data.
func FuzzLevelKernels(f *testing.F) {
	for _, s := range lvlSeeds {
		f.Add(s.idct, s.fdct, s.conv, s.n, s.data)
	}
	f.Fuzz(lvlCheck)
}

func lvlCheck(t *testing.T, idct, fdct, conv uint8, n uint16, data []byte) {
	lvlIDCT(t, data, idct)
	lvlFDCT(t, data, fdct)
	lvlYCC(t, data, conv, int(n))
}

// TestLevelKernelPaths runs the seeds of FuzzLevelKernels and checks that
// together they take every path lvlHit records.
func TestLevelKernelPaths(t *testing.T) {
	lvlHits = [lvlPaths]int{}
	for _, s := range lvlSeeds {
		lvlCheck(t, s.idct, s.fdct, s.conv, s.n, s.data)
	}
	for p, k := range lvlHits {
		if k == 0 {
			t.Errorf("no seed of FuzzLevelKernels takes the path %q", lvlPathNames[p])
		}
	}
}

// lvlSource hands out the fuzzer's bytes in order and cyclically, so that
// any input fills a shape of any size (with zeros if it is empty).
type lvlSource struct {
	b []byte
	i int
}

func (s *lvlSource) next() byte {
	if len(s.b) == 0 {
		return 0
	}
	v := s.b[s.i%len(s.b)]
	s.i++
	return v
}

// coef returns a value of the class c&3: an int8 (the usual decoded
// coefficients), an int16, any int32, or a wrap or saturation point of
// the kernels' arithmetic plus a small offset.
func (s *lvlSource) coef(c byte) int32 {
	switch c & 3 {
	case 0:
		return int32(int8(s.next()))
	case 1:
		return int32(int16(uint16(s.next())<<8 | uint16(s.next())))
	case 2:
		return int32(uint32(s.next())<<24 | uint32(s.next())<<16 | uint32(s.next())<<8 | uint32(s.next()))
	}
	return lvlExtremes[int(s.next())%len(lvlExtremes)] + int32(int8(s.next()))>>4
}

var lvlExtremes = [...]int32{
	0, 1 << 10, -1 << 10, 1 << 11, -1 << 11, 1 << 14, -1 << 14, 1 << 15, -1 << 15,
	1 << 16, -1 << 16, 1 << 19, -1 << 19, 1 << 26, math.MaxInt32, math.MinInt32,
}

// lvlIDCT checks DequantIDCT and DequantIDCTStore on a block of the shape
// shape&3 (DC only, rows 0-3, every row, a few coefficients anywhere) with
// coefficients of the class shape>>2 (see coef) and a table of the class
// shape>>4 (8-bit or 16-bit entries as in a DQT segment, any int32, wrap
// points).
func lvlIDCT(t *testing.T, data []byte, shape byte) {
	s := &lvlSource{b: data}
	var b, qt dct.Block
	switch shape & 3 {
	case 0:
		b[0] = s.coef(shape >> 2)
	case 1:
		for i := range 32 {
			b[i] = s.coef(shape >> 2)
		}
	case 2:
		for i := range b {
			b[i] = s.coef(shape >> 2)
		}
	case 3:
		for range 1 + s.next()%4 {
			b[s.next()%dct.BlockSize] = s.coef(shape >> 2)
		}
	}
	for i := range qt {
		switch shape >> 4 & 3 {
		case 0:
			qt[i] = int32(s.next())
		case 1:
			qt[i] = int32(s.next())<<8 | int32(s.next())
		default:
			qt[i] = s.coef(shape >> 4)
		}
	}

	switch {
	case lvlZero(b[1:]):
		lvlHit(lvlIDCTDC)
	case lvlZero(b[32:]):
		lvlHit(lvlIDCTTop)
	default:
		lvlHit(lvlIDCTFull)
	}
	for i := range b {
		if p := int64(b[i]) * int64(qt[i]); p != int64(int32(p)) {
			lvlHit(lvlIDCTWrap)
			break
		}
	}

	want, got := b, b
	dct.DequantIDCTRef(&want, &qt)
	dct.DequantIDCT(&got, &qt)
	if got != want {
		t.Fatalf("%v: DequantIDCT(%v, %v) = %v, want %v", testLevel, b, qt, got, want)
	}
	for _, v := range want {
		if v < -128 || v > 127 {
			lvlHit(lvlIDCTClamp)
			break
		}
	}

	// The smallest destination the kernels accept, with bytes around it
	// that they must not write.
	stride, off := 8+int(s.next()%25), int(s.next()%8)
	m := 7*stride + 8
	wdst := make([]byte, off+m+8)
	for i := range wdst {
		wdst[i] = s.next()
	}
	gdst := bytes.Clone(wdst)
	wb, gb := b, b
	dct.DequantIDCTStoreRef(&wb, &qt, wdst[off:off+m], stride)
	dct.DequantIDCTStore(&gb, &qt, gdst[off:off+m], stride)
	if !bytes.Equal(gdst, wdst) {
		t.Fatalf("%v: DequantIDCTStore(%v, %v, stride %d) wrote\n%v, want\n%v", testLevel, b, qt, stride, gdst, wdst)
	}
}

func lvlZero(b []int32) bool {
	for _, v := range b {
		if v != 0 {
			return false
		}
	}
	return true
}

// lvlFDCT checks FDCTQuant on 8x8 pixels of the pattern shape&3 (the
// fuzzer's bytes, one flat value, 0 and 255, a sawtooth) with a stride of
// the class shape>>2 (0, 1 to 7, 8 and more) and a table of the class
// shape>>4 (the fuzzer's bytes, all 1, one value, entries 1, 2, 3 and
// 255; a zero entry would only run FDCTQuantRef).
func lvlFDCT(t *testing.T, data []byte, shape byte) {
	s := &lvlSource{b: data}
	stride := 0
	switch shape >> 2 & 3 {
	case 0:
		lvlHit(lvlFDCTStride0)
	case 1:
		stride = 1 + int(s.next()%7)
		lvlHit(lvlFDCTOverlap)
	default:
		stride = 8 + int(s.next()%40)
		lvlHit(lvlFDCTStride)
	}
	v, d := s.next(), s.next()
	src := make([]byte, 7*stride+8) // the shortest FDCTQuant accepts
	for i := range src {
		switch shape & 3 {
		case 0:
			src[i] = s.next()
		case 1:
			src[i] = v
		case 2:
			src[i] = -(s.next() & 1)
		case 3:
			src[i] = v + byte(i)*d
		}
	}
	var q [dct.BlockSize]byte
	for i := range q {
		switch shape >> 4 & 3 {
		case 0:
			q[i] = max(1, s.next())
		case 1:
			q[i] = 1
		case 2:
			q[i] = max(1, d)
		case 3:
			q[i] = [...]byte{1, 2, 3, 255}[s.next()%4]
		}
	}
	if bytes.IndexByte(q[:], 1) >= 0 {
		lvlHit(lvlFDCTQ1)
	}

	var tab dct.Quant
	tab.Init(&q)
	var want, got [dct.BlockSize]int16
	wnz := dct.FDCTQuantRef(src, stride, &tab, &want)
	gnz := dct.FDCTQuant(src, stride, &tab, &got)
	if gnz != wnz || got != want {
		t.Fatalf("%v: FDCTQuant(%v, stride %d, %v) = %v, %#x; want %v, %#x", testLevel, src, stride, q, got, gnz, want, wnz)
	}
	switch wnz {
	case 0:
		lvlHit(lvlFDCTZero)
	case 1 << 63:
		lvlHit(lvlFDCTDC)
	}
}

// lvlYCC checks the ycc conversions on two rows of n%320 pixels with
// alpha of the class shape&3 (opaque; the fuzzer's bytes; opaque but for
// one pixel; runs of 16 pixels opaque or not by the bits of a byte) and
// colors of the class shape>>2&1 (the fuzzer's bytes, 0 and 255). The rows
// have more of the fuzzer's bytes past their len, which must not change
// the results, and the outputs bytes past their len that the kernels must
// not write.
func lvlYCC(t *testing.T, data []byte, shape byte, n int) {
	s := &lvlSource{b: data}
	n %= 5 * 64
	var rows [2][]byte
	for r := range rows {
		row := make([]byte, 4*n+64)
		pick, runs := int(s.next())<<8|int(s.next()), s.next()
		for p := range n {
			px := row[4*p : 4*p+4]
			for c := range 3 {
				px[c] = s.next()
				if shape>>2&1 != 0 {
					px[c] = -(px[c] & 1)
				}
			}
			px[3] = 255
			switch {
			case shape&3 == 1,
				shape&3 == 2 && p == pick%n,
				shape&3 == 3 && runs>>(p/16%8)&1 == 0:
				px[3] = s.next()
			}
		}
		for i := 4 * n; i < len(row); i++ {
			row[i] = s.next()
		}
		rows[r] = row[:4*n]
	}
	lvlYCCPaths(rows, n)

	for _, k := range []struct {
		name    string
		fn, ref func(src, y, cb, cr []byte, n int)
	}{
		{"RGBAToYCbCr", ycc.RGBAToYCbCr, ycc.RGBAToYCbCrRef},
		{"NRGBAToYCbCr", ycc.NRGBAToYCbCr, ycc.NRGBAToYCbCrRef},
	} {
		want, got := lvlPlanes(n, n, n), lvlPlanes(n, n, n)
		k.ref(rows[0], want[0], want[1], want[2], n)
		k.fn(rows[0], got[0], got[1], got[2], n)
		lvlComparePlanes(t, k.name, n, got, want)
	}
	for _, k := range []struct {
		name    string
		fn, ref func(src0, src1, y0, y1, cb, cr []byte, n int)
	}{
		{"RGBAToYCbCr420", ycc.RGBAToYCbCr420, ycc.RGBAToYCbCr420Ref},
		{"NRGBAToYCbCr420", ycc.NRGBAToYCbCr420, ycc.NRGBAToYCbCr420Ref},
	} {
		want, got := lvlPlanes(n, n, n/2, n/2), lvlPlanes(n, n, n/2, n/2)
		k.ref(rows[0], rows[1], want[0], want[1], want[2], want[3], n)
		k.fn(rows[0], rows[1], got[0], got[1], got[2], got[3], n)
		lvlComparePlanes(t, k.name, n, got, want)
	}
	want, got := lvlPlanes(2*n), lvlPlanes(2*n)
	ycc.Downsample2x2Ref(rows[0], rows[1], want[0], 2*n)
	ycc.Downsample2x2(rows[0], rows[1], got[0], 2*n)
	lvlComparePlanes(t, "Downsample2x2", 2*n, got, want)
}

// lvlYCCPaths records the paths of the x86 kernels (blocks of 16, 32 and
// 64 pixels; NEON: 16) for the rows of n pixels.
func lvlYCCPaths(rows [2][]byte, n int) {
	if n < 16 {
		lvlHit(lvlYCCShort)
	}
	if n%2 != 0 {
		lvlHit(lvlYCCOdd)
	}
	for _, bs := range []int{16, 32, 64} {
		if n < bs {
			break
		}
		if n%bs == 0 {
			lvlHit(lvlYCCExact)
		} else {
			lvlHit(lvlYCCTail)
		}
		// The blocks the kernels convert: at 0, bs, 2bs... and the last
		// one at n-bs.
		blocks, opaque, opaque2 := 0, 0, 0
		for x := 0; x < n; x += bs {
			x0 := min(x, n-bs)
			blocks++
			if lvlOpaque(rows[0][4*x0 : 4*(x0+bs)]) {
				opaque++
				if lvlOpaque(rows[1][4*x0 : 4*(x0+bs)]) {
					opaque2++
				}
			}
		}
		lvlOpaqueClass(lvlYCCOpaque, opaque, blocks)
		lvlOpaqueClass(lvlYCC420Opaque, opaque2, blocks)
	}
	for p := range n {
		if rows[0][4*p+3] == 0 || rows[1][4*p+3] == 0 {
			lvlHit(lvlYCCAlpha0)
			break
		}
	}
}

// lvlOpaqueClass records path (all of the blocks opaque), path+1 (some)
// or path+2 (none).
func lvlOpaqueClass(path, opaque, blocks int) {
	switch opaque {
	case blocks:
		lvlHit(path)
	case 0:
		lvlHit(path + 2)
	default:
		lvlHit(path + 1)
	}
}

func lvlOpaque(px []byte) bool {
	for i := 3; i < len(px); i += 4 {
		if px[i] != 255 {
			return false
		}
	}
	return true
}

// lvlCanary fills the bytes past the len of the outputs.
const lvlCanary = 0xa5

// lvlPlanes returns outputs of the lengths ns, each with 64 bytes of
// lvlCanary past its len.
func lvlPlanes(ns ...int) [][]byte {
	ps := make([][]byte, len(ns))
	for i, n := range ns {
		p := bytes.Repeat([]byte{lvlCanary}, n+64)
		ps[i] = p[:n]
	}
	return ps
}

func lvlComparePlanes(t *testing.T, name string, n int, got, want [][]byte) {
	t.Helper()
	for i := range got {
		if !bytes.Equal(got[i], want[i]) {
			t.Fatalf("%v: %s, n %d: output %d is\n%v, want\n%v", testLevel, name, n, i, got[i], want[i])
		}
		for _, c := range got[i][len(got[i]):cap(got[i])] {
			if c != lvlCanary {
				t.Fatalf("%v: %s, n %d: output %d written past its len", testLevel, name, n, i)
			}
		}
	}
}
