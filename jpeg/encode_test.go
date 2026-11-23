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

// Targeted tests of the encoder: byte-exactness against image/jpeg (std)
// for the layouts the strip pipeline special-cases, the entropy coder
// against image/jpeg's emit, the writer handling call by call, panics on
// malformed images, and allocations.

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/color/palette"
	std "image/jpeg"
	"io"
	encbits "math/bits"
	"math/rand"
	"sync"
	"testing"

	"github.com/lEx0/go-jpeg-turbo/internal/dct"
	"github.com/lEx0/go-jpeg-turbo/internal/ycc"
)

// encStd and encOurs encode to a bytes.Buffer, recovering panics.
func encStd(m image.Image, o *Options) encodeResult { return encodeStd(m, o) }

func encOurs(m image.Image, o *Options) encodeResult { return encodeOurs(m, o, false) }

func encQ(q int) *Options { return &Options{Quality: q} }

func encName(o *Options) string {
	if o == nil {
		return "nil"
	}
	return fmt.Sprint(o.Quality)
}

// encNoise is a deterministic noise fill.
func encNoise(seed int64, alpha bool) func(x, y int) color.NRGBA64 {
	return func(x, y int) color.NRGBA64 {
		h := uint64(x)*0x9E3779B97F4A7C15 ^ uint64(y)*0xC2B2AE3D27D4EB4F ^ uint64(seed)*0x165667B19E3779F9
		h ^= h >> 29
		h *= 0xBF58476D1CE4E5B9
		h ^= h >> 32
		a := uint16(0xffff)
		if alpha {
			a = uint16(h >> 48)
		}
		return color.NRGBA64{uint16(h), uint16(h >> 16), uint16(h >> 32), a}
	}
}

// encExtraMakers complements imageMakers with the other layouts the
// encoder special-cases.
var encExtraMakers = func() []imageMaker {
	ms := []imageMaker{
		{"Alpha16", func(r image.Rectangle, f func(x, y int) color.NRGBA64) image.Image {
			m := image.NewAlpha16(r)
			setAll(m, f)
			return m
		}},
		{"PalettedShort", func(r image.Rectangle, f func(x, y int) color.NRGBA64) image.Image {
			m := image.NewPaletted(r, palette.WebSafe[:100])
			setAll(m, f)
			return m
		}},
		{"PalettedMixed", func(r image.Rectangle, f func(x, y int) color.NRGBA64) image.Image {
			m := image.NewPaletted(r, color.Palette{color.Gray16{0x1234}, color.NRGBA{200, 100, 50, 128}, color.CMYK{9, 80, 7, 6}, color.NYCbCrA{color.YCbCr{1, 200, 3}, 44}, color.Alpha16{0xfedc}})
			setAll(m, f)
			return m
		}},
	}
	for _, ratio := range encRatios {
		ms = append(ms, imageMaker{fmt.Sprintf("NYCbCrA%v", ratio), func(r image.Rectangle, f func(x, y int) color.NRGBA64) image.Image {
			m := image.NewNYCbCrA(r, ratio)
			m.YCbCr = *ycbcrImage(r, ratio, f)
			for y := r.Min.Y; y < r.Max.Y; y++ {
				for x := r.Min.X; x < r.Max.X; x++ {
					m.A[m.AOffset(x, y)] = uint8(f(x, y).A >> 8)
				}
			}
			return m
		}})
		ms = append(ms, imageMaker{fmt.Sprintf("YCbCr%v", ratio), func(r image.Rectangle, f func(x, y int) color.NRGBA64) image.Image {
			return ycbcrImage(r, ratio, f)
		}})
	}
	return ms
}()

func TestEncSizes(t *testing.T) {
	var sizes []int
	if testing.Short() {
		sizes = []int{1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33}
	} else {
		for i := 1; i <= 40; i++ {
			sizes = append(sizes, i)
		}
	}
	for _, mk := range append(imageMakers[:len(imageMakers):len(imageMakers)], encExtraMakers...) {
		t.Run(mk.name, func(t *testing.T) {
			t.Parallel()
			for _, w := range sizes {
				for _, h := range sizes {
					fill := encNoise(int64(w*64+h), (w+h)%3 == 0)
					m := mk.make(image.Rect(0, 0, w, h), fill)
					compareEncode(t, fmt.Sprintf("%s/%dx%d", mk.name, w, h), m, nil, false)
				}
			}
		})
	}
}

func TestEncOrigins(t *testing.T) {
	origins := []image.Point{{-1, -1}, {-7, 3}, {5, -6}, {1, 0}, {0, 1}, {-2, -2}, {3, 3}, {-65, -17}, {1001, -999}}
	sizes := []image.Point{{1, 1}, {2, 2}, {5, 3}, {16, 16}, {17, 9}, {33, 18}}
	for _, mk := range append(imageMakers[:len(imageMakers):len(imageMakers)], encExtraMakers...) {
		t.Run(mk.name, func(t *testing.T) {
			t.Parallel()
			for _, o := range origins {
				for _, sz := range sizes {
					r := image.Rectangle{o, o.Add(sz)}
					m := mk.make(r, encNoise(int64(o.X*7+o.Y), true))
					for _, q := range []*Options{nil, encQ(100)} {
						compareEncode(t, fmt.Sprintf("%s/%v/q%s", mk.name, r, encName(q)), m, q, false)
					}
				}
			}
		})
	}
}

var encRatios = []image.YCbCrSubsampleRatio{
	image.YCbCrSubsampleRatio444, image.YCbCrSubsampleRatio422, image.YCbCrSubsampleRatio420,
	image.YCbCrSubsampleRatio440, image.YCbCrSubsampleRatio411, image.YCbCrSubsampleRatio410,
	image.YCbCrSubsampleRatio(99), // COffset treats unknown ratios as 4:4:4
}

// encRandYCbCr returns a YCbCr image with random samples (all of them,
// including those no pixel uses).
func encRandYCbCr(rng *rand.Rand, r image.Rectangle, ratio image.YCbCrSubsampleRatio) *image.YCbCr {
	m := image.NewYCbCr(r, ratio)
	rng.Read(m.Y)
	rng.Read(m.Cb)
	rng.Read(m.Cr)
	return m
}

func TestEncYCbCrSubImages(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	step := 1
	if testing.Short() {
		step = 3
	}
	for _, ratio := range encRatios {
		for _, base := range []image.Rectangle{image.Rect(0, 0, 44, 41), image.Rect(-9, -7, 35, 34), image.Rect(3, 5, 47, 46)} {
			m := encRandYCbCr(rng, base, ratio)
			for dx := 0; dx < 8; dx += step {
				for dy := 0; dy < 8; dy += step {
					for _, sz := range []image.Point{{1, 1}, {2, 3}, {9, 9}, {17, 16}, {20, 23}, {33, 31}} {
						min := base.Min.Add(image.Pt(dx, dy))
						r := image.Rectangle{min, min.Add(sz)}.Intersect(base)
						compareEncode(t, fmt.Sprintf("%v/%v/%v", ratio, base, r), m.SubImage(r), nil, false)
					}
				}
			}
		}
	}
}

// TestEncYCbCrStrides covers chroma and luma strides larger than the
// rows, and the fast 4:2:0/4:2:2 paths with all sizes around their tails.
func TestEncYCbCrStrides(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for _, ratio := range encRatios {
		for w := 1; w <= 35; w += 2 {
			for _, h := range []int{1, 2, 15, 16, 17} {
				r := image.Rect(0, 0, w, h)
				m := encRandYCbCr(rng, image.Rect(0, 0, w+13, h+5), ratio)
				sub := *m
				sub.Rect = r
				compareEncode(t, fmt.Sprintf("%v/%dx%d", ratio, w, h), &sub, encQ(97), false)
			}
		}
	}
}

// encPatterns are images built to stress the entropy coder: extreme DC
// differences, maximal AC coefficients, long zero runs.
func encPatterns() map[string]*image.RGBA {
	const w, h = 96, 64
	ps := map[string]*image.RGBA{}
	set := func(name string, f func(x, y int) uint8) {
		m := image.NewRGBA(image.Rect(0, 0, w, h))
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				v := f(x, y)
				i := m.PixOffset(x, y)
				m.Pix[i], m.Pix[i+1], m.Pix[i+2], m.Pix[i+3] = v, 255-v, v^0x5a, 255
			}
		}
		ps[name] = m
	}
	set("dcjumps", func(x, y int) uint8 { return uint8(-((x/8 + y/8) & 1)) })
	set("checker", func(x, y int) uint8 { return uint8(-((x + y) & 1)) })
	set("stripes", func(x, y int) uint8 { return uint8(-(x & 1)) })
	set("hstripes", func(x, y int) uint8 { return uint8(-(y & 1)) })
	set("corner", func(x, y int) uint8 {
		if x%8 == 7 && y%8 == 7 {
			return 255
		}
		return 0
	})
	set("ramp", func(x, y int) uint8 { return uint8(x*37 + y*91) })
	rng := rand.New(rand.NewSource(3))
	set("noise", func(x, y int) uint8 { return uint8(rng.Intn(256)) })
	return ps
}

func TestEncQualityExtremes(t *testing.T) {
	for name, m := range encPatterns() {
		for _, q := range []*Options{encQ(1), encQ(2), encQ(50), encQ(99), encQ(100)} {
			compareEncode(t, fmt.Sprintf("%s/q%s", name, encName(q)), m, q, false)
			var g image.Gray
			g.Pix, g.Stride, g.Rect = make([]byte, 96*64), 96, m.Rect
			for i := range g.Pix {
				g.Pix[i] = m.Pix[4*i]
			}
			compareEncode(t, fmt.Sprintf("%s/gray/q%s", name, encName(q)), &g, q, false)
		}
	}
}

// TestEncStuffing checks that outputs with many 0xff bytes are identical
// and that they do exercise the stuffing.
func TestEncStuffing(t *testing.T) {
	total := 0
	for name, m := range encPatterns() {
		for _, q := range []*Options{encQ(100), encQ(95), nil} {
			got, want := encOurs(m, q), encStd(m, q)
			if !bytes.Equal(got.out, want.out) || got.err != nil {
				t.Errorf("%s/q%s: output differs (err %v)", name, encName(q), got.err)
			}
			total += bytes.Count(got.out, []byte{0xff, 0x00})
		}
	}
	if total < 1000 {
		t.Errorf("only %d stuffed bytes in all outputs", total)
	}
}

// Entropy coder against image/jpeg's emit (copied below).

// encRefCoder is image/jpeg's bit writer and block coder, on quantized
// coefficients.
type encRefCoder struct {
	out         []byte
	bits, nBits uint32
	lut         [nHuffIndex]huffmanLUT
}

func encNewRefCoder() *encRefCoder {
	c := &encRefCoder{}
	for i, s := range theHuffmanSpec {
		c.lut[i].init(s)
	}
	return c
}

func (e *encRefCoder) emit(bits, nBits uint32) {
	nBits += e.nBits
	bits <<= 32 - nBits
	bits |= e.bits
	for nBits >= 8 {
		b := uint8(bits >> 24)
		e.out = append(e.out, b)
		if b == 0xff {
			e.out = append(e.out, 0x00)
		}
		bits <<= 8
		nBits -= 8
	}
	e.bits, e.nBits = bits, nBits
}

func (e *encRefCoder) emitHuff(h huffIndex, value int32) {
	x := e.lut[h][value]
	e.emit(x&(1<<24-1), x>>24)
}

func (e *encRefCoder) emitHuffRLE(h huffIndex, runLength, value int32) {
	a, b := value, value
	if a < 0 {
		a, b = -value, value-1
	}
	var nBits uint32
	if a < 0x100 {
		nBits = uint32(encbits.Len8(uint8(a)))
	} else {
		nBits = 8 + uint32(encbits.Len8(uint8(a>>8)))
	}
	e.emitHuff(h, runLength<<4|int32(nBits))
	if nBits > 0 {
		e.emit(uint32(b)&(1<<nBits-1), nBits)
	}
}

func (e *encRefCoder) writeCoefs(coef *[blockSize]int16, q quantIndex, prevDC int32) int32 {
	dc := int32(coef[0])
	e.emitHuffRLE(huffIndex(2*q+0), 0, dc-prevDC)
	h, runLength := huffIndex(2*q+1), int32(0)
	for zig := 1; zig < blockSize; zig++ {
		ac := int32(coef[zig])
		if ac == 0 {
			runLength++
		} else {
			for runLength > 15 {
				e.emitHuff(h, 0xf0)
				runLength -= 16
			}
			e.emitHuffRLE(h, runLength, ac)
			runLength = 0
		}
	}
	if runLength > 0 {
		e.emitHuff(h, 0x00)
	}
	return dc
}

// encRandBlock fills coef with a random block of one of several shapes;
// AC values stay within the 10 bits of size that baseline allows, DC
// within [-1024, 1023].
func encRandBlock(rng *rand.Rand, coef *[blockSize]int16) {
	*coef = [blockSize]int16{}
	val := func(maxBits int) int16 {
		s := 1 + rng.Intn(maxBits)
		v := int16(1<<(s-1) + rng.Intn(1<<(s-1)))
		if rng.Intn(2) == 0 {
			v = -v
		}
		return v
	}
	switch rng.Intn(4) {
	case 0:
		coef[0] = int16(-1024 + rng.Intn(2048))
	case 1:
		coef[0] = []int16{-1024, 1023, 0, -1, 1}[rng.Intn(5)]
	default:
		coef[0] = int16(rng.Intn(64) - 32)
	}
	switch rng.Intn(7) {
	case 0: // DC only
	case 1: // dense
		for i := 1; i < blockSize; i++ {
			if rng.Intn(4) != 0 {
				coef[i] = val(10)
			}
		}
	case 2: // sparse, long runs
		for k := rng.Intn(4); k >= 0; k-- {
			coef[1+rng.Intn(63)] = val(10)
		}
	case 3: // the last coefficient set (no EOB)
		coef[63] = val(10)
		if rng.Intn(2) == 0 {
			coef[1+rng.Intn(15)] = val(3)
		}
	case 4: // runs of exactly 15, 16, 17, 31, 32, 33...
		for i := 1 + rng.Intn(3); i < blockSize; i += 15 + rng.Intn(4) {
			coef[i] = val(10)
		}
	case 5: // long codes: large values after runs of 15 (codes of all 1's)
		for i := 16; i < blockSize; i += 16 {
			coef[i] = int16(1023 - rng.Intn(4))
			if rng.Intn(2) == 0 {
				coef[i] = -coef[i]
			}
		}
	case 6: // small values everywhere
		for i := 1; i < blockSize; i++ {
			coef[i] = int16(rng.Intn(3) - 1)
		}
	}
}

func encNZ(coef *[blockSize]int16) uint64 {
	var nz uint64
	for i, v := range coef {
		if v != 0 {
			nz |= 1 << (63 - i)
		}
	}
	return nz
}

func TestEncEntropyCoder(t *testing.T) {
	n := 200000
	if testing.Short() {
		n = 20000
	}
	rng := rand.New(rand.NewSource(4))
	ref := encNewRefCoder()
	e := new(encoder)
	e.encStart(io.Discard)
	var got []byte
	var coef [blockSize]int16
	var prevRef [3]int32
	for i := 0; i < n; {
		// A batch of 1 to encBatchMax blocks of random components.
		nb := 1 + rng.Intn(encBatchMax)
		for k := 0; k < nb; k++ {
			encRandBlock(rng, &coef)
			c := rng.Intn(3)
			prevRef[c] = ref.writeCoefs(&coef, quantIndex(min(c, 1)), prevRef[c])
			e.coefs[k], e.nzs[k], e.comps[k] = coef, encNZ(&coef), uint8(c)
		}
		e.nb = nb
		e.encCode()
		i += nb
		if cap(e.out)-len(e.out) < encOutReserve {
			got = append(got, e.out...)
			e.out = e.out[:0]
		}
		if e.dc != prevRef {
			t.Fatalf("block %d: DC %v, want %v", i, e.dc, prevRef)
		}
	}
	ref.emit(0x7f, 7)
	e.encFinish()
	got = append(got, e.out...)
	if !bytes.Equal(got, ref.out) {
		i := 0
		for i < len(got) && i < len(ref.out) && got[i] == ref.out[i] {
			i++
		}
		t.Fatalf("streams differ at byte %d (len %d, want %d)", i, len(got), len(ref.out))
	}
	if k := bytes.Count(got, []byte{0xff, 0}); k < n/100 {
		t.Errorf("only %d stuffed bytes", k)
	}
}

// TestEncFinishAlignments checks the final padding after every number of
// pending bits.
func TestEncFinishAlignments(t *testing.T) {
	for nb := 0; nb <= 64; nb++ {
		for _, pat := range []uint64{0, ^uint64(0), 0xaaaaaaaaaaaaaaaa, 0xff00ff00ff00ff7f} {
			ref := encNewRefCoder()
			e := new(encoder)
			e.encStart(io.Discard)
			for k := 0; k < nb; k += 16 {
				l := min(16, nb-k)
				v := uint32(pat>>k) & (1<<l - 1)
				ref.emit(v, uint32(l))
				e.bits = e.bits<<l | uint64(v)
				e.nBits += uint(l)
			}
			ref.emit(0x7f, 7)
			e.encFinish()
			if !bytes.Equal(e.out, ref.out) {
				t.Errorf("%d bits of %x: got % x, want % x", nb, pat, e.out, ref.out)
			}
		}
	}
}

func TestEncPutWord(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	buf := make([]byte, 16)
	for i := 0; i < 100000; i++ {
		w := rng.Uint64()
		for k := 0; k < 8; k++ {
			if rng.Intn(3) == 0 {
				w |= 0xff << (8 * k)
			}
		}
		var want []byte
		for k := 56; k >= 0; k -= 8 {
			want = append(want, byte(w>>k))
			if byte(w>>k) == 0xff {
				want = append(want, 0)
			}
		}
		n := encPutWordFF(buf, 0, w)
		if !bytes.Equal(buf[:n], want) {
			t.Fatalf("%016x: % x, want % x", w, buf[:n], want)
		}
	}
}

func TestEncHuffTables(t *testing.T) {
	for c := 0; c < 2; c++ {
		var dc, ac huffmanLUT
		dc.init(theHuffmanSpec[2*c])
		ac.init(theHuffmanSpec[2*c+1])
		check := func(name string, lut huffmanLUT, i int, v uint64) {
			var x uint32
			if i < len(lut) {
				x = lut[i]
			}
			s := uint(i & 15)
			if want := uint64(x&0xffffff)<<s | uint64(x>>24+uint32(s))<<32; v != want {
				t.Errorf("class %d %s[%#x] = %x, want %x", c, name, i, v, want)
			}
		}
		for i, v := range encHuffTables[c].dc {
			check("dc", dc, i, v)
		}
		for i, v := range encHuffTables[c].ac {
			check("ac", ac, i, v)
		}
	}
}

// Writers.

// encRecWriter records the calls it gets; it fails (or, with short,
// returns a short count and no error) once more than fail bytes would have
// been written, if fail >= 0.
type encRecWriter struct {
	buf   bytes.Buffer
	calls []int
	fail  int
	short bool
}

func (w *encRecWriter) Write(p []byte) (int, error) {
	w.calls = append(w.calls, len(p))
	if w.fail >= 0 && w.buf.Len()+len(p) > w.fail {
		k := w.fail - w.buf.Len()
		w.buf.Write(p[:k])
		if w.short {
			return k, nil
		}
		return k, errTestReader
	}
	w.buf.Write(p)
	return len(p), nil
}

// encFlushWriter is a writer (Flush, Write, WriteByte) that is not a
// bufio.Writer.
type encFlushWriter struct {
	encRecWriter
	flushes int
}

func (w *encFlushWriter) WriteByte(b byte) error {
	_, err := w.Write([]byte{b})
	return err
}

func (w *encFlushWriter) Flush() error {
	w.flushes++
	return nil
}

type encWriterRun struct {
	err     error
	panic   any
	out     []byte
	calls   []int
	flushes int
	buffed  int
}

func (r encWriterRun) String() string {
	return fmt.Sprintf("err=%v panic=%v len=%d calls=%d flushes=%d buffered=%d", r.err, r.panic, len(r.out), len(r.calls), r.flushes, r.buffed)
}

// encRunWriter encodes m with enc to a writer of the given kind: "raw"
// (an io.Writer), "bufioN" (a bufio.Writer of size N over it), "rw" (a
// bufio.ReadWriter), or "flush" (encFlushWriter).
func encRunWriter(enc func(io.Writer) error, kind string, fail int, short bool) (r encWriterRun) {
	rec := &encRecWriter{fail: fail, short: short}
	fw := &encFlushWriter{encRecWriter: encRecWriter{fail: fail, short: short}}
	var bw *bufio.Writer
	defer func() {
		r.panic = recover()
		if kind == "flush" {
			rec = &fw.encRecWriter
			r.flushes = fw.flushes
		}
		r.out, r.calls = rec.buf.Bytes(), rec.calls
		if bw != nil {
			r.buffed = bw.Buffered()
		}
	}()
	var size int
	switch {
	case kind == "raw":
		r.err = enc(rec)
	case kind == "flush":
		r.err = enc(fw)
	case kind == "rw":
		bw = bufio.NewWriterSize(rec, 100)
		r.err = enc(bufio.NewReadWriter(bufio.NewReader(bytes.NewReader(nil)), bw))
	default:
		// "bufioN", or "bufioNpK": with K bytes already buffered.
		var prefix int
		if _, err := fmt.Sscanf(kind, "bufio%dp%d", &size, &prefix); err != nil {
			if _, err := fmt.Sscanf(kind, "bufio%d", &size); err != nil {
				panic(kind)
			}
		}
		bw = bufio.NewWriterSize(rec, size)
		bw.Write(bytes.Repeat([]byte{0xab}, prefix))
		r.err = enc(bw)
	}
	return r
}

func encSameRun(got, want encWriterRun, calls bool) bool {
	if (got.panic == nil) != (want.panic == nil) || !sameError(got.err, want.err) || errors.Is(got.err, errTestReader) != errors.Is(want.err, errTestReader) ||
		got.err != want.err && (got.err == io.ErrShortWrite || want.err == io.ErrShortWrite) {
		return false
	}
	if !bytes.Equal(got.out, want.out) || got.flushes != want.flushes || got.buffed != want.buffed {
		return false
	}
	if calls && fmt.Sprint(got.calls) != fmt.Sprint(want.calls) {
		return false
	}
	return true
}

var encWriterKinds = []string{"raw", "bufio1", "bufio7", "bufio64", "bufio64p37", "bufio4096", "bufio4096p4095", "bufio100000", "rw", "flush"}

// TestEncWriters checks, for every kind of writer, that the writer sees the
// same bytes as with image/jpeg, and (except for encFlushWriter, to which
// image/jpeg writes the scan byte by byte) the very same Write calls.
func TestEncWriters(t *testing.T) {
	ims := map[string]image.Image{
		"rgba100x75":  imageMakers[0].make(image.Rect(0, 0, 100, 75), fills(3)[0].f),
		"gray300x200": imageMakers[5].make(image.Rect(0, 0, 300, 200), fills(4)[0].f),
		"ycc1x1":      imageMakers[12].make(image.Rect(0, 0, 1, 1), fills(4)[0].f),
		"empty":       image.NewRGBA(image.Rect(0, 0, 0, 0)),
	}
	for name, m := range ims {
		for _, kind := range encWriterKinds {
			for _, q := range []int{1, 75, 100} {
				got := encRunWriter(func(w io.Writer) error { return Encode(w, m, encQ(q)) }, kind, -1, false)
				want := encRunWriter(func(w io.Writer) error { return std.Encode(w, m, &std.Options{Quality: q}) }, kind, -1, false)
				if !encSameRun(got, want, kind != "flush") {
					t.Errorf("%s/%s/q%d: got %v, want %v", name, kind, q, got, want)
				}
			}
		}
	}
}

// TestEncWriterErrors makes the writer fail (or write short) after every
// byte offset of the output.
func TestEncWriterErrors(t *testing.T) {
	small := imageMakers[0].make(image.Rect(0, 0, 40, 24), fills(3)[0].f) // ~2 KB
	big := imageMakers[0].make(image.Rect(0, 0, 100, 75), fills(3)[0].f)  // ~10 KB
	for _, tc := range []struct {
		name string
		m    image.Image
		q    int
		step int
	}{{"small", small, 90, 1}, {"big", big, 100, 17}} {
		full := encStd(tc.m, encQ(tc.q)).out
		step := tc.step
		if testing.Short() {
			step = 61
		}
		var offsets []int
		for n := 0; n <= len(full)+1; n += step {
			offsets = append(offsets, n)
		}
		// Around the chunk boundaries of image/jpeg's bufio.Writer, of
		// e.out, and the end.
		for _, b := range []int{4096, 8192, 12288, encOutSize - encOutReserve, len(full)} {
			for d := -2; d <= 2; d++ {
				offsets = append(offsets, b+d)
			}
		}
		for _, kind := range []string{"raw", "bufio64", "bufio4096", "flush"} {
			for _, short := range []bool{false, true} {
				if short && kind != "raw" && kind != "bufio4096" {
					// With "flush", image/jpeg ignores the count a
					// Write returns. With a small bufio.Writer, it
					// loops forever on a short write of a header
					// segment longer than the buffer (bufio.Writer.Write
					// retries large writes, and we make the same calls).
					continue
				}
				for _, n := range offsets {
					got := encRunWriter(func(w io.Writer) error { return Encode(w, tc.m, encQ(tc.q)) }, kind, n, short)
					want := encRunWriter(func(w io.Writer) error { return std.Encode(w, tc.m, &std.Options{Quality: tc.q}) }, kind, n, short)
					if !encSameRun(got, want, kind != "flush") {
						t.Fatalf("%s/%s/short=%v: failing after %d bytes: got %v, want %v", tc.name, kind, short, n, got, want)
					}
				}
			}
		}
	}
}

// encStickyWriter fails once, then accepts everything.
type encStickyWriter struct {
	encRecWriter
	failed bool
}

func (w *encStickyWriter) Write(p []byte) (int, error) {
	if !w.failed && w.buf.Len()+len(p) > 5000 {
		w.failed = true
		return 0, errTestReader
	}
	return w.encRecWriter.Write(p)
}

func TestEncWriterRecovers(t *testing.T) {
	m := imageMakers[0].make(image.Rect(0, 0, 100, 75), fills(3)[0].f)
	got := &encStickyWriter{encRecWriter: encRecWriter{fail: -1}}
	want := &encStickyWriter{encRecWriter: encRecWriter{fail: -1}}
	gerr, werr := Encode(got, m, encQ(100)), std.Encode(want, m, &std.Options{Quality: 100})
	if gerr != werr || !bytes.Equal(got.buf.Bytes(), want.buf.Bytes()) || fmt.Sprint(got.calls) != fmt.Sprint(want.calls) {
		t.Errorf("got %v %d %v, want %v %d %v", gerr, got.buf.Len(), got.calls, werr, want.buf.Len(), want.calls)
	}
}

// TestEncWriterErrorReadsPixels checks that, as image/jpeg does, Encode
// keeps calling At after the writer fails.
func TestEncWriterErrorReadsPixels(t *testing.T) {
	m := &encCountImage{Image: imageMakers[2].make(image.Rect(0, 0, 37, 21), fills(5)[0].f)}
	if err := Encode(&errWriter{n: 0}, m, nil); err != errTestReader {
		t.Fatal(err)
	}
	if m.n < 37*21 {
		t.Errorf("At called %d times, want >= %d", m.n, 37*21)
	}
	p := &encPanicImage{Image: m.Image, at: image.Pt(36, 20)}
	if r := encOurs(p, nil); r.panic == nil {
		t.Errorf("At panicking at the last pixel: no panic")
	}
}

type encCountImage struct {
	image.Image
	n int
}

func (m *encCountImage) At(x, y int) color.Color { m.n++; return m.Image.At(x, y) }

type encPanicImage struct {
	image.Image
	at image.Point
}

func (m *encPanicImage) At(x, y int) color.Color {
	if image.Pt(x, y) == m.at {
		panic("At")
	}
	return m.Image.At(x, y)
}

func TestEncHuge(t *testing.T) {
	if testing.Short() {
		t.Skip("slow")
	}
	for _, r := range []image.Rectangle{image.Rect(0, 0, 65535, 1), image.Rect(0, 0, 1, 65535), image.Rect(-3, -1, 65532, 18), image.Rect(0, 0, 33, 65535)} {
		for _, mk := range []imageMaker{imageMakers[0], imageMakers[2], imageMakers[5], imageMakers[12], imageMakers[16]} {
			m := mk.make(r, encNoise(int64(r.Dx()), false))
			compareEncode(t, fmt.Sprintf("%s/%v", mk.name, r), m, nil, false)
		}
	}
}

// TestEncConcurrent encodes from many goroutines at once (run with -race).
func TestEncConcurrent(t *testing.T) {
	type job struct {
		m    image.Image
		q    *Options
		want []byte
	}
	var jobs []job
	for i, mk := range imageMakers {
		r := image.Rect(i, -i, i+17+9*i, 23+5*i)
		m := mk.make(r, encNoise(int64(i), i%2 == 0))
		for _, q := range []*Options{nil, encQ(10 * i)} {
			jobs = append(jobs, job{m, q, encStd(m, q).out})
		}
	}
	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for k := 0; k < 3*len(jobs); k++ {
				j := jobs[(g*7+k)%len(jobs)]
				var buf bytes.Buffer
				var w io.Writer = &buf
				if k%3 == 1 {
					w = bufio.NewWriter(&buf)
				}
				if err := Encode(w, j.m, j.q); err != nil {
					t.Error(err)
					return
				}
				if !bytes.Equal(buf.Bytes(), j.want) {
					t.Errorf("goroutine %d job %d: output differs", g, k)
					return
				}
			}
		}(g)
	}
	wg.Wait()
}

func TestEncAllocs(t *testing.T) {
	if encRaceEnabled {
		t.Skip("sync.Pool drops items under the race detector")
	}
	cases := []struct {
		name string
		m    image.Image
		// conv is a color conversion kernel the encoder calls for m (it
		// may allocate in a reference implementation).
		conv func()
	}{
		{"Gray", imageMakers[5].make(image.Rect(0, 0, 200, 100), fills(1)[0].f), nil},
		{"YCbCr444", imageMakers[10].make(image.Rect(0, 0, 200, 100), fills(1)[0].f), nil},
		{"YCbCr420", imageMakers[12].make(image.Rect(0, 0, 200, 100), fills(1)[0].f), nil},
		{"YCbCr422", imageMakers[11].make(image.Rect(0, 0, 200, 100), fills(1)[0].f), nil},
		{"YCbCr410", imageMakers[15].make(image.Rect(0, 0, 200, 100), fills(1)[0].f), nil},
		{"RGBA", imageMakers[0].make(image.Rect(0, 0, 200, 100), fills(1)[0].f), encAllocConv(false)},
		{"NRGBA", imageMakers[2].make(image.Rect(0, 0, 200, 100), fills(1)[0].f), encAllocConv(true)},
		{"CMYK", imageMakers[8].make(image.Rect(0, 0, 200, 100), fills(1)[0].f), encAllocConv(false)},
	}
	for _, c := range cases {
		if c.conv != nil && testing.AllocsPerRun(10, c.conv) > 0 {
			t.Logf("%s: skipped, the conversion kernel allocates", c.name)
			continue
		}
		for _, w := range []io.Writer{io.Discard, bufio.NewWriter(io.Discard)} {
			n := testing.AllocsPerRun(20, func() {
				if err := Encode(w, c.m, nil); err != nil {
					t.Fatal(err)
				}
			})
			if n > 0 {
				t.Errorf("%s/%T: %v allocations per Encode", c.name, w, n)
			}
		}
	}
}

// encAllocConv returns a call of the 4:2:0 conversion kernel as Encode
// makes it.
func encAllocConv(nrgba bool) func() {
	const n = 200
	rgba, out := make([]byte, 8*n), make([]byte, 3*n)
	return func() {
		f := ycc.RGBAToYCbCr420
		if nrgba {
			f = ycc.NRGBAToYCbCr420
		}
		f(rgba[:4*n], rgba[4*n:], out[:n], out[n:2*n], out[2*n:5*n/2], out[5*n/2:], n)
	}
}

// Pixel paths.

// TestEncRows checks every encRow kind (and encStripAt for encRowAt)
// against At(x, y).RGBA()>>8, on random pixels (all values of the 1- and
// 2-byte formats).
func TestEncRows(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	const w, h = 256, 257
	r := image.Rect(-3, 2, w-3, h+2)
	pals := []color.Palette{palette.Plan9, palette.WebSafe, palette.Plan9[:17], {color.Gray16{0x1234}, color.NRGBA{1, 2, 3, 4}, color.CMYK{9, 8, 7, 6}, color.NYCbCrA{color.YCbCr{1, 2, 3}, 4}, color.Alpha{5}, color.Alpha16{0xfedc}, color.Gray{77}, color.YCbCr{200, 10, 250}, color.RGBA64{1, 2, 3, 4}, color.NRGBA64{0xffff, 0x8000, 1, 0x7fff}}}
	var ims []image.Image
	for _, k := range []int{1, 2, 4, 8} {
		var m image.Image
		var pix []byte
		switch k {
		case 1:
			a := image.NewAlpha(r)
			m, pix = a, a.Pix
		case 2:
			g := image.NewGray16(r)
			a := image.NewAlpha16(r)
			rng.Read(a.Pix)
			ims = append(ims, a)
			m, pix = g, g.Pix
		case 4:
			c := image.NewCMYK(r)
			rng.Read(c.Pix)
			ims = append(ims, c)
			rgba := image.NewRGBA(r)
			m, pix = rgba, rgba.Pix
		case 8:
			a := image.NewRGBA64(r)
			rng.Read(a.Pix)
			ims = append(ims, a)
			n := image.NewNRGBA64(r)
			m, pix = n, n.Pix
		}
		rng.Read(pix)
		// Every 1- and 2-byte value appears in the first rows.
		for i := 0; i < 65536 && 2*i+1 < len(pix); i++ {
			if k == 2 {
				pix[2*i], pix[2*i+1] = byte(i>>8), byte(i)
			} else if k == 1 && i < 256 {
				pix[i] = byte(i)
			}
		}
		ims = append(ims, m)
	}
	for _, p := range pals {
		m := image.NewPaletted(r, p)
		for i := range m.Pix {
			m.Pix[i] = uint8(rng.Intn(len(p)))
		}
		ims = append(ims, m)
	}
	for _, ratio := range encRatios {
		m := image.NewNYCbCrA(r, ratio)
		rng.Read(m.Y)
		rng.Read(m.Cb)
		rng.Read(m.Cr)
		rng.Read(m.A)
		ims = append(ims, m, wrapped{m})
	}
	for _, m := range ims {
		e := new(encoder)
		e.encStart(io.Discard)
		b := m.Bounds()
		e.m, e.rect, e.width, e.height = m, b, b.Dx(), b.Dy()
		e.encSetup(m)
		if p, ok := m.(*image.Paletted); !e.pure && e.rowKind != encRowAt && (!ok || len(p.Palette) >= 256) {
			t.Errorf("%T: not pure", m)
		}
		if e.kind != encSrcRows && e.kind != encSrcGray {
			continue
		}
		e.row[0] = make([]byte, 4*e.width)
		e.pw = e.width + 5
		gray := make([]byte, e.pw)
		for y := b.Min.Y; y < b.Max.Y; y++ {
			var got []byte
			if e.rowKind == encRowAt {
				if (y-b.Min.Y)%16 == 0 {
					e.encStripAt(y)
				}
				got = e.strip[(y-e.stripY)*4*e.width:]
			} else {
				got = e.encRow(0, y)
			}
			if e.kind == encSrcGray {
				e.encGrayRow(y, gray)
			}
			for x := 0; x < e.width; x++ {
				r, g, bb, _ := m.At(b.Min.X+x, y).RGBA()
				if got[4*x] != uint8(r>>8) || got[4*x+1] != uint8(g>>8) || got[4*x+2] != uint8(bb>>8) {
					t.Fatalf("%T (rowKind %d): pixel (%d, %d) = % x, want %02x %02x %02x", m, e.rowKind, b.Min.X+x, y, got[4*x:4*x+3], r>>8, g>>8, bb>>8)
				}
				if e.kind == encSrcGray {
					yy, cb, cr := color.RGBToYCbCr(uint8(r>>8), uint8(g>>8), uint8(bb>>8))
					if gray[x] != yy || cb != 128 || cr != 128 {
						t.Fatalf("%T: pixel (%d, %d): gray %d, want %d %d %d", m, b.Min.X+x, y, gray[x], yy, cb, cr)
					}
				}
			}
			for x := e.width; x < e.pw && e.kind == encSrcGray; x++ {
				if gray[x] != gray[e.width-1] {
					t.Fatalf("%T: padding not replicated", m)
				}
			}
		}
	}
}

// TestEncGrayIdentity checks the identity encSrcGray relies on.
func TestEncGrayIdentity(t *testing.T) {
	for v := 0; v < 256; v++ {
		if y, cb, cr := color.RGBToYCbCr(uint8(v), uint8(v), uint8(v)); y != uint8(v) || cb != 128 || cr != 128 {
			t.Errorf("RGBToYCbCr(%d, %d, %d) = %d, %d, %d", v, v, v, y, cb, cr)
		}
	}
}

func TestEncChromaOffsets(t *testing.T) {
	for _, ratio := range encRatios {
		for minX := -9; minX <= 9; minX++ {
			for minY := -9; minY <= 9; minY += 3 {
				m := &image.YCbCr{SubsampleRatio: ratio, CStride: 1000, Rect: image.Rect(minX, minY, minX+20, minY+20)}
				for y := minY; y < minY+20; y++ {
					for x := minX; x < minX+20; x++ {
						got := (encCY(ratio, y)-encCY(ratio, minY))*m.CStride + encCX(ratio, x) - encCX(ratio, minX)
						if want := m.COffset(x, y); got != want {
							t.Fatalf("%v, Min (%d, %d): (%d, %d): %d, want %d", ratio, minX, minY, x, y, got, want)
						}
					}
				}
			}
		}
	}
}

func TestEncRowsOK(t *testing.T) {
	for pixLen := 0; pixLen < 30; pixLen++ {
		for stride := -6; stride < 30; stride++ {
			for rowBytes := 1; rowBytes < 9; rowBytes++ {
				for rows := 1; rows < 6; rows++ {
					want := true
					for r := 0; r < rows; r++ {
						if o := r * stride; o < 0 || o+rowBytes > pixLen {
							want = false
						}
					}
					if got := encRowsOK(pixLen, stride, rowBytes, rows); got != want {
						t.Fatalf("encRowsOK(%d, %d, %d, %d) = %v", pixLen, stride, rowBytes, rows, got)
					}
				}
			}
		}
	}
	const maxInt = int(^uint(0) >> 1)
	for _, stride := range []int{maxInt, maxInt/2 + 1, maxInt / 3, -maxInt - 1, -maxInt / 2} {
		if encRowsOK(1<<20, stride, 4, 3) {
			t.Errorf("encRowsOK(1<<20, %d, 4, 3) = true", stride)
		}
	}
}

func TestEncAvg(t *testing.T) {
	a, b := make([]byte, 65536+40), make([]byte, 65536+40)
	for i := 0; i < 65536; i++ {
		a[i], b[i] = byte(i), byte(i>>8)
	}
	for n := 0; n <= 40; n++ {
		for off := 0; off < 3; off++ {
			dst := make([]byte, 65536+n)
			encAvg(dst[off:], a[off:off+len(dst)-off], b[off:off+len(dst)-off])
			for i := off; i < len(dst); i++ {
				if want := uint8((uint(a[i]) + uint(b[i]) + 1) >> 1); dst[i] != want {
					t.Fatalf("n=%d off=%d: avg(%d, %d) = %d, want %d", n, off, a[i], b[i], dst[i], want)
				}
			}
		}
	}
}

// Malformed images: image/jpeg panics on some, and must agree with us on
// which.

func TestEncMalformed(t *testing.T) {
	r := image.Rect(1, 2, 18, 11) // 17x9
	short := func(pix []byte, n int) []byte { return pix[:len(pix)-n] }
	shortCap := func(pix []byte, n int) []byte { return pix[: len(pix)-n : len(pix)] }
	var ims = map[string]image.Image{}
	rng := rand.New(rand.NewSource(7))
	for _, n := range []int{1, 2, 3, 4, 5} {
		mk := func(name string, m image.Image) { ims[fmt.Sprintf("%s-%d", name, n)] = m }
		rgba := image.NewRGBA(r)
		rng.Read(rgba.Pix)
		rgba.Pix = short(rgba.Pix, n)
		mk("RGBA", rgba)
		nrgba := image.NewNRGBA(r)
		rng.Read(nrgba.Pix)
		nrgba.Pix = short(nrgba.Pix, n)
		mk("NRGBA", nrgba)
		nrgbaCap := image.NewNRGBA(r)
		rng.Read(nrgbaCap.Pix)
		nrgbaCap.Pix = shortCap(nrgbaCap.Pix, n) // NRGBAAt slices up to cap
		mk("NRGBAcap", nrgbaCap)
		rgba64 := image.NewRGBA64(r)
		rgba64.Pix = shortCap(rgba64.Pix, n)
		mk("RGBA64cap", rgba64)
		nrgba64 := image.NewNRGBA64(r)
		nrgba64.Pix = short(nrgba64.Pix, n)
		mk("NRGBA64", nrgba64)
		cmyk := image.NewCMYK(r)
		cmyk.Pix = shortCap(cmyk.Pix, n)
		mk("CMYKcap", cmyk)
		g16 := image.NewGray16(r)
		g16.Pix = short(g16.Pix, n)
		mk("Gray16", g16)
		a16 := image.NewAlpha16(r)
		a16.Pix = shortCap(a16.Pix, n)
		mk("Alpha16cap", a16)
		a := image.NewAlpha(r)
		a.Pix = short(a.Pix, n)
		mk("Alpha", a)
		g := image.NewGray(r)
		g.Pix = short(g.Pix, n)
		mk("Gray", g)
		p := image.NewPaletted(r, palette.Plan9)
		p.Pix = short(p.Pix, n)
		mk("Paletted", p)
		for _, ratio := range encRatios {
			y := encRandYCbCr(rng, r, ratio)
			y.Y = short(y.Y, n)
			mk(fmt.Sprintf("YCbCr%v-Y", ratio), y)
			c := encRandYCbCr(rng, r, ratio)
			c.Cr = short(c.Cr, n)
			mk(fmt.Sprintf("YCbCr%v-Cr", ratio), c)
			ya := image.NewNYCbCrA(r, ratio)
			ya.A = short(ya.A, n)
			mk(fmt.Sprintf("NYCbCrA%v-A", ratio), ya)
		}
	}
	// Negative and zero strides.
	for _, stride := range []int{-4, 0} {
		m := image.NewRGBA(r)
		rng.Read(m.Pix)
		m.Stride = stride
		ims[fmt.Sprintf("RGBA-stride%d", stride)] = m
		g := image.NewGray(r)
		g.Stride = stride
		ims[fmt.Sprintf("Gray-stride%d", stride)] = g
		y := encRandYCbCr(rng, r, image.YCbCrSubsampleRatio420)
		y.CStride = stride
		ims[fmt.Sprintf("YCbCr420-cstride%d", stride)] = y
	}
	// Palettes.
	for name, pal := range map[string]color.Palette{"empty": {}, "short": palette.Plan9[:3], "nil": {color.RGBA{1, 2, 3, 4}, nil}, "custom": {encCustomColor{}, color.Gray{1}}} {
		m := image.NewPaletted(r, pal)
		for i := range m.Pix {
			m.Pix[i] = uint8(i % 2)
		}
		ims["Paletted-"+name] = m
		m2 := image.NewPaletted(r, pal)
		ims["Paletted0-"+name] = m2
	}
	// NewYCbCr allocates too few chroma samples for some negative
	// rectangles.
	for _, ratio := range encRatios {
		for _, nr := range []image.Rectangle{image.Rect(-3, -3, 2, 2), image.Rect(-5, -1, -2, 4), image.Rect(-7, -7, -1, -1)} {
			ims[fmt.Sprintf("YCbCr%v-%v", ratio, nr)] = encRandYCbCr(rng, nr, ratio)
		}
	}
	// Empty and inverted rectangles: no pixel is read.
	for _, er := range []image.Rectangle{{image.Pt(5, 5), image.Pt(0, 0)}, {image.Pt(0, 5), image.Pt(7, 0)}, {image.Pt(5, 0), image.Pt(0, 7)}, image.Rect(3, 3, 3, 9), image.Rect(3, 3, 9, 3)} {
		ims[fmt.Sprintf("RGBA-%v", er)] = &image.RGBA{Rect: er}
		ims[fmt.Sprintf("Gray-%v", er)] = &image.Gray{Rect: er}
		ims[fmt.Sprintf("YCbCr-%v", er)] = &image.YCbCr{Rect: er}
		ims[fmt.Sprintf("At-%v", er)] = hugeImage{er}
	}
	for name, m := range ims {
		for _, q := range []*Options{nil, encQ(100)} {
			compareEncode(t, name, m, q, false)
			compareEncode(t, name+"/bufio", m, q, true)
			// The panic values too (encode_regress_test.go).
			encStrictCompare(t, name+"/strict", encStatic(m), q)
		}
	}
}

type encCustomColor struct{}

func (encCustomColor) RGBA() (r, g, b, a uint32) { return 0x1234, 0x5678, 0x9abc, 0xffff }

// encLayoutImage builds an image of one of many types with the given
// geometry and pixels, optionally with a larger stride or a shortened
// buffer (fuzzing).
func encLayoutImage(kind uint8, r image.Rectangle, extra int, cut int, pix []byte) image.Image {
	fill := func(b []byte) {
		for i := range b {
			b[i] = pix[i%len(pix)] ^ byte(i*7)
		}
	}
	cutf := func(b []byte) []byte { return b[:max(0, len(b)-cut)] }
	switch kind % 9 {
	case 0, 1, 2, 3, 4, 5, 6:
		ratio := encRatios[int(kind%9)%len(encRatios)]
		big := r
		big.Max.X += extra
		m := image.NewYCbCr(big, ratio)
		fill(m.Y)
		fill(m.Cb)
		fill(m.Cr)
		m.Cr = cutf(m.Cr)
		return m.SubImage(r)
	case 7:
		m := image.NewRGBA(r)
		fill(m.Pix)
		m.Pix = cutf(m.Pix)
		return m
	default:
		m := image.NewNRGBA(r)
		fill(m.Pix)
		m.Pix = cutf(m.Pix)
		return m
	}
}

func FuzzEncLayout(f *testing.F) {
	f.Add(uint8(2), int8(-3), int8(5), uint8(17), uint8(9), uint8(3), uint8(0), 75, []byte("seed"))
	f.Add(uint8(0), int8(1), int8(1), uint8(33), uint8(16), uint8(0), uint8(1), 100, []byte{0xff, 0, 0x80})
	f.Fuzz(func(t *testing.T, kind uint8, x0, y0 int8, w, h, extra, cut uint8, q int, pix []byte) {
		if len(pix) == 0 {
			pix = []byte{0}
		}
		r := image.Rect(int(x0), int(y0), int(x0)+int(w%70), int(y0)+int(h%70))
		var m image.Image
		func() {
			defer func() { recover() }() // NewYCbCr may panic on some rectangles
			m = encLayoutImage(kind, r, int(extra%9), int(cut%4), pix)
		}()
		if m == nil {
			return
		}
		compareEncode(t, fmt.Sprintf("%d/%v/+%d/-%d/q%d", kind, r, extra%9, cut%4, q), m, &Options{Quality: q}, false)
	})
}

// Benchmarks of the encoder's own stages.

// BenchmarkEncEntropy codes real quantized blocks: ours against image/jpeg's
// emit.
func BenchmarkEncEntropy(b *testing.B) {
	src, err := std.Decode(bytes.NewReader(readFile(b, "../testdata/photo-960x1280.jpg")))
	if err != nil {
		b.Fatal(err)
	}
	yimg := src.(*image.YCbCr)
	for _, q := range []int{75, 100} {
		var qt [blockSize]byte
		scale := 200 - 2*q
		for i := range qt {
			qt[i] = uint8(max(1, min(255, (int(unscaledQuant[0][i])*scale+50)/100)))
		}
		e := new(encoder)
		e.encStart(io.Discard)
		e.qt[0].Init(&qt)
		var blocks [][blockSize]int16
		var nzs []uint64
		for y := 0; y+8 <= 256; y += 8 {
			for x := 0; x+8 <= 960; x += 8 {
				var c [blockSize]int16
				nz := dct.FDCTQuant(yimg.Y[y*yimg.YStride+x:], yimg.YStride, &e.qt[0], &c)
				blocks = append(blocks, c)
				nzs = append(nzs, nz)
			}
		}
		b.Run(fmt.Sprintf("q=%d/impl=turbo", q), func(b *testing.B) {
			b.SetBytes(int64(64 * len(blocks)))
			for b.Loop() {
				for i := 0; i+6 <= len(blocks); i += 6 {
					for k := 0; k < 6; k++ {
						e.coefs[k], e.nzs[k], e.comps[k] = blocks[i+k], nzs[i+k], 0
					}
					e.nb = 6
					e.encCode()
					if cap(e.out)-len(e.out) < encOutReserve {
						e.out = e.out[:0]
					}
				}
			}
		})
		b.Run(fmt.Sprintf("q=%d/impl=std", q), func(b *testing.B) {
			b.SetBytes(int64(64 * len(blocks)))
			ref := encNewRefCoder()
			var prev int32
			for b.Loop() {
				for i := range blocks {
					prev = ref.writeCoefs(&blocks[i], 0, prev)
					ref.out = ref.out[:0]
				}
			}
		})
	}
}

// BenchmarkEncSmall measures the fixed cost of an Encode call.
func BenchmarkEncSmall(b *testing.B) {
	for _, sz := range []int{1, 16, 64} {
		m := imageMakers[0].make(image.Rect(0, 0, sz, sz), fills(1)[0].f)
		for _, impl := range benchImpls {
			b.Run(fmt.Sprintf("size=%d/impl=%s", sz, impl.name), func(b *testing.B) {
				b.ReportAllocs()
				var buf bytes.Buffer
				for b.Loop() {
					buf.Reset()
					if err := impl.encode(&buf, m, 75); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// BenchmarkEncTypes encodes every image type of imageMakers (a 512x512
// gradient), which shows the cost of each pixel path.
func BenchmarkEncTypes(b *testing.B) {
	fill := fills(1)[2].f
	for _, mk := range imageMakers {
		m := mk.make(image.Rect(0, 0, 512, 512), fill)
		for _, impl := range benchImpls {
			b.Run(fmt.Sprintf("img=%s/impl=%s", mk.name, impl.name), func(b *testing.B) {
				b.SetBytes(512 * 512)
				b.ReportAllocs()
				var buf bytes.Buffer
				for b.Loop() {
					buf.Reset()
					if err := impl.encode(&buf, m, 75); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
