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

// Differential tests against the standard library: every decode must return
// an image identical to image/jpeg's (same type, bounds, strides and every
// byte of the pixel buffers, padding included) or an error with the same
// text and kind, and every encode must produce the same bytes.

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
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"runtime/debug"
	"sort"
	"strings"
	"testing"
	"testing/iotest"
)

// API compatibility with image/jpeg (compile-time).
var (
	_ func(io.Reader) (image.Image, error)         = Decode
	_ func(io.Reader) (image.Config, error)        = DecodeConfig
	_ func(io.Writer, image.Image, *Options) error = Encode
	_ int                                          = DefaultQuality
	_ error                                        = FormatError("")
	_ error                                        = UnsupportedError("")
	_ Reader                                       = (*bufio.Reader)(nil)
	_                                              = Options{Quality: 1}
)

// corpusFiles lists the JPEG files of the differential tests.
func corpusFiles(tb testing.TB) []string {
	tb.Helper()
	var files []string
	for _, pat := range []string{"../testdata/*.jpeg", "../testdata/*.jpg", "../testdata/corpus/*.jpeg", "../testdata/bench/*.jpg"} {
		m, err := filepath.Glob(pat)
		if err != nil {
			tb.Fatal(err)
		}
		files = append(files, m...)
	}
	sort.Strings(files)
	if len(files) < 100 {
		tb.Fatalf("only %d corpus files found; run go run ./_tools/gencorpus", len(files))
	}
	return files
}

func readFile(tb testing.TB, name string) []byte {
	tb.Helper()
	b, err := os.ReadFile(name)
	if err != nil {
		tb.Fatal(err)
	}
	return b
}

// errKind classifies an error so that ours and image/jpeg's can be
// compared: the same text is required, and the same kind.
func errKind(err error) string {
	switch err.(type) {
	case nil:
		return "nil"
	case FormatError, std.FormatError:
		return "FormatError"
	case UnsupportedError, std.UnsupportedError:
		return "UnsupportedError"
	}
	switch err {
	case io.ErrUnexpectedEOF:
		return "io.ErrUnexpectedEOF"
	case io.EOF:
		return "io.EOF"
	case errTestReader:
		return "errTestReader"
	}
	return fmt.Sprintf("%T", err)
}

func sameError(a, b error) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return errKind(a) == errKind(b) && a.Error() == b.Error()
}

type decodeResult struct {
	img   image.Image
	cfg   image.Config
	err   error
	panic any
}

func (r decodeResult) String() string {
	if r.panic != nil {
		return fmt.Sprintf("panic(%v)", r.panic)
	}
	if r.err != nil {
		return fmt.Sprintf("error %s %q", errKind(r.err), r.err)
	}
	if r.img != nil {
		return fmt.Sprintf("%T %v", r.img, r.img.Bounds())
	}
	return fmt.Sprintf("config %+v", r.cfg)
}

func catchPanic(r *decodeResult) {
	if p := recover(); p != nil {
		r.panic = p
	}
}

func decodeOurs(r io.Reader) (res decodeResult) {
	defer catchPanic(&res)
	res.img, res.err = Decode(r)
	return
}

func decodeStd(r io.Reader) (res decodeResult) {
	defer catchPanic(&res)
	res.img, res.err = std.Decode(r)
	return
}

func configOurs(r io.Reader) (res decodeResult) {
	defer catchPanic(&res)
	res.cfg, res.err = DecodeConfig(r)
	return
}

func configStd(r io.Reader) (res decodeResult) {
	defer catchPanic(&res)
	res.cfg, res.err = std.DecodeConfig(r)
	return
}

// diffImages describes the first difference between two decoded images, or
// returns "" if they are identical (reflect.DeepEqual: same concrete type,
// rectangle, strides, subsample ratio and buffer contents and lengths).
func diffImages(got, want image.Image) string {
	if reflect.DeepEqual(got, want) {
		return ""
	}
	if reflect.TypeOf(got) != reflect.TypeOf(want) {
		return fmt.Sprintf("type %T, want %T", got, want)
	}
	if got.Bounds() != want.Bounds() {
		return fmt.Sprintf("bounds %v, want %v", got.Bounds(), want.Bounds())
	}
	planes := func(m image.Image) (names []string, bufs [][]byte, desc string) {
		switch m := m.(type) {
		case *image.YCbCr:
			return []string{"Y", "Cb", "Cr"}, [][]byte{m.Y, m.Cb, m.Cr}, fmt.Sprintf("YStride=%d CStride=%d ratio=%v", m.YStride, m.CStride, m.SubsampleRatio)
		case *image.Gray:
			return []string{"Pix"}, [][]byte{m.Pix}, fmt.Sprintf("Stride=%d", m.Stride)
		case *image.RGBA:
			return []string{"Pix"}, [][]byte{m.Pix}, fmt.Sprintf("Stride=%d", m.Stride)
		case *image.CMYK:
			return []string{"Pix"}, [][]byte{m.Pix}, fmt.Sprintf("Stride=%d", m.Stride)
		}
		return nil, nil, fmt.Sprintf("%#v", m)
	}
	gn, gb, gd := planes(got)
	_, wb, wd := planes(want)
	if gd != wd {
		return fmt.Sprintf("layout %s, want %s", gd, wd)
	}
	for i := range gb {
		if len(gb[i]) != len(wb[i]) {
			return fmt.Sprintf("len(%s) = %d, want %d", gn[i], len(gb[i]), len(wb[i]))
		}
		for j := range gb[i] {
			if gb[i][j] != wb[i][j] {
				return fmt.Sprintf("%s[%d] = %d, want %d (%s)", gn[i], j, gb[i][j], wb[i][j], gd)
			}
		}
	}
	return "images differ"
}

// compareDecode decodes data with both packages (through readers made by
// mk) and reports any difference.
func compareDecode(t testing.TB, name string, data []byte, mk func([]byte) io.Reader) {
	t.Helper()
	got := decodeOurs(mk(data))
	want := decodeStd(mk(data))
	if got.panic != nil || want.panic != nil {
		if got.panic == nil || want.panic == nil {
			t.Errorf("%s: Decode: got %v, want %v", name, got, want)
		}
		return
	}
	if !sameError(got.err, want.err) {
		t.Errorf("%s: Decode: got %v, want %v", name, got, want)
		return
	}
	if want.err == nil {
		if d := diffImages(got.img, want.img); d != "" {
			t.Errorf("%s: Decode: %s", name, d)
		}
	}
	gotC := configOurs(mk(data))
	wantC := configStd(mk(data))
	if (gotC.panic == nil) != (wantC.panic == nil) || !sameError(gotC.err, wantC.err) || gotC.cfg != wantC.cfg {
		t.Errorf("%s: DecodeConfig: got %v, want %v", name, gotC, wantC)
	}
}

var errTestReader = errors.New("test reader error")

// errAfterReader returns data[:n] and then errTestReader.
type errAfterReader struct {
	data []byte
	n    int
}

func (r *errAfterReader) Read(p []byte) (int, error) {
	if r.n <= 0 {
		return 0, errTestReader
	}
	k := min(len(p), r.n, len(r.data))
	if k == 0 {
		return 0, errTestReader
	}
	copy(p, r.data[:k])
	r.data, r.n = r.data[k:], r.n-k
	return k, nil
}

var readers = []struct {
	name string
	mk   func([]byte) io.Reader
}{
	{"bytes", func(b []byte) io.Reader { return bytes.NewReader(b) }},
	{"onebyte", func(b []byte) io.Reader { return iotest.OneByteReader(bytes.NewReader(b)) }},
	{"half", func(b []byte) io.Reader { return iotest.HalfReader(bytes.NewReader(b)) }},
	{"dataerr", func(b []byte) io.Reader { return iotest.DataErrReader(bytes.NewReader(b)) }},
	{"bufio", func(b []byte) io.Reader { return bufio.NewReaderSize(bytes.NewReader(b), 16) }},
}

func TestCompatDecodeCorpus(t *testing.T) {
	for _, f := range corpusFiles(t) {
		data := readFile(t, f)
		for _, r := range readers {
			if r.name != "bytes" && testing.Short() && len(data) > 64<<10 {
				continue
			}
			compareDecode(t, filepath.Base(f)+"/"+r.name, data, r.mk)
		}
	}
}

// TestCompatDecodeRegistered checks that image.Decode finds a decoder for
// JPEG data (ours, or image/jpeg's when both are linked in) and that the
// result is the same.
func TestCompatDecodeRegistered(t *testing.T) {
	data := readFile(t, "../testdata/video-001.jpeg")
	m, name, err := image.Decode(bytes.NewReader(data))
	if err != nil || name != "jpeg" {
		t.Fatalf("image.Decode: %v, %q", err, name)
	}
	want, _ := std.Decode(bytes.NewReader(data))
	if d := diffImages(m, want); d != "" {
		t.Fatal(d)
	}
}

// entropyStart returns the offset just past the first SOS segment header,
// or len(data)/2 if there is none.
func entropyStart(data []byte) int {
	for i := 2; i+3 < len(data); i++ {
		if data[i] == 0xff && data[i+1] == sosMarker {
			n := int(data[i+2])<<8 | int(data[i+3])
			if i+2+n < len(data) {
				return i + 2 + n
			}
		}
	}
	return len(data) / 2
}

func TestCompatDecodeTruncated(t *testing.T) {
	points := 48
	if testing.Short() {
		points = 8
	}
	for _, f := range corpusFiles(t) {
		data := readFile(t, f)
		seen := map[int]bool{}
		try := func(n int) {
			if n < 0 || n > len(data) || seen[n] {
				return
			}
			seen[n] = true
			compareDecode(t, fmt.Sprintf("%s[:%d]", filepath.Base(f), n), data[:n], readers[0].mk)
		}
		for i := 0; i <= points; i++ {
			try(i * len(data) / points)
		}
		for i := 1; i <= 16; i++ {
			try(len(data) - i)
		}
		es := entropyStart(data)
		for i := -4; i <= 4; i++ {
			try(es + i)
		}
	}
}

// mutate applies one random edit, mostly inside the entropy-coded data.
func mutate(rng *rand.Rand, data []byte) []byte {
	b := append([]byte(nil), data...)
	es := entropyStart(b)
	pos := func() int {
		if es < len(b) && rng.Intn(4) != 0 {
			return es + rng.Intn(len(b)-es)
		}
		return rng.Intn(len(b))
	}
	switch rng.Intn(9) {
	case 0, 1, 2: // flip a bit
		p := pos()
		b[p] ^= 1 << rng.Intn(8)
	case 3: // random byte
		b[pos()] = byte(rng.Intn(256))
	case 4: // 0xff
		b[pos()] = 0xff
	case 5: // insert a marker
		p := pos()
		m := []byte{0xff, byte(0xd0 + rng.Intn(8))}
		if rng.Intn(3) == 0 {
			m[1] = []byte{0xd9, 0xff, 0x00, 0xc4, 0xda, 0xdd}[rng.Intn(6)]
		}
		b = append(b[:p], append(m, b[p:]...)...)
	case 6: // delete bytes
		p := pos()
		n := 1 + rng.Intn(8)
		if p+n > len(b) {
			n = len(b) - p
		}
		b = append(b[:p], b[p+n:]...)
	case 7: // duplicate a run of bytes
		p := pos()
		n := min(1+rng.Intn(64), len(b)-p)
		run := append([]byte(nil), b[p:p+n]...)
		b = append(b[:p], append(run, b[p:]...)...)
	case 8: // header byte
		if es > 2 {
			p := 2 + rng.Intn(es-2)
			b[p] = byte(rng.Intn(256))
		}
	}
	return b
}

func TestCompatDecodeMutated(t *testing.T) {
	per := 24
	if testing.Short() {
		per = 3
	}
	rng := rand.New(rand.NewSource(1))
	for _, f := range corpusFiles(t) {
		data := readFile(t, f)
		if len(data) < 4 {
			continue
		}
		for i := 0; i < per; i++ {
			m := mutate(rng, data)
			if rng.Intn(3) == 0 {
				m = mutate(rng, m)
			}
			compareDecode(t, fmt.Sprintf("%s/mut%d", filepath.Base(f), i), m, readers[0].mk)
		}
	}
}

func TestCompatDecodeReaderError(t *testing.T) {
	step := 1
	for _, f := range corpusFiles(t) {
		data := readFile(t, f)
		if len(data) > 32<<10 {
			continue
		}
		step = max(1, len(data)/24)
		if testing.Short() {
			step = max(1, len(data)/4)
		}
		for n := 0; n < len(data); n += step {
			compareDecode(t, fmt.Sprintf("%s/errafter%d", filepath.Base(f), n), data, func(b []byte) io.Reader {
				return &errAfterReader{data: b, n: n}
			})
		}
	}
}

// Encoder inputs.

type imageMaker struct {
	name string
	make func(r image.Rectangle, fill func(x, y int) color.NRGBA64) image.Image
}

func setAll(m interface {
	Set(x, y int, c color.Color)
	Bounds() image.Rectangle
}, fill func(x, y int) color.NRGBA64) {
	b := m.Bounds()
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			m.Set(x, y, fill(x, y))
		}
	}
}

func ycbcrImage(r image.Rectangle, ratio image.YCbCrSubsampleRatio, fill func(x, y int) color.NRGBA64) *image.YCbCr {
	m := image.NewYCbCr(r, ratio)
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			c := fill(x, y)
			yy, cb, cr := color.RGBToYCbCr(uint8(c.R>>8), uint8(c.G>>8), uint8(c.B>>8))
			m.Y[m.YOffset(x, y)] = yy
			// NewYCbCr allocates too few chroma samples for some
			// rectangles with negative coordinates (truncating division);
			// image/jpeg then panics on them, and so must we.
			if ci := m.COffset(x, y); ci >= 0 && ci < len(m.Cb) {
				m.Cb[ci] = cb
				m.Cr[ci] = cr
			}
		}
	}
	return m
}

// wrapped hides the concrete type, forcing the generic At path.
type wrapped struct{ image.Image }

var imageMakers = []imageMaker{
	{"RGBA", func(r image.Rectangle, f func(x, y int) color.NRGBA64) image.Image {
		m := image.NewRGBA(r)
		setAll(m, f)
		return m
	}},
	{"RGBAraw", func(r image.Rectangle, f func(x, y int) color.NRGBA64) image.Image {
		// Not premultiplied: channels above alpha, as some code produces.
		m := image.NewRGBA(r)
		for y := r.Min.Y; y < r.Max.Y; y++ {
			for x := r.Min.X; x < r.Max.X; x++ {
				c := f(x, y)
				i := m.PixOffset(x, y)
				m.Pix[i], m.Pix[i+1], m.Pix[i+2], m.Pix[i+3] = uint8(c.R>>8), uint8(c.G>>8), uint8(c.B>>8), uint8(c.A>>8)
			}
		}
		return m
	}},
	{"NRGBA", func(r image.Rectangle, f func(x, y int) color.NRGBA64) image.Image {
		m := image.NewNRGBA(r)
		setAll(m, f)
		return m
	}},
	{"RGBA64", func(r image.Rectangle, f func(x, y int) color.NRGBA64) image.Image {
		m := image.NewRGBA64(r)
		setAll(m, f)
		return m
	}},
	{"NRGBA64", func(r image.Rectangle, f func(x, y int) color.NRGBA64) image.Image {
		m := image.NewNRGBA64(r)
		setAll(m, f)
		return m
	}},
	{"Gray", func(r image.Rectangle, f func(x, y int) color.NRGBA64) image.Image {
		m := image.NewGray(r)
		setAll(m, f)
		return m
	}},
	{"Gray16", func(r image.Rectangle, f func(x, y int) color.NRGBA64) image.Image {
		m := image.NewGray16(r)
		setAll(m, f)
		return m
	}},
	{"Alpha", func(r image.Rectangle, f func(x, y int) color.NRGBA64) image.Image {
		m := image.NewAlpha(r)
		setAll(m, f)
		return m
	}},
	{"CMYK", func(r image.Rectangle, f func(x, y int) color.NRGBA64) image.Image {
		m := image.NewCMYK(r)
		setAll(m, f)
		return m
	}},
	{"Paletted", func(r image.Rectangle, f func(x, y int) color.NRGBA64) image.Image {
		m := image.NewPaletted(r, palette.Plan9)
		setAll(m, f)
		return m
	}},
	{"YCbCr444", func(r image.Rectangle, f func(x, y int) color.NRGBA64) image.Image {
		return ycbcrImage(r, image.YCbCrSubsampleRatio444, f)
	}},
	{"YCbCr422", func(r image.Rectangle, f func(x, y int) color.NRGBA64) image.Image {
		return ycbcrImage(r, image.YCbCrSubsampleRatio422, f)
	}},
	{"YCbCr420", func(r image.Rectangle, f func(x, y int) color.NRGBA64) image.Image {
		return ycbcrImage(r, image.YCbCrSubsampleRatio420, f)
	}},
	{"YCbCr440", func(r image.Rectangle, f func(x, y int) color.NRGBA64) image.Image {
		return ycbcrImage(r, image.YCbCrSubsampleRatio440, f)
	}},
	{"YCbCr411", func(r image.Rectangle, f func(x, y int) color.NRGBA64) image.Image {
		return ycbcrImage(r, image.YCbCrSubsampleRatio411, f)
	}},
	{"YCbCr410", func(r image.Rectangle, f func(x, y int) color.NRGBA64) image.Image {
		return ycbcrImage(r, image.YCbCrSubsampleRatio410, f)
	}},
	{"NYCbCrA420", func(r image.Rectangle, f func(x, y int) color.NRGBA64) image.Image {
		m := image.NewNYCbCrA(r, image.YCbCrSubsampleRatio420)
		m.YCbCr = *ycbcrImage(r, image.YCbCrSubsampleRatio420, f)
		for y := r.Min.Y; y < r.Max.Y; y++ {
			for x := r.Min.X; x < r.Max.X; x++ {
				m.A[m.AOffset(x, y)] = uint8(f(x, y).A >> 8)
			}
		}
		return m
	}},
	{"Wrapped", func(r image.Rectangle, f func(x, y int) color.NRGBA64) image.Image {
		m := image.NewNRGBA(r)
		setAll(m, f)
		return wrapped{m}
	}},
}

// fills returns pixel generators: random noise (with and without alpha),
// smooth gradients and flat areas.
func fills(seed int64) []struct {
	name string
	f    func(x, y int) color.NRGBA64
} {
	rng := rand.New(rand.NewSource(seed))
	noise := make([]uint16, 4*4096)
	for i := range noise {
		noise[i] = uint16(rng.Intn(1 << 16))
	}
	at := func(x, y, c int) uint16 { return noise[(4*((x*31+y*17)&4095)+c)%len(noise)] }
	return []struct {
		name string
		f    func(x, y int) color.NRGBA64
	}{
		{"noise", func(x, y int) color.NRGBA64 {
			return color.NRGBA64{at(x, y, 0), at(x, y, 1), at(x, y, 2), 0xffff}
		}},
		{"noisealpha", func(x, y int) color.NRGBA64 {
			return color.NRGBA64{at(x, y, 0), at(x, y, 1), at(x, y, 2), at(x, y, 3)}
		}},
		{"gradient", func(x, y int) color.NRGBA64 {
			return color.NRGBA64{uint16(x * 2311), uint16(y * 1999), uint16((x + y) * 997), 0xffff}
		}},
		{"flat", func(x, y int) color.NRGBA64 {
			if (x/9+y/7)%2 == 0 {
				return color.NRGBA64{0xffff, 0xffff, 0xffff, 0xffff}
			}
			return color.NRGBA64{0, 0x8000, 0x1234, 0xffff}
		}},
	}
}

var encodeSizes = []image.Point{{0, 0}, {1, 1}, {1, 17}, {17, 1}, {7, 9}, {8, 8}, {15, 16}, {16, 16}, {17, 17}, {33, 47}, {100, 75}, {257, 129}}

var encodeOptions = []*Options{nil, {Quality: -5}, {Quality: 0}, {Quality: 1}, {Quality: 2}, {Quality: 10}, {Quality: 25}, {Quality: 49}, {Quality: 50}, {Quality: 51}, {Quality: 75}, {Quality: 90}, {Quality: 95}, {Quality: 99}, {Quality: 100}, {Quality: 101}, {Quality: 1000}}

func stdOptions(o *Options) *std.Options {
	if o == nil {
		return nil
	}
	return &std.Options{Quality: o.Quality}
}

type encodeResult struct {
	out   []byte
	err   error
	panic any
}

func encodeOurs(m image.Image, o *Options, viaBufio bool) (res encodeResult) {
	defer func() {
		if p := recover(); p != nil {
			res.panic = p
		}
	}()
	var buf bytes.Buffer
	if viaBufio {
		w := bufio.NewWriterSize(&buf, 64)
		res.err = Encode(w, m, o)
		// Encode must flush a writer that has a Flush method.
		if w.Buffered() != 0 {
			res.err = fmt.Errorf("%d bytes left unflushed", w.Buffered())
		}
	} else {
		res.err = Encode(&buf, m, o)
	}
	res.out = buf.Bytes()
	return
}

func encodeStd(m image.Image, o *Options) (res encodeResult) {
	defer func() {
		if p := recover(); p != nil {
			res.panic = p
		}
	}()
	var buf bytes.Buffer
	res.err = std.Encode(&buf, m, stdOptions(o))
	res.out = buf.Bytes()
	return
}

func compareEncode(t testing.TB, name string, m image.Image, o *Options, viaBufio bool) {
	t.Helper()
	got := encodeOurs(m, o, viaBufio)
	want := encodeStd(m, o)
	if (got.panic == nil) != (want.panic == nil) {
		t.Errorf("%s: panic %v, want %v", name, got.panic, want.panic)
		return
	}
	if (got.err == nil) != (want.err == nil) || (got.err != nil && got.err.Error() != want.err.Error()) {
		t.Errorf("%s: error %v, want %v", name, got.err, want.err)
		return
	}
	if !bytes.Equal(got.out, want.out) {
		i := 0
		for i < len(got.out) && i < len(want.out) && got.out[i] == want.out[i] {
			i++
		}
		t.Errorf("%s: output differs at byte %d (len %d, want %d)", name, i, len(got.out), len(want.out))
	}
}

func TestCompatEncode(t *testing.T) {
	sizes := encodeSizes
	opts := encodeOptions
	if testing.Short() {
		sizes = []image.Point{{1, 1}, {17, 17}, {33, 47}}
		opts = []*Options{nil, {Quality: 1}, {Quality: 100}}
	}
	for _, mk := range imageMakers {
		for _, fl := range fills(int64(len(mk.name))) {
			for _, sz := range sizes {
				for _, origin := range []image.Point{{0, 0}, {-3, 5}, {1, 1}} {
					r := image.Rectangle{origin, origin.Add(sz)}
					m := mk.make(r, fl.f)
					for _, o := range opts {
						q := "nil"
						if o != nil {
							q = fmt.Sprint(o.Quality)
						}
						name := fmt.Sprintf("%s/%s/%v/q%s", mk.name, fl.name, r, q)
						compareEncode(t, name, m, o, false)
					}
					compareEncode(t, fmt.Sprintf("%s/%s/%v/bufio", mk.name, fl.name, r), m, nil, true)
				}
			}
		}
	}
}

// TestCompatEncodeSubImage encodes sub-images with odd offsets, whose
// chroma samples (YCbCr) and pixel rows do not start at a block boundary.
func TestCompatEncodeSubImage(t *testing.T) {
	type subImager interface {
		image.Image
		SubImage(image.Rectangle) image.Image
	}
	fl := fills(7)[2].f
	for _, mk := range imageMakers {
		base := mk.make(image.Rect(0, 0, 61, 45), fl)
		s, ok := base.(subImager)
		if !ok {
			continue
		}
		for _, r := range []image.Rectangle{image.Rect(1, 1, 60, 44), image.Rect(3, 2, 20, 41), image.Rect(5, 7, 6, 8), image.Rect(17, 13, 61, 45), image.Rect(2, 3, 2, 3)} {
			for _, o := range []*Options{nil, {Quality: 95}} {
				compareEncode(t, fmt.Sprintf("%s/%v/%v", mk.name, r, o), s.SubImage(r), o, false)
			}
		}
	}
}

// TestCompatEncodeDecoded re-encodes every corpus picture image/jpeg can
// decode (all the image types and subsampling ratios its decoder returns).
func TestCompatEncodeDecoded(t *testing.T) {
	for i, f := range corpusFiles(t) {
		if testing.Short() && i%8 != 0 {
			continue
		}
		m, err := std.Decode(bytes.NewReader(readFile(t, f)))
		if err != nil {
			continue
		}
		for _, o := range []*Options{nil, {Quality: 90}} {
			compareEncode(t, fmt.Sprintf("%s/%v", filepath.Base(f), o), m, o, false)
		}
	}
}

// hugeImage has bounds too large to encode; its pixels must never be read.
type hugeImage struct{ r image.Rectangle }

func (m hugeImage) ColorModel() color.Model { return color.RGBAModel }
func (m hugeImage) Bounds() image.Rectangle { return m.r }
func (m hugeImage) At(x, y int) color.Color { panic("At called") }

func TestCompatEncodeTooLarge(t *testing.T) {
	for _, r := range []image.Rectangle{image.Rect(0, 0, 1<<16, 1), image.Rect(0, 0, 1, 1<<16), image.Rect(-5, 0, 1<<16-5, 1)} {
		compareEncode(t, fmt.Sprint(r), hugeImage{r}, nil, false)
	}
}

// errWriter fails after n bytes.
type errWriter struct{ n int }

func (w *errWriter) Write(p []byte) (int, error) {
	if len(p) > w.n {
		k := w.n
		w.n = 0
		return k, errTestReader
	}
	w.n -= len(p)
	return len(p), nil
}

func TestCompatEncodeWriterError(t *testing.T) {
	m := imageMakers[0].make(image.Rect(0, 0, 100, 75), fills(3)[0].f)
	var full bytes.Buffer
	if err := std.Encode(&full, m, nil); err != nil {
		t.Fatal(err)
	}
	for n := 0; n < full.Len(); n += 97 {
		err := Encode(&errWriter{n: n}, m, nil)
		if err != errTestReader {
			t.Errorf("writer failing after %d bytes: error %v, want %v", n, err, errTestReader)
		}
	}
}

func FuzzCompatDecode(f *testing.F) {
	for i, file := range corpusFiles(f) {
		if i%4 == 0 {
			if b := readFile(f, file); len(b) < 64<<10 {
				f.Add(b)
			}
		}
	}
	debug.SetMaxStack(64 << 20)
	f.Fuzz(func(t *testing.T, b []byte) {
		if cfg, err := std.DecodeConfig(bytes.NewReader(b)); err == nil && cfg.Width*cfg.Height > 4e6 {
			return
		}
		compareDecode(t, "fuzz", b, readers[0].mk)
	})
}

func FuzzCompatEncode(f *testing.F) {
	f.Add(uint8(0), uint8(5), uint8(7), 75, []byte("seed pixels"))
	f.Add(uint8(5), uint8(16), uint8(16), 100, bytes.Repeat([]byte{0xff, 0, 0x80}, 100))
	f.Add(uint8(12), uint8(33), uint8(2), 1, []byte{1, 2, 3, 4, 5, 6, 7, 8, 9})
	f.Fuzz(func(t *testing.T, kind, w, h uint8, q int, pix []byte) {
		if len(pix) == 0 {
			pix = []byte{0}
		}
		fill := func(x, y int) color.NRGBA64 {
			i := 4 * (y*int(w) + x)
			c := func(k int) uint16 { v := pix[(i+k)%len(pix)]; return uint16(v)<<8 | uint16(v^byte(x+y)) }
			return color.NRGBA64{c(0), c(1), c(2), c(3)}
		}
		mk := imageMakers[int(kind)%len(imageMakers)]
		r := image.Rect(int(kind>>4), int(kind>>5), int(kind>>4)+int(w%80), int(kind>>5)+int(h%80))
		m := mk.make(r, fill)
		compareEncode(t, fmt.Sprintf("%s/%v/q%d", mk.name, r, q), m, &Options{Quality: q}, false)
	})
}

func TestCompatCorpusPresent(t *testing.T) {
	files := corpusFiles(t)
	var kinds = map[string]bool{}
	for _, f := range files {
		base := filepath.Base(f)
		for _, k := range []string{"_prog", "_rst", "_gray", "_rgb", "_arith", "_p12", "_422", "_420", "_444", "_440", "_411", "_410", "221122"} {
			if strings.Contains(base, k) {
				kinds[k] = true
			}
		}
	}
	if len(kinds) < 13 {
		t.Errorf("corpus lacks some kinds: have %v", kinds)
	}
}
