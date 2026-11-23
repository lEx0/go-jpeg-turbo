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

// Decoding and encoding from many goroutines at once, and from inside the
// callbacks of another Decode or Encode (a Write, WriteByte, Read or At
// method). The encoder pools its state (encPool) and the decoder may come
// to pool its own (decoders, Huffman lookup tables): state shared by two
// calls that overlap, or not reset after a call that failed or panicked,
// shows here as a wrong result. Re-entrancy finds a non-reentrant cache
// deterministically; the concurrent tests find shared state through
// wrong results, and through the race detector under make race. From the
// verification of concurrent use.

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"image"
	"image/color"
	"io"
	"math/rand"
	"path/filepath"
	"sync"
	"testing"
)

// ccDec is a decoder input with image/jpeg's results.
type ccDec struct {
	name string
	data []byte
	want decodeResult
	cfg  decodeResult
}

// ccEnc is an encoder input with image/jpeg's output.
type ccEnc struct {
	name string
	m    image.Image
	o    *Options
	out  []byte
}

// ccInputs returns decoder inputs (every 13th corpus file, and generated
// progressive streams) and encoder inputs (every image type, several sizes
// and qualities) that image/jpeg encodes without error.
func ccInputs(t *testing.T) ([]ccDec, []ccEnc) {
	var decs []ccDec
	add := func(name string, data []byte) {
		decs = append(decs, ccDec{name, data, decodeStd(bytes.NewReader(data)), configStd(bytes.NewReader(data))})
	}
	for i, f := range corpusFiles(t) {
		if i%13 == 0 {
			add(filepath.Base(f), readFile(t, f))
		}
	}
	rng := rand.New(rand.NewSource(77))
	for i := 0; i < 24; i++ {
		add(fmt.Sprintf("progressive%d", i), pgRandomJPEG(rng, 100, false, false))
	}
	var encs []ccEnc
	for k, mk := range imageMakers {
		for _, sz := range []image.Point{{1, 1}, {17, 9}, {64, 48}, {131, 77}} {
			origin := image.Pt(k%3, k%2)
			m := mk.make(image.Rectangle{origin, origin.Add(sz)}, fills(int64(k))[k%4].f)
			for _, q := range []int{1, 50, 75, 100} {
				o := encQ(q)
				if r := encStd(m, o); r.err == nil && r.panic == nil {
					encs = append(encs, ccEnc{fmt.Sprintf("%s/%v/q%d", mk.name, sz, q), m, o, r.out})
				}
			}
		}
	}
	return decs, encs
}

// ccCheckDec decodes c (or decodes its config) from r and reports a
// difference from image/jpeg.
func ccCheckDec(c *ccDec, r io.Reader, config bool) string {
	if config {
		got := configOurs(r)
		if (got.panic == nil) != (c.cfg.panic == nil) || !sameError(got.err, c.cfg.err) || got.cfg != c.cfg.cfg {
			return fmt.Sprintf("%s: DecodeConfig: got %v, want %v", c.name, got, c.cfg)
		}
		return ""
	}
	got := decodeOurs(r)
	if (got.panic == nil) != (c.want.panic == nil) || !sameError(got.err, c.want.err) {
		return fmt.Sprintf("%s: Decode: got %v, want %v", c.name, got, c.want)
	}
	if c.want.err == nil && c.want.panic == nil {
		if d := diffImages(got.img, c.want.img); d != "" {
			return c.name + ": Decode: " + d
		}
	}
	return ""
}

// ccPlainWriter is an io.Writer without other methods.
type ccPlainWriter struct{ b bytes.Buffer }

func (w *ccPlainWriter) Write(p []byte) (int, error) { return w.b.Write(p) }

// ccCheckEnc encodes c to a writer of kind k (a bytes.Buffer, a small
// bufio.Writer or a plain io.Writer) and reports a difference from
// image/jpeg's output.
func ccCheckEnc(c *ccEnc, k int) (diff string) {
	defer func() {
		if p := recover(); p != nil {
			diff = fmt.Sprintf("%s (writer %d): panic %v", c.name, k%3, p)
		}
	}()
	var got []byte
	var err error
	switch k % 3 {
	case 0:
		var b bytes.Buffer
		err = Encode(&b, c.m, c.o)
		got = b.Bytes()
	case 1:
		var w ccPlainWriter
		bw := bufio.NewWriterSize(&w, 100+k%1000)
		err = Encode(bw, c.m, c.o)
		got = w.b.Bytes()
	case 2:
		var w ccPlainWriter
		err = Encode(&w, c.m, c.o)
		got = w.b.Bytes()
	}
	if err != nil || !bytes.Equal(got, c.out) {
		return fmt.Sprintf("%s (writer %d): error %v, output equal %v", c.name, k%3, err, bytes.Equal(got, c.out))
	}
	return ""
}

// ccPanicAt panics in At at one pixel.
type ccPanicAt struct {
	image.Image
	at image.Point
}

func (m ccPanicAt) At(x, y int) color.Color {
	if image.Pt(x, y) == m.at {
		panic("ccPanicAt")
	}
	return m.Image.At(x, y)
}

// ccPanicWriter panics once more than n bytes would have been written.
type ccPanicWriter struct{ n int }

func (w *ccPanicWriter) Write(p []byte) (int, error) {
	if len(p) > w.n {
		panic("ccPanicWriter")
	}
	w.n -= len(p)
	return len(p), nil
}

var errCCWriter = errors.New("concurrent test writer error")

// ccFailWriter fails once more than n bytes would have been written.
type ccFailWriter struct{ n int }

func (w *ccFailWriter) Write(p []byte) (int, error) {
	if len(p) > w.n {
		k := w.n
		w.n = 0
		return k, errCCWriter
	}
	w.n -= len(p)
	return len(p), nil
}

// TestConcurrent runs, from 16 goroutines, a mix of Decode and
// DecodeConfig calls and of Encode calls that succeed, that fail on the
// writer, that panic in At, on a malformed image or in the writer; every
// successful call must give image/jpeg's result, whatever the calls before
// it on the same pooled state did.
func TestConcurrent(t *testing.T) {
	decs, encs := ccInputs(t)
	broken := image.NewGray(image.Rect(0, 0, 30, 30))
	broken.Pix = broken.Pix[:400]
	brokenRGBA := image.NewRGBA(image.Rect(0, 0, 40, 33))
	brokenRGBA.Pix = brokenRGBA.Pix[:4000]
	iters := 300
	if testing.Short() {
		iters = 60
	}
	// call makes one call and describes what went wrong, if anything; a
	// panic the call should not make (state shared with another
	// goroutine can cause one anywhere) fails the test instead of the run.
	call := func(rng *rand.Rand) (e string) {
		defer func() {
			if p := recover(); p != nil {
				e = fmt.Sprintf("unexpected panic: %v", p)
			}
		}()
		c := &encs[rng.Intn(len(encs))]
		switch rng.Intn(8) {
		case 0, 1:
			d := &decs[rng.Intn(len(decs))]
			return ccCheckDec(d, bytes.NewReader(d.data), false)
		case 2:
			d := &decs[rng.Intn(len(decs))]
			return ccCheckDec(d, bytes.NewReader(d.data), true)
		case 3:
			if err := Encode(&ccFailWriter{n: rng.Intn(len(c.out))}, c.m, c.o); err != errCCWriter {
				return fmt.Sprintf("%s: failing writer: error %v", c.name, err)
			}
		case 4:
			b := c.m.Bounds()
			at := image.Pt(b.Min.X+rng.Intn(b.Dx()), b.Min.Y+rng.Intn(b.Dy()))
			if r := encOurs(ccPanicAt{c.m, at}, c.o); r.panic != "ccPanicAt" {
				return fmt.Sprintf("%s: At panicking at %v: panic %v", c.name, at, r.panic)
			}
		case 5:
			var m image.Image = broken
			if rng.Intn(2) == 0 {
				m = brokenRGBA
			}
			if r := encOurs(m, encQ(rng.Intn(100))); r.panic == nil {
				return fmt.Sprintf("%T with a short Pix: no panic", m)
			}
		case 6:
			defer func() {
				if p := recover(); p != "ccPanicWriter" {
					e = fmt.Sprintf("%s: panicking writer: panic %v", c.name, p)
				}
			}()
			Encode(&ccPanicWriter{n: rng.Intn(len(c.out))}, c.m, c.o)
		default:
			return ccCheckEnc(c, rng.Intn(1000))
		}
		return ""
	}
	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(g)))
			for it := 0; it < iters; it++ {
				if e := call(rng); e != "" {
					t.Errorf("goroutine %d, call %d: %s", g, it, e)
					return
				}
			}
		}(g)
	}
	wg.Wait()
}

// ccReenter is the work the re-entrant callbacks do: an Encode and a
// Decode of other inputs, whose results must be image/jpeg's.
type ccReenter struct {
	t    *testing.T
	decs []ccDec
	encs []ccEnc
	n    int
}

func (r *ccReenter) reenter() {
	r.n++
	if e := ccCheckEnc(&r.encs[r.n%len(r.encs)], r.n); e != "" {
		r.t.Errorf("re-entered Encode: %s", e)
	}
	d := &r.decs[r.n%len(r.decs)]
	if e := ccCheckDec(d, bytes.NewReader(d.data), r.n%5 == 0); e != "" {
		r.t.Errorf("re-entered Decode: %s", e)
	}
}

// ccReentrantWriter re-enters on every Write.
type ccReentrantWriter struct {
	*ccReenter
	b bytes.Buffer
}

func (w *ccReentrantWriter) Write(p []byte) (int, error) {
	w.reenter()
	return w.b.Write(p)
}

// ccReentrantFlushWriter is ccReentrantWriter with WriteByte and Flush,
// the methods that make image/jpeg write to it directly.
type ccReentrantFlushWriter struct{ ccReentrantWriter }

func (w *ccReentrantFlushWriter) WriteByte(c byte) error {
	_, err := w.Write([]byte{c})
	return err
}

func (w *ccReentrantFlushWriter) Flush() error {
	w.reenter()
	return nil
}

// ccReentrantReader re-enters on every Read, and returns at most 777
// bytes.
type ccReentrantReader struct {
	*ccReenter
	r io.Reader
}

func (r *ccReentrantReader) Read(p []byte) (int, error) {
	r.reenter()
	return r.r.Read(p[:min(len(p), 777)])
}

// ccReentrantImage re-enters on every 97th At.
type ccReentrantImage struct {
	image.Image
	*ccReenter
	calls int
}

func (m *ccReentrantImage) At(x, y int) color.Color {
	if m.calls++; m.calls%97 == 0 {
		m.reenter()
	}
	return m.Image.At(x, y)
}

// TestReentrant calls Encode and Decode from inside the Write, WriteByte,
// Flush, Read and At callbacks of another Encode or Decode.
func TestReentrant(t *testing.T) {
	decs, encs := ccInputs(t)
	step := 7
	if testing.Short() {
		step = 23
	}
	for i := 0; i < len(encs); i += step {
		c := &encs[i]
		re := &ccReenter{t: t, decs: decs, encs: encs, n: i}
		w := &ccReentrantWriter{ccReenter: re}
		if err, p := encCatch(func() error { return Encode(w, c.m, c.o) }); err != nil || p != nil || !bytes.Equal(w.b.Bytes(), c.out) {
			t.Errorf("%s: re-entrant writer: error %v, panic %v, output equal %v", c.name, err, p, bytes.Equal(w.b.Bytes(), c.out))
		}
		fw := &ccReentrantFlushWriter{ccReentrantWriter{ccReenter: re}}
		if err, p := encCatch(func() error { return Encode(fw, c.m, c.o) }); err != nil || p != nil || !bytes.Equal(fw.b.Bytes(), c.out) {
			t.Errorf("%s: re-entrant flush writer: error %v, panic %v, output equal %v", c.name, err, p, bytes.Equal(fw.b.Bytes(), c.out))
		}
		// Wrapping hides the type, and with it the fast paths: compare
		// with image/jpeg's output for the wrapped image.
		want := encStd(wrapped{c.m}, c.o).out
		got := encOurs(&ccReentrantImage{Image: c.m, ccReenter: re}, c.o)
		if got.err != nil || got.panic != nil || !bytes.Equal(got.out, want) {
			t.Errorf("%s: re-entrant image: error %v, panic %v, output equal %v", c.name, got.err, got.panic, bytes.Equal(got.out, want))
		}
	}
	for i := range decs {
		if testing.Short() && i%3 != 0 {
			continue
		}
		d := &decs[i]
		re := &ccReenter{t: t, decs: decs, encs: encs, n: i}
		for _, config := range []bool{false, true} {
			if e := ccCheckDec(d, &ccReentrantReader{ccReenter: re, r: bytes.NewReader(d.data)}, config); e != "" {
				t.Errorf("re-entrant reader: %s", e)
			}
		}
	}
}

// TestConcurrentInputs checks the inputs of TestConcurrent and
// TestReentrant from one goroutine, without re-entering, so that a
// failure there is about concurrency or re-entrancy, not the inputs.
func TestConcurrentInputs(t *testing.T) {
	decs, encs := ccInputs(t)
	for i := range decs {
		for _, config := range []bool{false, true} {
			if e := ccCheckDec(&decs[i], bytes.NewReader(decs[i].data), config); e != "" {
				t.Error(e)
			}
		}
	}
	for i := range encs {
		if e := ccCheckEnc(&encs[i], i); e != "" {
			t.Error(e)
		}
	}
}
