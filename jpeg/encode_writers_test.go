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

// More of the encoder's writer handling (encode_out.go) against
// image/jpeg's: writers failing while e.out is drained in the middle of
// the scan, writer kinds that encWriterKinds lacks, and misuse of the API
// (nil and zero-value writers, readers and images).

import (
	"bufio"
	"bytes"
	"fmt"
	"image"
	"image/color"
	std "image/jpeg"
	"io"
	"math/rand"
	"strings"
	"testing"
)

// TestEncWriterErrorsMidScan encodes an image of about 135 KB, more than
// encOutSize-encOutReserve, so that the scan loop drains e.out (drain with
// final false) before the end, and makes the writer fail, or write short,
// around those drain points and the 4096-byte chunk boundaries near them.
// TestEncWriterErrors' outputs (up to 17 KB) are drained only at the end.
// The writers see the same calls, bytes and error as with image/jpeg.
func TestEncWriterErrorsMidScan(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	m := image.NewRGBA(image.Rect(0, 0, 300, 220))
	rng.Read(m.Pix)
	const q = 100
	full := encStd(m, encQ(q)).out
	if len(full) < 2*(encOutSize-encOutReserve) {
		t.Fatalf("%d bytes of output: not enough to drain twice in the scan", len(full))
	}
	step, span := 37, 600
	if testing.Short() {
		step, span = 151, 450
	}
	var offsets []int
	for b := encOutSize - encOutReserve; b < len(full); b += encOutSize - encOutReserve {
		for d := -span; d <= span; d += step {
			offsets = append(offsets, b+d)
		}
		// The chunk boundaries before and after the drain point.
		for _, c := range []int{b / encChunk * encChunk, (b/encChunk + 1) * encChunk} {
			for d := -3; d <= 3; d++ {
				offsets = append(offsets, c+d)
			}
		}
	}
	offsets = append(offsets, 0, 1, len(full)-1, len(full), len(full)+1)
	for _, kind := range []string{"raw", "bufio4096", "rw"} {
		for _, short := range []bool{false, true} {
			if short && kind != "raw" {
				continue
			}
			for _, n := range offsets {
				got := encRunWriter(func(w io.Writer) error { return Encode(w, m, encQ(q)) }, kind, n, short)
				want := encRunWriter(func(w io.Writer) error { return std.Encode(w, m, &std.Options{Quality: q}) }, kind, n, short)
				if !encSameRun(got, want, true) {
					t.Fatalf("%s/short=%v: failing after %d of %d bytes: got %v, want %v", kind, short, n, len(full), got, want)
				}
			}
		}
	}
	// And without a failure, through every writer kind.
	for _, kind := range append(encWriterKinds[:len(encWriterKinds):len(encWriterKinds)], encMoreWriterKinds...) {
		got := encRunMoreWriter(func(w io.Writer) error { return Encode(w, m, encQ(q)) }, kind, -1, false)
		want := encRunMoreWriter(func(w io.Writer) error { return std.Encode(w, m, &std.Options{Quality: q}) }, kind, -1, false)
		if !encSameRun(got, want, kind != "flush") {
			t.Errorf("%s: got %v, want %v", kind, got, want)
		}
	}
}

// encWriteFlusher has Write and Flush but no WriteByte, so it is not an
// image/jpeg "writer": image/jpeg wraps it in a bufio.Writer like any
// io.Writer and never calls its Flush, and neither may Encode.
type encWriteFlusher struct {
	encRecWriter
	flushes int
}

func (w *encWriteFlusher) Flush() error {
	w.flushes++
	return nil
}

// encMoreWriterKinds are writer kinds of encRunMoreWriter beyond
// encWriterKinds: "writeflush" (encWriteFlusher) and "bufiostickyN" (a
// bufio.Writer of size N that already holds an error from an earlier
// write, which it returns from every later call).
var encMoreWriterKinds = []string{"writeflush", "bufiosticky1", "bufiosticky16", "bufiosticky4096"}

// encRunMoreWriter is encRunWriter for encWriterKinds and
// encMoreWriterKinds.
func encRunMoreWriter(enc func(io.Writer) error, kind string, fail int, short bool) (r encWriterRun) {
	var size int
	switch {
	case kind == "writeflush":
		w := &encWriteFlusher{encRecWriter: encRecWriter{fail: fail, short: short}}
		defer func() {
			r.panic = recover()
			r.out, r.calls, r.flushes = w.buf.Bytes(), w.calls, w.flushes
		}()
		r.err = enc(w)
	case strings.HasPrefix(kind, "bufiosticky"):
		if _, err := fmt.Sscanf(kind, "bufiosticky%d", &size); err != nil {
			panic(kind)
		}
		rec := &encRecWriter{fail: 0}
		bw := bufio.NewWriterSize(rec, size)
		bw.Write(make([]byte, size+1)) // Fails, and bw keeps the error.
		rec.fail, rec.calls = fail, nil
		defer func() {
			r.panic = recover()
			r.out, r.calls, r.buffed = rec.buf.Bytes(), rec.calls, bw.Buffered()
		}()
		r.err = enc(bw)
	default:
		return encRunWriter(enc, kind, fail, short)
	}
	return r
}

// TestEncWritersMore is TestEncWriters for encMoreWriterKinds: the same
// Write calls and bytes as with image/jpeg, no Flush call on a writer
// without WriteByte, and the error a bufio.Writer already holds.
func TestEncWritersMore(t *testing.T) {
	ims := map[string]image.Image{
		"rgba100x75":  imageMakers[0].make(image.Rect(0, 0, 100, 75), fills(3)[0].f),
		"gray300x200": imageMakers[5].make(image.Rect(0, 0, 300, 200), fills(4)[0].f),
		"ycc1x1":      imageMakers[12].make(image.Rect(0, 0, 1, 1), fills(4)[0].f),
		"empty":       image.NewRGBA(image.Rect(0, 0, 0, 0)),
	}
	for name, m := range ims {
		for _, kind := range encMoreWriterKinds {
			for _, q := range []int{1, 75, 100} {
				got := encRunMoreWriter(func(w io.Writer) error { return Encode(w, m, encQ(q)) }, kind, -1, false)
				want := encRunMoreWriter(func(w io.Writer) error { return std.Encode(w, m, &std.Options{Quality: q}) }, kind, -1, false)
				if !encSameRun(got, want, true) {
					t.Errorf("%s/%s/q%d: got %v, want %v", name, kind, q, got, want)
				}
			}
		}
	}
	// encWriteFlusher failing, or writing short, after every few bytes.
	m := ims["rgba100x75"]
	full := encStd(m, encQ(90)).out
	for n := 0; n <= len(full)+1; n += 97 {
		for _, short := range []bool{false, true} {
			got := encRunMoreWriter(func(w io.Writer) error { return Encode(w, m, encQ(90)) }, "writeflush", n, short)
			want := encRunMoreWriter(func(w io.Writer) error { return std.Encode(w, m, &std.Options{Quality: 90}) }, "writeflush", n, short)
			if !encSameRun(got, want, true) {
				t.Fatalf("writeflush/short=%v: failing after %d bytes: got %v, want %v", short, n, got, want)
			}
		}
	}
}

// encCatch returns f's error and the value it panics with, if any.
func encCatch(f func() error) (err error, p any) {
	defer func() { p = recover() }()
	return f(), nil
}

// encSameOutcome reports whether two (error, panic) outcomes are the same:
// the same error text, and panic values of the same type and text.
func encSameOutcome(e1 error, p1 any, e2 error, p2 any) bool {
	return fmt.Sprint(e1) == fmt.Sprint(e2) && fmt.Sprintf("%T %v", p1, p1) == fmt.Sprintf("%T %v", p2, p2)
}

// TestCompatMisuse checks that misuse of the API and unusual arguments give
// the same error or panic (value and type) as with image/jpeg: nil, typed
// nil and zero-value images, an image.Uniform (unbounded) and an
// image.Rectangle used as an image, images whose buffers are missing; nil
// and zero-value writers, including bufio.Writer and bufio.ReadWriter
// ones; and nil readers. From the verification of the encoder's and
// decoder's argument handling.
func TestCompatMisuse(t *testing.T) {
	images := []struct {
		name string
		m    image.Image
	}{
		{"nil", nil},
		{"typed nil RGBA", (*image.RGBA)(nil)},
		{"typed nil Gray", (*image.Gray)(nil)},
		{"typed nil YCbCr", (*image.YCbCr)(nil)},
		{"zero RGBA", &image.RGBA{}},
		{"zero YCbCr", &image.YCbCr{}},
		{"zero Paletted", &image.Paletted{}},
		{"zero NYCbCrA", &image.NYCbCrA{}},
		{"Uniform", image.NewUniform(color.RGBA{1, 2, 3, 4})},
		{"Rectangle", image.Rect(-5, -3, 30, 20)},
		{"Rectangle 65535 wide", image.Rect(0, 0, 1<<16-1, 3)},
		{"Paletted without palette", &image.Paletted{Rect: image.Rect(0, 0, 3, 3), Pix: make([]byte, 9), Stride: 3}},
		{"YCbCr without planes", &image.YCbCr{Rect: image.Rect(0, 0, 3, 3)}},
		{"Gray without Pix", &image.Gray{Rect: image.Rect(0, 0, 3, 3)}},
	}
	for _, c := range images {
		var got, want bytes.Buffer
		ge, gp := encCatch(func() error { return Encode(&got, c.m, nil) })
		we, wp := encCatch(func() error { return std.Encode(&want, c.m, nil) })
		if !encSameOutcome(ge, gp, we, wp) || (gp == nil && !bytes.Equal(got.Bytes(), want.Bytes())) {
			t.Errorf("Encode(%s): error %v, panic %T %v, %d bytes; want error %v, panic %T %v, %d bytes", c.name, ge, gp, gp, got.Len(), we, wp, wp, want.Len())
		}
	}
	m := image.NewRGBA(image.Rect(0, 0, 40, 40))
	writers := []struct {
		name string
		mk   func() io.Writer
	}{
		{"nil", func() io.Writer { return nil }},
		{"typed nil bufio.Writer", func() io.Writer { return (*bufio.Writer)(nil) }},
		{"zero bufio.Writer", func() io.Writer { return &bufio.Writer{} }},
		{"typed nil bufio.ReadWriter", func() io.Writer { return (*bufio.ReadWriter)(nil) }},
		{"ReadWriter with a nil Writer", func() io.Writer {
			return bufio.NewReadWriter(bufio.NewReader(strings.NewReader("")), nil)
		}},
		{"ReadWriter with a zero Writer", func() io.Writer { return &bufio.ReadWriter{Writer: &bufio.Writer{}} }},
		{"io.Discard", func() io.Writer { return io.Discard }},
		{"bufio over bufio", func() io.Writer { return bufio.NewWriterSize(bufio.NewWriterSize(io.Discard, 10), 3) }},
	}
	for _, c := range writers {
		ge, gp := encCatch(func() error { return Encode(c.mk(), m, nil) })
		we, wp := encCatch(func() error { return std.Encode(c.mk(), m, nil) })
		if !encSameOutcome(ge, gp, we, wp) {
			t.Errorf("Encode to %s: error %v, panic %T %v; want error %v, panic %T %v", c.name, ge, gp, gp, we, wp, wp)
		}
	}
	for _, r := range []io.Reader{nil, (*bytes.Reader)(nil), (*bufio.Reader)(nil)} {
		ge, gp := encCatch(func() error { _, err := Decode(r); return err })
		we, wp := encCatch(func() error { _, err := std.Decode(r); return err })
		if !encSameOutcome(ge, gp, we, wp) {
			t.Errorf("Decode(%T): error %v, panic %T %v; want error %v, panic %T %v", r, ge, gp, gp, we, wp, wp)
		}
		ge, gp = encCatch(func() error { _, err := DecodeConfig(r); return err })
		we, wp = encCatch(func() error { _, err := std.DecodeConfig(r); return err })
		if !encSameOutcome(ge, gp, we, wp) {
			t.Errorf("DecodeConfig(%T): error %v, panic %T %v; want error %v, panic %T %v", r, ge, gp, gp, we, wp, wp)
		}
	}
}
