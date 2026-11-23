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

// Switching the kernel levels in a running process. TestMain fixes one
// level per run (make levels runs one process per level), so state that
// the codec derives from the level and keeps, such as the quantization
// tables a pooled encoder caches with the kernels' layouts of one level,
// would pass every run and still give wrong output after a switch. These
// tests switch the levels of internal/dct and internal/ycc independently
// between calls and restore both (TestMain checks that). On arm64 and with
// -tags purego there is one level, and they check the clamping only. From
// the verification of the amd64 levels.

import (
	"bytes"
	"fmt"
	"image"
	std "image/jpeg"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/lEx0/go-jpeg-turbo/internal/cpuinfo"
	"github.com/lEx0/go-jpeg-turbo/internal/dct"
	"github.com/lEx0/go-jpeg-turbo/internal/ycc"
)

// lsKeepLevels restores the levels of dct and ycc at the end of the test.
func lsKeepLevels(tb testing.TB) {
	d, y := dct.Level(), ycc.Level()
	tb.Cleanup(func() {
		dct.SetLevel(d)
		ycc.SetLevel(y)
	})
}

func lsSet(tb testing.TB, d, y cpuinfo.Level) {
	tb.Helper()
	dct.SetLevel(d)
	ycc.SetLevel(y)
	if dct.Level() != d || ycc.Level() != y {
		tb.Fatalf("SetLevel(%v, %v): levels %v, %v", d, y, dct.Level(), ycc.Level())
	}
}

// lsDecodeInputs returns JPEG streams: every eighth corpus file (every
// 32nd with -short) and image/jpeg encodings of odd sizes, color and gray.
func lsDecodeInputs(tb testing.TB) (names []string, datas [][]byte) {
	every := 8
	if testing.Short() {
		every = 32
	}
	for i, f := range corpusFiles(tb) {
		if i%every == 0 {
			names = append(names, filepath.Base(f))
			datas = append(datas, readFile(tb, f))
		}
	}
	for _, w := range []int{1, 7, 9, 17, 33, 65, 129} {
		for _, h := range []int{1, 9, 17} {
			m := imageMakers[0].make(image.Rect(0, 0, w, h), encNoise(int64(w+h), false))
			g := imageMakers[5].make(image.Rect(0, 0, w, h), encNoise(int64(w*h), false))
			for _, q := range []int{1, 50, 100} {
				names = append(names, fmt.Sprintf("rgba%dx%dq%d", w, h, q), fmt.Sprintf("gray%dx%dq%d", w, h, q))
				datas = append(datas, encStd(m, encQ(q)).out, encStd(g, encQ(q)).out)
			}
		}
	}
	return names, datas
}

type lsEncCase struct {
	name string
	m    image.Image
	o    *Options
	want encodeResult
}

// lsEncodeInputs returns every image type at several sizes, with
// image/jpeg's results.
func lsEncodeInputs() []lsEncCase {
	makers := append(imageMakers[:len(imageMakers):len(imageMakers)], encExtraMakers...)
	opts := []*Options{nil, {Quality: 1}, {Quality: 100}, {Quality: 50}, {Quality: 90}, {Quality: 23}}
	var cs []lsEncCase
	for mi, mk := range makers {
		for _, sz := range []image.Point{{1, 1}, {17, 9}, {64, 16}, {67, 33}, {130, 17}, {255, 3}} {
			origin := image.Pt(mi%3, -mi%2)
			m := mk.make(image.Rectangle{origin, origin.Add(sz)}, encNoise(int64(mi*1000+sz.X), mi%2 == 1))
			o := opts[(mi+sz.X)%len(opts)]
			cs = append(cs, lsEncCase{fmt.Sprintf("%s/%v", mk.name, sz), m, o, encodeStd(m, o)})
		}
	}
	return cs
}

func lsCheckEncode(t *testing.T, c *lsEncCase, viaBufio bool, at string) {
	t.Helper()
	got := encodeOurs(c.m, c.o, viaBufio)
	if (got.panic == nil) != (c.want.panic == nil) || (got.err == nil) != (c.want.err == nil) || !bytes.Equal(got.out, c.want.out) {
		t.Errorf("%s [%s]: Encode: panic %v, error %v, %d bytes; want panic %v, error %v, %d bytes",
			c.name, at, got.panic, got.err, len(got.out), c.want.panic, c.want.err, len(c.want.out))
	}
}

// TestLevelSwitch encodes the same image at one level and then at another
// with the same quality, so that the second Encode reuses the pooled
// encoder and the tables it cached at the first level; then it switches
// the levels of dct and ycc at random, independently, between Decode and
// Encode calls.
func TestLevelSwitch(t *testing.T) {
	lsKeepLevels(t)
	levels := cpuinfo.Levels()
	encs := lsEncodeInputs()
	for _, from := range levels {
		for _, to := range levels {
			for i := 0; i < len(encs); i += 5 {
				c := &encs[i]
				lsSet(t, from, from)
				lsCheckEncode(t, c, false, fmt.Sprintf("at %v", from))
				lsSet(t, to, to)
				lsCheckEncode(t, c, i%2 == 0, fmt.Sprintf("at %v after %v", to, from))
			}
		}
	}

	names, datas := lsDecodeInputs(t)
	wantDec := make([]decodeResult, len(datas))
	for i, d := range datas {
		wantDec[i] = decodeStd(bytes.NewReader(d))
	}
	rng := rand.New(rand.NewSource(99))
	iters := 6 * (len(encs) + len(datas))
	if testing.Short() {
		iters = len(encs) + len(datas)
	}
	for it := 0; it < iters; it++ {
		dl, yl := levels[rng.Intn(len(levels))], levels[rng.Intn(len(levels))]
		lsSet(t, dl, yl)
		at := fmt.Sprintf("call %d, dct %v, ycc %v", it, dl, yl)
		if rng.Intn(2) == 0 {
			lsCheckEncode(t, &encs[rng.Intn(len(encs))], rng.Intn(4) == 0, at)
			continue
		}
		i := rng.Intn(len(datas))
		got, want := decodeOurs(bytes.NewReader(datas[i])), wantDec[i]
		if (got.panic == nil) != (want.panic == nil) || !sameError(got.err, want.err) {
			t.Errorf("%s [%s]: Decode: got %v, want %v", names[i], at, got, want)
		} else if want.err == nil && want.panic == nil {
			if d := diffImages(got.img, want.img); d != "" {
				t.Errorf("%s [%s]: Decode: %s", names[i], at, d)
			}
		}
	}
}

// TestLevelClamp checks that SetLevel clamps a level the machine lacks (or
// that does not exist) to [cpuinfo.Go, cpuinfo.Max] and returns the
// previous one, and that the codec still matches image/jpeg at the clamped
// levels.
func TestLevelClamp(t *testing.T) {
	lsKeepLevels(t)
	d0, y0 := dct.Level(), ycc.Level()
	for _, l := range []cpuinfo.Level{-100, -1, cpuinfo.Go, cpuinfo.SSE41, cpuinfo.AVX2, cpuinfo.AVX512, cpuinfo.AVX512 + 1, 1 << 30} {
		want := min(max(l, cpuinfo.Go), cpuinfo.Max)
		if pd, py := dct.SetLevel(l), ycc.SetLevel(l); pd != d0 || py != y0 {
			t.Errorf("SetLevel(%d) returned %v, %v; want the previous levels %v, %v", l, pd, py, d0, y0)
		}
		d0, y0 = dct.Level(), ycc.Level()
		if d0 != want || y0 != want {
			t.Errorf("SetLevel(%d): levels %v, %v; want %v", l, d0, y0, want)
		}
		m := imageMakers[0].make(image.Rect(0, 0, 131, 17), encNoise(int64(l), false))
		compareEncode(t, fmt.Sprintf("SetLevel(%d)", l), m, nil, false)
		var b bytes.Buffer
		if err := std.Encode(&b, m, nil); err != nil {
			t.Fatal(err)
		}
		compareDecode(t, fmt.Sprintf("SetLevel(%d)", l), b.Bytes(), readers[0].mk)
	}
}
