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

// The decoder's use of the io.Reader, call by call. compareDecode checks
// what Decode returns; these tests also check how it got there: every Read
// call must have the same len(p) and return the same (n, err) as with
// image/jpeg, in the same order. That is observable by a caller who reads
// one JPEG after another from a stream (MJPEG over HTTP, a socket, a pipe)
// or whose reader has side effects, and it is what a larger buffer, a
// read-ahead or a fast path that refills on its own would change while
// every image stays the same.

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	std "image/jpeg"
	"io"
	"path/filepath"
	"testing"
	"testing/iotest"
)

// rtCall is one Read call: the length of the buffer given and the result.
type rtCall struct {
	lenP, n int
	err     string
}

// rtTrace records every Read call made on r.
type rtTrace struct {
	r     io.Reader
	calls []rtCall
}

func (t *rtTrace) Read(p []byte) (int, error) {
	n, err := t.r.Read(p)
	s := ""
	if err != nil {
		s = err.Error()
	}
	t.calls = append(t.calls, rtCall{len(p), n, s})
	return n, err
}

var errRTScript = errors.New("scripted reader error")

// rtScript is a scripted reader of data. Each Read returns at most c bytes
// (0: no limit). The call that reaches offset k returns err together with
// the bytes before k; after it the reader keeps failing if sticky, or
// serves the rest of the data (a flaky reader). With zeroEvery > 0, every
// zeroEvery-th call returns (0, nil); with zeroBurst > 0, the calls from
// offset k on return (0, nil) zeroBurst times before the scripted error.
type rtScript struct {
	data      []byte
	off       int
	c         int
	k         int
	err       error
	sticky    bool
	fired     bool
	zeroEvery int
	zeroBurst int
	calls     int
}

func (r *rtScript) Read(p []byte) (int, error) {
	r.calls++
	if r.zeroEvery > 0 && r.calls%r.zeroEvery == 0 {
		return 0, nil
	}
	if r.zeroBurst > 0 && r.off >= r.k {
		r.zeroBurst--
		return 0, nil
	}
	if r.fired && r.sticky {
		return 0, r.err
	}
	if r.off >= len(r.data) {
		return 0, io.EOF
	}
	m := len(p)
	if r.c > 0 && m > r.c {
		m = r.c
	}
	if r.err != nil && !r.fired && r.off+m >= r.k {
		n := copy(p[:max(r.k-r.off, 0)], r.data[r.off:])
		r.off += n
		r.fired = true
		return n, r.err
	}
	n := copy(p[:m], r.data[r.off:])
	r.off += n
	return n, nil
}

type rtResult struct {
	img   image.Image
	cfg   image.Config
	err   error
	panic string
	calls []rtCall
}

// rtRun decodes (or decodes the config) from a reader made by mk with this
// package or image/jpeg, recording the Read calls.
func rtRun(mk func() io.Reader, ours, config bool) (res rtResult) {
	tr := &rtTrace{r: mk()}
	defer func() {
		if p := recover(); p != nil {
			res.panic = fmt.Sprint(p)
		}
		res.calls = tr.calls
	}()
	switch {
	case ours && config:
		res.cfg, res.err = DecodeConfig(tr)
	case ours:
		res.img, res.err = Decode(tr)
	case config:
		res.cfg, res.err = std.DecodeConfig(tr)
	default:
		res.img, res.err = std.Decode(tr)
	}
	return res
}

// rtCompare checks that Decode and DecodeConfig make the same Read calls
// as image/jpeg's on readers made by mk, and return the same result.
func rtCompare(t *testing.T, name string, mk func() io.Reader) {
	t.Helper()
	for _, config := range []bool{false, true} {
		what := "Decode"
		if config {
			what = "DecodeConfig"
		}
		got, want := rtRun(mk, true, config), rtRun(mk, false, config)
		if got.panic != want.panic || !sameError(got.err, want.err) {
			t.Errorf("%s: %s: panic %q, error %v; want panic %q, error %v", name, what, got.panic, got.err, want.panic, want.err)
			continue
		}
		i := 0
		for i < len(got.calls) && i < len(want.calls) && got.calls[i] == want.calls[i] {
			i++
		}
		if i < len(got.calls) || i < len(want.calls) {
			g, w := "none", "none"
			if i < len(got.calls) {
				g = fmt.Sprintf("%+v", got.calls[i])
			}
			if i < len(want.calls) {
				w = fmt.Sprintf("%+v", want.calls[i])
			}
			t.Errorf("%s: %s: Read call %d is %s, want %s (%d calls, want %d)", name, what, i, g, w, len(got.calls), len(want.calls))
			continue
		}
		switch {
		case config && got.cfg != want.cfg:
			t.Errorf("%s: DecodeConfig: %+v, want %+v", name, got.cfg, want.cfg)
		case !config && want.err == nil && want.panic == "":
			if d := diffImages(got.img, want.img); d != "" {
				t.Errorf("%s: Decode: %s", name, d)
			}
		}
	}
}

// rtFiles are the inputs of the trace tests: every kind of file (baseline,
// progressive, restart intervals, CMYK, RGB, gray, a truncated one, one
// that must be rejected), and the larger ones span several 4096-byte
// buffers.
var rtFiles = []string{
	"video-001.jpeg",
	"video-001.progressive.jpeg",
	"video-001.restart2.jpeg",
	"video-001.cmyk.jpeg",
	"video-001.rgb.jpeg",
	"video-001.separate.dc.progression.progressive.jpeg",
	"video-001.progressive.truncated.jpeg",
	"video-005.gray.jpeg",
	"video-001.q50.410.progressive.jpeg",
	"corpus/photo257x131_q100prog.jpeg",
	"corpus/photo257x131_rst8.jpeg",
	"corpus/video150x103_221122_progrst.jpeg",
	"corpus/video150x103_221122_rst1.jpeg",
	"corpus/photo257x131_arith.jpeg",
	"corpus/photo17x15_412111_progrst.jpeg",
}

// rtOffsets returns the offsets at which the scripted readers of
// TestDecReadTraces fail: about budget offsets spread over the n bytes,
// every offset within span of a buffer boundary (decBufBoundaries), and
// the last few bytes.
func rtOffsets(n, budget, span int) []int {
	seen := map[int]bool{}
	var out []int
	add := func(k int) {
		if k >= 0 && k <= n && !seen[k] {
			seen[k] = true
			out = append(out, k)
		}
	}
	for k := 0; k <= n; k += max(1, n/budget) {
		add(k)
	}
	for _, b := range decBufBoundaries(n) {
		for o := -span; o <= span; o++ {
			add(b + o)
		}
	}
	for o := 0; o <= span; o++ {
		add(n - o)
	}
	return out
}

// TestDecReadTraces compares the Read calls on readers that return short
// chunks, (0, nil) now and then, an error together with the last bytes, a
// flaky error followed by more data, io.EOF together with data, and a
// burst of (0, nil) before an error, at offsets spread over the input and
// around the buffer boundaries; with the fast entropy decoder forced on
// for every scan and not. From the verification of the decoder's
// io.Reader behaviour: it catches a change of the buffer size, a
// read-ahead, or a fast path that refills the buffer itself, none of which
// changes the decoded images.
func TestDecReadTraces(t *testing.T) {
	budget, span := 120, 6
	chunks := []int{0, 1, 2, 3, 7, 100, 4093, 4094, 4095, 4097}
	if testing.Short() {
		budget, span = 6, 1
		chunks = []int{0, 1, 7, 4095}
	}
	for _, force := range []bool{false, true} {
		t.Run(fmt.Sprintf("force=%v", force), func(t *testing.T) {
			if force {
				decForceFast(t)
			}
			for fi, f := range rtFiles {
				if testing.Short() && fi%2 != btoi(force) {
					// Each file once, with or without the fast path.
					continue
				}
				data := readFile(t, filepath.Join("../testdata", f))
				base := filepath.Base(f)
				for _, c := range chunks {
					name := fmt.Sprintf("%s/chunk%d", base, c)
					rtCompare(t, name, func() io.Reader { return &rtScript{data: data, c: c} })
					rtCompare(t, name+"/zeroEvery2", func() io.Reader { return &rtScript{data: data, c: c, zeroEvery: 2} })
					rtCompare(t, name+"/timeout", func() io.Reader { return iotest.TimeoutReader(&rtScript{data: data, c: c}) })
					rtCompare(t, name+"/dataerr", func() io.Reader { return iotest.DataErrReader(&rtScript{data: data, c: c}) })
					rtCompare(t, name+"/half", func() io.Reader { return iotest.HalfReader(&rtScript{data: data, c: c}) })
				}
				for _, k := range rtOffsets(len(data), budget, span) {
					for _, c := range []int{0, 1000} {
						name := fmt.Sprintf("%s/chunk%d", base, c)
						rtCompare(t, fmt.Sprintf("%s/errWithData@%d", name, k), func() io.Reader {
							return &rtScript{data: data, c: c, k: k, err: errRTScript, sticky: true}
						})
						rtCompare(t, fmt.Sprintf("%s/flaky@%d", name, k), func() io.Reader {
							return &rtScript{data: data, c: c, k: k, err: errRTScript}
						})
						rtCompare(t, fmt.Sprintf("%s/eofWithData@%d", name, k), func() io.Reader {
							return &rtScript{data: data, c: c, k: k, err: io.EOF, sticky: true}
						})
						rtCompare(t, fmt.Sprintf("%s/zeroBurst@%d", name, k), func() io.Reader {
							return &rtScript{data: data, c: c, k: k, zeroBurst: 20, err: errRTScript, sticky: true}
						})
					}
				}
			}
		})
	}
}

func btoi(b bool) int {
	if b {
		return 1
	}
	return 0
}

// TestDecReadTracesStall gives the decoders a reader that stops making
// progress: it returns (0, nil) 1000 times in a row (where a
// bufio.Reader-style loop would give up with io.ErrNoProgress after 100)
// and then fails. image/jpeg retries such reads for as long as they last,
// and so must Decode, with the same calls. It replaces the verification's
// timed "(0, nil) forever" test, which needed goroutines and a clock.
func TestDecReadTracesStall(t *testing.T) {
	for _, f := range rtFiles[:6] {
		data := readFile(t, filepath.Join("../testdata", f))
		for _, k := range []int{0, 1, 2, 100, len(data) / 2, 4096, len(data) - 2, len(data)} {
			if k > len(data) {
				continue
			}
			rtCompare(t, fmt.Sprintf("%s/stall@%d", filepath.Base(f), k), func() io.Reader {
				return &rtScript{data: data, k: k, zeroBurst: 1000, err: errRTScript, sticky: true}
			})
		}
	}
}

// TestDecReadTracesCorpus runs a few scripted readers over the whole
// corpus (every eighth file with -short), with the fast entropy decoder
// forced on and not.
func TestDecReadTracesCorpus(t *testing.T) {
	for _, force := range []bool{false, true} {
		t.Run(fmt.Sprintf("force=%v", force), func(t *testing.T) {
			if force {
				decForceFast(t)
			}
			for i, f := range corpusFiles(t) {
				if testing.Short() && i%8 != 4*btoi(force) {
					continue
				}
				data := readFile(t, f)
				n, es := len(data), entropyStart(data)
				if n > 64<<10 && testing.Short() {
					continue
				}
				mks := []struct {
					name string
					mk   func() io.Reader
				}{
					{"chunk0", func() io.Reader { return &rtScript{data: data} }},
					{"chunk7", func() io.Reader { return &rtScript{data: data, c: 7} }},
					{"chunk4095", func() io.Reader { return &rtScript{data: data, c: 4095} }},
					{"zeroEvery3", func() io.Reader { return &rtScript{data: data, c: 1000, zeroEvery: 3} }},
					{"timeout", func() io.Reader { return iotest.TimeoutReader(&rtScript{data: data, c: 3000}) }},
					{"flakyMid", func() io.Reader { return &rtScript{data: data, k: es + (n-es)/2, err: errRTScript} }},
					{"errES+3", func() io.Reader { return &rtScript{data: data, c: 512, k: es + 3, err: errRTScript, sticky: true} }},
					{"errMid", func() io.Reader {
						return &rtScript{data: data, c: 512, k: es + (n-es)/2, err: errRTScript, sticky: true}
					}},
					{"eofEnd-2", func() io.Reader { return &rtScript{data: data, k: n - 2, err: io.EOF, sticky: true} }},
					{"zeroBurstES", func() io.Reader {
						return &rtScript{data: data, c: 100, k: es + 1, zeroBurst: 5, err: errRTScript, sticky: true}
					}},
				}
				for _, m := range mks {
					if m.name == "chunk7" && n > 64<<10 {
						continue
					}
					rtCompare(t, filepath.Base(f)+"/"+m.name, m.mk)
				}
			}
		})
	}
}

// TestDecTrailingData decodes a JPEG followed by more data, as in an MJPEG
// stream where the next frame follows the EOI marker, and checks that
// Decode consumes as many bytes of the reader as image/jpeg, so that the
// caller can go on reading the next frame from the same place.
func TestDecTrailingData(t *testing.T) {
	for _, force := range []bool{false, true} {
		t.Run(fmt.Sprintf("force=%v", force), func(t *testing.T) {
			if force {
				decForceFast(t)
			}
			for _, f := range rtFiles {
				data := readFile(t, filepath.Join("../testdata", f))
				for _, tail := range []int{0, 1, 2, 100, 5000, 70000} {
					next := bytes.Repeat([]byte{0xff, 0xd8, 0xff, 0xe0}, tail/4+1)[:tail]
					in := append(append([]byte(nil), data...), next...)
					for _, c := range []int{0, 1, 513} {
						got, want := &rtScript{data: in, c: c}, &rtScript{data: in, c: c}
						Decode(got)
						std.Decode(want)
						if got.off != want.off {
							t.Errorf("%s with %d more bytes, chunk %d: consumed %d bytes, want %d", filepath.Base(f), tail, c, got.off, want.off)
						}
					}
				}
			}
		})
	}
}
