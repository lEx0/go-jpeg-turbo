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

package suite

import (
	"bytes"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"time"
)

// RSSMain is the main function of the peak-memory tools (../std/rss,
// ../turbo/rss). Each run measures one workload in a fresh process, since
// the peak resident memory (getrusage maxrss) only ever grows, and prints
// one line: the wall time of the workload, the peak resident memory before
// it (the Go runtime and the input) and the peak resident memory of the
// whole process, and for an encode the size of the output.
//
// Modes:
//   - dec12: one Decode of the generated 12 MP JPEG (420q90_4000x3000)
//     from memory;
//   - enc12: one Encode of the generated 12 MP *image.RGBA
//     (RGBA_4000x3000) at quality -q into a writer that only counts;
//   - gen: make the generated inputs if needed (see Generate) and print
//     their sizes and SHA-256 sums, for run.sh's env.txt;
//   - info: print Codec.Describe.
func RSSMain(c Codec) {
	mode := flag.String("mode", "", "dec12, enc12, gen or info")
	quality := flag.Int("q", 90, "enc12: Options.Quality")
	flag.Parse()

	fail := func(err error) {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	var base int64
	var size int
	var t0 time.Time
	start := func() {
		runtime.GC()
		base = peakRSS()
		t0 = time.Now()
	}
	switch *mode {
	case "info":
		fmt.Println(c.Describe())
		return
	case "gen":
		files, err := Generate()
		if err != nil {
			fail(err)
		}
		for _, f := range files {
			match := "the reference"
			if f.Sum != f.Want {
				match = "NOT the reference " + f.Want
			}
			fmt.Printf("%s: %d bytes, sha256 %s (%s)\n", filepath.Base(f.Path), f.Size, f.Sum, match)
		}
		return
	case "dec12":
		data, err := Photo12JPEG()
		if err != nil {
			fail(err)
		}
		start()
		if _, err := c.Decode(bytes.NewReader(data)); err != nil {
			fail(err)
		}
	case "enc12":
		m, err := Photo12RGBA()
		if err != nil {
			fail(err)
		}
		start()
		if size, err = encodedSize(c, m, *quality); err != nil {
			fail(err)
		}
	default:
		fail(fmt.Errorf("unknown -mode %q", *mode))
	}
	fmt.Printf("wall_ms=%.1f base_mb=%.1f peak_mb=%.1f", float64(time.Since(t0).Microseconds())/1000, float64(base)/1e6, float64(peakRSS())/1e6)
	if size > 0 {
		fmt.Printf(" bytes=%d", size)
	}
	fmt.Println()
}
