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

import (
	"bytes"
	std "image/jpeg"
	"path/filepath"
	"runtime"
	"testing"
)

// daConfigBudget is the most DecodeConfig may allocate: 16,432 bytes
// with Go 1.27 on 64-bit, nearly all of it the decoder (image/jpeg's
// 13,616). A fixed budget rather than a margin over image/jpeg's
// allocation, because the tests compare with the image/jpeg of whatever
// toolchain runs them: a newer Go whose decoder is smaller would
// otherwise fail the test with nothing changed here.
const daConfigBudget = 20 << 10

// daAllocated returns the number of bytes f allocates (the least of three
// runs, to leave out what other goroutines allocate meanwhile).
func daAllocated(f func()) uint64 {
	least := ^uint64(0)
	for range 3 {
		var a, b runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&a)
		f()
		runtime.ReadMemStats(&b)
		least = min(least, b.TotalAlloc-a.TotalAlloc)
	}
	return least
}

// TestDecAllocs compares the memory Decode and DecodeConfig allocate with
// image/jpeg's: for real files, and for files whose header declares a
// size far larger than their data, which both decoders reject after
// allocating the image (and, for progressive files, the coefficients) of
// the declared size. Decode may allocate at most 10% and 64 KB more, so
// that a malicious header cannot make the fast paths amplify memory use
// (masks allocated for every block before any data, a second coefficient
// buffer). DecodeConfig allocates the decoder and nothing that depends on
// the input; it may allocate at most daConfigBudget. The declared sizes
// stop at 2048x2048 so that the test stays within the memory of a CI
// machine. From the verification of the decoder.
func TestDecAllocs(t *testing.T) {
	if encRaceEnabled {
		t.Skip("the race detector changes the allocations")
	}
	type input struct {
		name string
		data []byte
	}
	var in []input
	files := []string{
		"bench/photo-960x1280-q75-420.jpg",
		"bench/photo-960x1280-q85-420-prog.jpg",
		"bench/photo-960x1280-q92-422-rst.jpg",
		"video-001.cmyk.jpeg",
		"video-001.jpeg",
		"video-005.gray.q50.progressive.jpeg",
		"corpus/photo16x16_412111_prog.jpeg",
		"corpus/photo257x131_q100prog.jpeg",
	}
	if !testing.Short() {
		files = append(files, "bench/photo-1920x2560-q85-420.jpg", "bench/photo-1920x2560-q85-gray.jpg")
	}
	for _, f := range files {
		in = append(in, input{filepath.Base(f), readFile(t, filepath.Join("../testdata", f))})
	}
	for _, f := range []string{"video-001.jpeg", "video-001.progressive.jpeg", "video-005.gray.jpeg", "video-005.gray.q50.progressive.jpeg"} {
		data := readFile(t, filepath.Join("../testdata", f))
		for _, sz := range [][2]int{{2048, 2048}, {2047, 1153}} {
			in = append(in, input{filepath.Base(f) + "@" + itoa(sz[0]) + "x" + itoa(sz[1]), dsPatchSize(data, sz[0], sz[1])})
		}
	}
	for _, x := range in {
		got := daAllocated(func() { Decode(bytes.NewReader(x.data)) })
		want := daAllocated(func() { std.Decode(bytes.NewReader(x.data)) })
		t.Logf("Decode %s: %d bytes, image/jpeg %d (%.3f)", x.name, got, want, float64(got)/float64(want))
		if float64(got) > 1.10*float64(want)+64<<10 {
			t.Errorf("Decode %s: allocates %d bytes, image/jpeg %d", x.name, got, want)
		}
		got = daAllocated(func() { DecodeConfig(bytes.NewReader(x.data)) })
		want = daAllocated(func() { std.DecodeConfig(bytes.NewReader(x.data)) })
		t.Logf("DecodeConfig %s: %d bytes, image/jpeg %d (%.3f)", x.name, got, want, float64(got)/float64(want))
		if got > daConfigBudget {
			t.Errorf("DecodeConfig %s: allocates %d bytes, more than %d (image/jpeg %d)", x.name, got, daConfigBudget, want)
		}
	}
}
