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

// Real files whose frame header declares another size than the data
// holds: 0, 1, odd sizes, sizes a little off and sizes far larger. The
// decoder sizes its image, its coefficient buffers and its block loops
// from the header, and the data then ends early, runs into the next
// marker or leaves blocks unvisited; mutate() changes header bytes only at
// random. From the verification of the decoder's frame handling.

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"io"
	"path/filepath"
	"testing"
)

// dsPatchSize returns data with the size in its first SOF0, SOF1 or SOF2
// segment set to w x h, or nil if it has none.
func dsPatchSize(data []byte, w, h int) []byte {
	out := append([]byte(nil), data...)
	for i := 2; i+9 < len(out); i++ {
		if out[i] == 0xff && (out[i+1] == sof0Marker || out[i+1] == sof1Marker || out[i+1] == sof2Marker) {
			binary.BigEndian.PutUint16(out[i+5:], uint16(h))
			binary.BigEndian.PutUint16(out[i+7:], uint16(w))
			return out
		}
	}
	return nil
}

// TestDecPatchedSizes decodes the files of the Read-call trace tests with
// patched sizes, comparing results and Read calls (rtCompare) through a
// bytes.Reader and a reader of 5 bytes at a time, with the fast entropy
// decoder forced on and not.
func TestDecPatchedSizes(t *testing.T) {
	sizes := []image.Point{{0, 0}, {0, 7}, {7, 0}, {1, 1}, {1, 300}, {300, 1}, {149, 103}, {151, 104}, {9, 9}, {2000, 3}, {3, 2000}, {800, 600}}
	for _, force := range []bool{false, true} {
		t.Run(fmt.Sprintf("force=%v", force), func(t *testing.T) {
			if force {
				decForceFast(t)
			}
			for _, f := range rtFiles {
				data := readFile(t, filepath.Join("../testdata", f))
				for _, sz := range sizes {
					p := dsPatchSize(data, sz.X, sz.Y)
					if p == nil {
						continue
					}
					name := fmt.Sprintf("%s@%dx%d", filepath.Base(f), sz.X, sz.Y)
					rtCompare(t, name, func() io.Reader { return bytes.NewReader(p) })
					rtCompare(t, name+"/chunk5", func() io.Reader { return &rtScript{data: p, c: 5} })
				}
			}
		})
	}
}

// TestDecProgZeroDims sets the width or the height of progressive files
// (gray, 4:2:0, with restart intervals) to 0 and to other small values, and
// decodes them through every reader, with the fast path forced on and not.
func TestDecProgZeroDims(t *testing.T) {
	files := []string{"photo16x16_420_prog.jpeg", "photo33x31_grayprog.jpeg", "video150x103_221122_progrst.jpeg", "photo17x15_411_progrst.jpeg"}
	dims := []image.Point{{0, 16}, {16, 0}, {0, 0}, {1, 1}, {1, 300}, {300, 1}, {9, 2}}
	for _, force := range []bool{false, true} {
		t.Run(fmt.Sprintf("force=%v", force), func(t *testing.T) {
			if force {
				decForceFast(t)
			}
			for _, f := range files {
				data := readFile(t, filepath.Join("../testdata/corpus", f))
				for _, d := range dims {
					p := dsPatchSize(data, d.X, d.Y)
					if p == nil {
						t.Fatalf("%s: no SOF segment", f)
					}
					for _, r := range readers {
						compareDecode(t, fmt.Sprintf("%s@%dx%d/%s", f, d.X, d.Y, r.name), p, r.mk)
					}
				}
			}
		})
	}
}
