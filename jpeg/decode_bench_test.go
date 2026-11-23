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
	"testing"
)

// BenchmarkDecodeReuse decodes the sequential benchmark images (see
// benchDecodeFiles) into the image of an earlier decode, so that it
// measures the entropy decoding and reconstruction (decScanSeq) without the
// allocation and first touch of the image, which make BenchmarkCompatDecode
// noisy on a loaded machine. MB/s is megapixels per second.
//
// It is the benchmark for changes of the block loop. One such change was
// tried for the store-to-load forwarding stall that the dct benchmarks see
// (the vector loads of DequantIDCTStore cannot take their data from the
// narrower stores that just zeroed and filled the block, about 14 ns per
// block there). On the EPYC 8124P (Zen 4) the conflicts happen in the
// decoder too, but out-of-order execution hides the wait behind the
// entropy decoding of the next block: with perf stat (cycles,
// ex_no_retire.load_not_complete) the reconstruction costs about 75 cycles
// per block of the 420q75 image at the AVX2 level and the kernel adds no
// cycles of retirement waiting for loads; deferring each reconstruction by
// one to three blocks made the decode 2-8% slower (SSE4.1 and AVX2), and
// handing the kernel copies of the blocks written long before was no
// faster than handing it the fresh block.
func BenchmarkDecodeReuse(b *testing.B) {
	for _, f := range benchDecodeFiles {
		data := readFile(b, f.path)
		var d0 decoder
		m, err := d0.decode(bytes.NewReader(data), false)
		if err != nil {
			b.Fatal(err)
		}
		if d0.progressive || d0.flex || d0.nComp == 4 {
			continue
		}
		redecode := func(d *decoder, r *bytes.Reader) {
			*d = decoder{img1: d0.img1, img3: d0.img3}
			r.Reset(data)
			if _, err := d.decode(r, false); err != nil {
				b.Fatal(err)
			}
		}
		d, r := new(decoder), new(bytes.Reader)
		redecode(d, r)
		want, err := std.Decode(bytes.NewReader(data))
		if err != nil {
			b.Fatal(err)
		}
		if s := diffImages(m, want); s != "" {
			b.Fatalf("%s: %s", f.name, s)
		}
		b.Run("img="+f.name, func(b *testing.B) {
			b.SetBytes(int64(m.Bounds().Dx() * m.Bounds().Dy()))
			b.ReportAllocs()
			for b.Loop() {
				redecode(d, r)
			}
		})
	}
}
