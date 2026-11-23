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

package dct

import (
	"image"
	"image/jpeg"
	"os"
	"sync"
	"testing"
)

// Real-picture inputs: the 8x8 sample blocks of the planes of a benchmark
// photo, and the coefficient blocks image/jpeg's encoder makes of them at
// quality 75 (what a decoder gets for such a file).

type photoData struct {
	samples [][BlockSize]byte // Y blocks, then Cb, Cr
	tabs    []*Quant          // per block
	coefs   []Block           // quantized coefficients, natural order
	qts     []*Block          // natural order, per block
}

var photoOnce = sync.OnceValues(func() (*photoData, error) {
	f, err := os.Open("../../testdata/bench/photo-960x1280-q75-420.jpg")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	m, err := jpeg.Decode(f)
	if err != nil {
		return nil, err
	}
	ycc := m.(*image.YCbCr)
	var p photoData
	var tabs [2]*Quant
	var qts [2]*Block
	for i := range tabs {
		q := qualityTable(75, i)
		tabs[i] = newQuant(&q)
		qts[i] = new(Block)
		for zig, v := range q {
			qts[i][Unzig[zig]] = int32(v)
		}
	}
	plane := func(pix []byte, stride, w, h, ti int) {
		for by := 0; by+8 <= h; by += 8 {
			for bx := 0; bx+8 <= w; bx += 8 {
				var s [BlockSize]byte
				for y := 0; y < 8; y++ {
					copy(s[8*y:8*y+8], pix[(by+y)*stride+bx:])
				}
				var out [BlockSize]int16
				FDCTQuantRef(s[:], 8, tabs[ti], &out)
				var c Block
				for zig, v := range out {
					c[Unzig[zig]] = int32(v)
				}
				p.samples = append(p.samples, s)
				p.tabs = append(p.tabs, tabs[ti])
				p.coefs = append(p.coefs, c)
				p.qts = append(p.qts, qts[ti])
			}
		}
	}
	r := ycc.Rect
	plane(ycc.Y, ycc.YStride, r.Dx(), r.Dy(), 0)
	cr := image.Rect(0, 0, (r.Dx()+1)/2, (r.Dy()+1)/2)
	plane(ycc.Cb, ycc.CStride, cr.Dx(), cr.Dy(), 1)
	plane(ycc.Cr, ycc.CStride, cr.Dx(), cr.Dy(), 1)
	return &p, nil
})

func photo(tb testing.TB) *photoData {
	p, err := photoOnce()
	if err != nil {
		tb.Skip(err)
	}
	return p
}

// TestPhotoKernels runs every kernel on the photo's blocks.
func TestPhotoKernels(t *testing.T) {
	p := photo(t)
	forLevels(t, func(t *testing.T) { testPhotoKernels(t, p) })
}

func testPhotoKernels(t *testing.T, p *photoData) {
	storeImpls, idctImpls, fdctImpls := atLevel(storeImpls), atLevel(idctImpls), atLevel(fdctImpls)
	var dc, low, right int
	for i := range p.coefs {
		b := &p.coefs[i]
		var ac, bottom, rgt int32
		for k := 1; k < BlockSize; k++ {
			ac |= b[k]
			if k >= 32 {
				bottom |= b[k]
			}
			if k%8 >= 4 {
				rgt |= b[k]
			}
		}
		if ac == 0 {
			dc++
		} else if bottom == 0 {
			low++
		}
		if ac != 0 && rgt == 0 {
			right++
		}
		for _, impl := range storeImpls {
			checkStore(t, nil, impl.name, impl.f, b, p.qts[i], 64, 3)
		}
		for _, impl := range idctImpls {
			checkIDCT(t, impl.name, impl.f, b, p.qts[i])
		}
		for _, impl := range fdctImpls {
			checkFDCT(t, impl.name, impl.f, &p.samples[i], p.tabs[i], 24, 5)
		}
	}
	n := len(p.coefs)
	t.Logf("%d blocks: %.1f%% DC only, %.1f%% other with rows 4-7 zero, %.1f%% non-DC with columns 4-7 zero",
		n, 100*float64(dc)/float64(n), 100*float64(low)/float64(n), 100*float64(right)/float64(n))
}

// photoSample returns benchBlocks blocks spread over the photo, so that the
// benchmarks see its distribution of blocks with a cache-resident working
// set (like a decoder, which reconstructs each block right after decoding
// it).
func photoSample(p *photoData) *photoData {
	var s photoData
	for i := 0; i < benchBlocks; i++ {
		k := i * len(p.coefs) / benchBlocks
		s.samples = append(s.samples, p.samples[k])
		s.tabs = append(s.tabs, p.tabs[k])
		s.coefs = append(s.coefs, p.coefs[k])
		s.qts = append(s.qts, p.qts[k])
	}
	return &s
}

func BenchmarkPhotoDequantIDCTStore(b *testing.B) {
	p := photoSample(photo(b))
	dst := make([]byte, 64*8)
	cs := sparseBlocks(p.coefs)
	run := func(f storeFn) func(b *testing.B) {
		return func(b *testing.B) {
			var w Block
			i := 0
			for b.Loop() {
				k := i % len(p.coefs)
				fill(&w, cs[k])
				f(&w, p.qts[k], dst, 64)
				i++
			}
		}
	}
	b.Run("impl=ref", run(DequantIDCTStoreRef))
	b.Run("impl=go", run(dequantIDCTStoreGo))
	runSIMD(b, "impl=", run(DequantIDCTStore))
	b.Run("impl=prep", run(func(b, qt *Block, dst []byte, stride int) {}))
}

func BenchmarkPhotoFDCTQuant(b *testing.B) {
	p := photoSample(photo(b))
	run := func(f fdctFn) func(b *testing.B) {
		return func(b *testing.B) {
			var out [BlockSize]int16
			i := 0
			for b.Loop() {
				k := i % len(p.samples)
				f(p.samples[k][:], 8, p.tabs[k], &out)
				i++
			}
		}
	}
	b.Run("impl=ref", run(FDCTQuantRef))
	b.Run("impl=go", run(fdctQuantGo))
	runSIMD(b, "impl=", run(FDCTQuant))
}
