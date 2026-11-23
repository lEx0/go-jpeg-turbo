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

package ycc

import "testing"

// TestNRGBAMixedAlpha converts every RGB color once with alphas that
// change within the vector blocks (mixedAlpha), through NRGBAToYCbCr and
// NRGBAToYCbCr420 at every level, on the long rows of the exhaustive
// tests (under the race detector the same sample of them). In
// TestNRGBAToYCbCrExhaustive and its 4:2:0 counterpart every pixel of a
// row has the same alpha, so all the lanes of a block do too: a kernel
// that took a pixel's alpha from the wrong lane (a wrong shuffle mask, a
// broadcast of one lane) or that checked only some of the alphas before
// converting a block as opaque would pass them. Here the blocks of each
// row, and of each pair of 4:2:0 rows, mix opaque, transparent and other
// alphas, or are opaque but for one pixel at any lane, over every color.
// Adapted from the mixed-alpha exhaustive checks of the kernels'
// verification, which went over all 2^32 colors and alphas: the uniform
// alpha tests already prove every (color, alpha) pair in each lane, so
// one pass over the colors covers the lanes at about 1/256 of their cost.
func TestNRGBAMixedAlpha(t *testing.T) { forEachLevel(t, testNRGBAMixedAlpha) }

// mixedAlpha returns the alpha of pixel i of a TestNRGBAMixedAlpha sweep
// (whose row 0 has color i) in row (0 or 1). Each run of 64 pixels is,
// by a hash of the run and the row, opaque (a quarter of the runs),
// opaque but for one pixel (a quarter) or mixed (a hash of the pixel:
// 255 for a quarter of the pixels, 0 for an eighth, any value for the
// rest). The vector blocks start at the start of a row, which is not a
// multiple of 64, so they also straddle runs.
func mixedAlpha(i, row int) byte {
	g := mix32(uint32(i>>6)<<1 | uint32(row))
	switch g >> 30 {
	case 0:
		return 255
	case 1:
		if i&63 != int(g&63) {
			return 255
		}
		return byte(mix32(uint32(i)) % 255)
	}
	h := mix32(uint32(i)<<1 | uint32(row) | 1<<31)
	switch h >> 29 {
	case 0, 1:
		return 255
	case 2:
		return 0
	}
	return byte(h)
}

// mix32 is a 32-bit integer hash (a bijection).
func mix32(x uint32) uint32 {
	x ^= x >> 16
	x *= 0x7feb352d
	x ^= x >> 15
	x *= 0x846ca68b
	x ^= x >> 16
	return x
}

func testNRGBAMixedAlpha(t *testing.T) {
	tab, pm := tables()
	// pre returns the premultiplied color, the index into tab.
	pre := func(c int, a byte) int {
		p := &pm[a]
		return int(p[c>>16])<<16 | int(p[c>>8&0xff])<<8 | int(p[c&0xff])
	}
	const n = exhaustiveChunk // even
	chunks := exhaustiveChunks()
	var fe firstErr
	parallel(len(chunks), func(j int) {
		if fe.failed() {
			return
		}
		start := chunks[j] * n
		m := min(n, 1<<24-start)
		s0, s1 := make([]byte, 4*m), make([]byte, 4*m)
		for i := range m {
			c, d := start+i, perm(start+i)
			s0[4*i], s0[4*i+1], s0[4*i+2], s0[4*i+3] = byte(c>>16), byte(c>>8), byte(c), mixedAlpha(c, 0)
			s1[4*i], s1[4*i+1], s1[4*i+2], s1[4*i+3] = byte(d>>16), byte(d>>8), byte(d), mixedAlpha(c, 1)
		}
		y, cb, cr := make([]byte, m), make([]byte, m), make([]byte, m)
		NRGBAToYCbCr(s0, y, cb, cr, m)
		for i := range m {
			c := start + i
			a := mixedAlpha(c, 0)
			e := pre(c, a)
			if y[i] != tab[0][e] || cb[i] != tab[1][e] || cr[i] != tab[2][e] {
				fe.set("NRGBAToYCbCr(%06x, alpha %d) at %d of a row starting at color %06x = %d %d %d, want %d %d %d",
					c, a, i, start, y[i], cb[i], cr[i], tab[0][e], tab[1][e], tab[2][e])
				return
			}
		}
		y0, y1, cb2, cr2 := make([]byte, m), make([]byte, m), make([]byte, m/2), make([]byte, m/2)
		NRGBAToYCbCr420(s0, s1, y0, y1, cb2, cr2, m)
		check420(&fe, "NRGBAToYCbCr420 with mixed alphas", tab, start, m, y0, y1, cb2, cr2, func(i int) (int, int) {
			c, d := start+i, perm(start+i)
			return pre(c, mixedAlpha(c, 0)), pre(d, mixedAlpha(c, 1))
		})
	})
	if fe.err != nil {
		t.Fatalf("%s: %v", implName(), fe.err)
	}
}
