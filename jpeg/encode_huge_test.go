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
	"fmt"
	"image"
	"image/color/palette"
	"math/rand"
	"testing"
)

// TestEncHugeTypes is TestEncHuge for the image types that test leaves
// out: Paletted, CMYK, Gray16, Alpha, RGBA64, YCbCr of every subsampling
// ratio (at an odd origin, which keeps them off the 4:2:0 and 4:2:2 fast
// paths) and an RGBA image seen only through At, 65535 pixels wide or high
// (the largest size image/jpeg encodes) and one pixel more (which it
// rejects). The encoder sizes its row buffers and chroma column tables per
// type from the width, and these sizes take them to their limit. From the
// verification of the encoder.
func TestEncHugeTypes(t *testing.T) {
	if testing.Short() {
		t.Skip("slow")
	}
	rng := rand.New(rand.NewSource(5))
	for _, r := range []image.Rectangle{
		image.Rect(-7, 3, 65535-7, 5), image.Rect(1, -1, 4, 65534), image.Rect(0, 0, 65535, 17),
		image.Rect(0, 0, 65536, 1), image.Rect(0, 0, 1, 65536),
	} {
		p := image.NewPaletted(r, palette.Plan9[:100])
		for i := range p.Pix {
			p.Pix[i] = uint8(rng.Intn(100))
		}
		c := image.NewCMYK(r)
		rng.Read(c.Pix)
		g16 := image.NewGray16(r)
		rng.Read(g16.Pix)
		a := image.NewAlpha(r)
		rng.Read(a.Pix)
		r64 := image.NewRGBA64(r)
		rng.Read(r64.Pix)
		rgba := image.NewRGBA(r)
		rng.Read(rgba.Pix)
		ims := []image.Image{p, c, g16, a, r64, wrapped{rgba}}
		for _, ratio := range encRatios[:6] {
			y := image.NewYCbCr(r.Add(image.Pt(1, 1)), ratio)
			rng.Read(y.Y)
			rng.Read(y.Cb)
			rng.Read(y.Cr)
			ims = append(ims, y)
		}
		for _, m := range ims {
			for _, q := range []int{1, 100} {
				compareEncode(t, fmt.Sprintf("%T/%v/q%d", m, m.Bounds(), q), m, encQ(q), false)
			}
		}
	}
}
