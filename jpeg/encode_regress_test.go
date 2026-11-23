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

// Regression tests of the encoder against image/jpeg (std) that compare the
// panic value too, not only whether Encode panics:
//
//   - malformed images (a Pix, Y, Cb, Cr or A too short for the rectangle,
//     bad strides, palette indexes past the palette) and images whose At
//     panics at some pixels must panic at the pixel image/jpeg reads first,
//     with the same value: image/jpeg reads MCU by MCU, four 8x8 blocks
//     each, row by row;
//   - an *image.RGBA whose Pix ends exactly at a pixel must report index 0
//     of that pixel, as image/jpeg's color.RGBToYCbCr(pix[0], pix[1],
//     pix[2]) does;
//   - rectangles near the int limits, where Dx or Dy overflows or
//     image/jpeg's coordinate loops (x += 16) overflow, must behave as
//     image/jpeg's loops do.

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/color/palette"
	std "image/jpeg"
	"io"
	"math"
	"math/rand"
	"testing"
)

// encStrictResult is what an Encode call did: the bytes written, the error
// text and the panic value with its type.
type encStrictResult struct {
	out   []byte
	err   string
	panic string
}

func (r encStrictResult) String() string {
	if r.panic != "" {
		return fmt.Sprintf("panic %q (%d bytes written)", r.panic, len(r.out))
	}
	return fmt.Sprintf("error %q, %d bytes", r.err, len(r.out))
}

func encStrictRun(enc func(io.Writer) error) (r encStrictResult) {
	var buf bytes.Buffer
	defer func() {
		if p := recover(); p != nil {
			r.panic = fmt.Sprintf("%T: %v", p, p)
		}
		r.out = buf.Bytes()
	}()
	if err := enc(&buf); err != nil {
		r.err = err.Error()
	}
	return
}

// encStrictCompare encodes the image that mk returns (called once per
// encoder, so that images with state start afresh) with both encoders. The
// panic values and errors must be the same; without a panic, so must the
// bytes. With a panic, fewer bytes may have reached the writer (deviation
// (b)), but they must be a prefix of the same stream.
func encStrictCompare(t testing.TB, name string, mk func() image.Image, o *Options) (panicked bool) {
	t.Helper()
	got := encStrictRun(func(w io.Writer) error { return Encode(w, mk(), o) })
	want := encStrictRun(func(w io.Writer) error { return std.Encode(w, mk(), stdOptions(o)) })
	if got.panic != want.panic || got.err != want.err {
		t.Errorf("%s: got %v, want %v", name, got, want)
		return want.panic != ""
	}
	if want.panic == "" {
		if !bytes.Equal(got.out, want.out) {
			t.Errorf("%s: output differs (len %d, want %d)", name, len(got.out), len(want.out))
		}
		return false
	}
	n := min(len(got.out), len(want.out))
	if !bytes.Equal(got.out[:n], want.out[:n]) {
		t.Errorf("%s: panicked after writing different bytes", name)
	}
	return true
}

func encStatic(m image.Image) func() image.Image { return func() image.Image { return m } }

// encBadAt is an image whose At panics, with the coordinates, at the
// pixels of bad, and after limit calls (so that a runaway loop of At calls
// ends). It records the calls if log is set.
type encBadAt struct {
	r     image.Rectangle
	bad   map[image.Point]bool
	limit int
	n     int
	log   bool
	calls []image.Point
}

func (m *encBadAt) ColorModel() color.Model { return color.RGBAModel }
func (m *encBadAt) Bounds() image.Rectangle { return m.r }
func (m *encBadAt) At(x, y int) color.Color {
	m.n++
	if m.limit > 0 && m.n > m.limit {
		panic(fmt.Sprintf("encBadAt: more than %d calls", m.limit))
	}
	if m.log {
		m.calls = append(m.calls, image.Pt(x, y))
	}
	if m.bad[image.Pt(x, y)] {
		panic(fmt.Sprintf("encBadAt: pixel (%d,%d) unavailable", x, y))
	}
	return color.NRGBA{uint8(x*37 + y), uint8(y*91 - x), uint8(x ^ y), uint8(128 + x*y)}
}

func encBad(r image.Rectangle, pts ...image.Point) func() image.Image {
	return func() image.Image {
		m := &encBadAt{r: r, bad: map[image.Point]bool{}, limit: 1 << 20}
		for _, p := range pts {
			m.bad[p] = true
		}
		return m
	}
}

// encTestColor is a palette color of a type the encoder does not know.
type encTestColor struct{ v uint8 }

func (c encTestColor) RGBA() (r, g, b, a uint32) {
	v := uint32(c.v) * 0x101
	return v, 0xffff - v, v / 2, 0xffff
}

// TestEncRegressRGBAPixelEnd: an *image.RGBA whose Pix ends exactly at a
// pixel (Pix one pixel short, or empty).
func TestEncRegressRGBAPixelEnd(t *testing.T) {
	encStrictCompare(t, "1x1/nil", encStatic(&image.RGBA{Rect: image.Rect(0, 0, 1, 1)}), nil)
	encStrictCompare(t, "1x1/stride4", encStatic(&image.RGBA{Pix: []byte{}, Stride: 4, Rect: image.Rect(0, 0, 1, 1)}), nil)
	for w := 1; w <= 8; w++ {
		for cut := 1; cut <= 8; cut++ {
			m := image.NewRGBA(image.Rect(0, 0, w, 1))
			if cut > len(m.Pix) {
				continue
			}
			m.Pix = m.Pix[:len(m.Pix)-cut]
			encStrictCompare(t, fmt.Sprintf("%dx1/cut%d", w, cut), encStatic(m), nil)
		}
	}
	m := image.NewRGBA(image.Rect(0, 0, 17, 9))
	m.Pix = m.Pix[:len(m.Pix)-4]
	encStrictCompare(t, "17x9/cut4", encStatic(m), nil)
	// Only the alpha byte of the last pixel is missing: image/jpeg does not
	// read it, so this encodes.
	m = image.NewRGBA(image.Rect(2, 3, 19, 12))
	rand.New(rand.NewSource(1)).Read(m.Pix)
	m.Pix = m.Pix[:len(m.Pix)-1]
	if encStrictCompare(t, "17x9/cut1", encStatic(m), encQ(90)) {
		t.Errorf("17x9/cut1: image/jpeg panics")
	}
}

// TestEncRegressVisitOrder: images with more than one bad pixel or sample
// must fail at the one image/jpeg reads first.
func TestEncRegressVisitOrder(t *testing.T) {
	cases := map[string]func() image.Image{}
	add := func(name string, m image.Image) { cases[name] = encStatic(m) }

	add("Gray9x2/Pix8", &image.Gray{Pix: make([]byte, 8), Stride: 9, Rect: image.Rect(0, 0, 9, 2)})
	add("Gray16x2/Pix12", &image.Gray{Pix: make([]byte, 12), Stride: 16, Rect: image.Rect(0, 0, 16, 2)})
	g := image.NewGray(image.Rect(0, 0, 16, 8))
	g.Pix = g.Pix[:108]
	add("Gray16x8/Pix108", g)

	rgba := image.NewRGBA(image.Rect(0, 0, 2, 1))
	rgba.Pix = rgba.Pix[:4]
	add("RGBA2x1/Pix4", rgba)
	rgba = image.NewRGBA(image.Rect(0, 0, 16, 2))
	rgba.Pix = rgba.Pix[:48]
	add("RGBA16x2/Pix48", rgba)

	nrgba := image.NewNRGBA(image.Rect(0, 0, 9, 2))
	nrgba.Pix = nrgba.Pix[:32:32]
	add("NRGBA9x2/Pix32", nrgba)
	nrgba = image.NewNRGBA(image.Rect(0, 0, 16, 2))
	nrgba.Pix = nrgba.Pix[:48:48]
	add("NRGBA16x2/Pix48", nrgba)
	cmyk := image.NewCMYK(image.Rect(0, 0, 16, 2))
	cmyk.Pix = cmyk.Pix[:48:48]
	add("CMYK16x2/Pix48", cmyk)
	g16 := image.NewGray16(image.Rect(0, 0, 16, 2))
	g16.Pix = g16.Pix[:24]
	add("Gray16/16x2/Pix24", g16)

	y444 := image.NewYCbCr(image.Rect(0, 0, 16, 2), image.YCbCrSubsampleRatio444)
	y444.Y = y444.Y[:12]
	add("YCbCr444/16x2/Y12", y444)
	y444 = image.NewYCbCr(image.Rect(0, 0, 16, 2), image.YCbCrSubsampleRatio444)
	y444.Y, y444.Cb = y444.Y[:31], y444.Cb[:2]
	add("YCbCr444/16x2/Y31Cb2", y444)
	y420 := image.NewYCbCr(image.Rect(0, 0, 16, 16), image.YCbCrSubsampleRatio420)
	y420.Y = y420.Y[:236]
	add("YCbCr420/16x16/Y236", y420)
	ya := image.NewNYCbCrA(image.Rect(0, 0, 16, 2), image.YCbCrSubsampleRatio444)
	ya.Y = ya.Y[:12]
	add("NYCbCrA444/16x2/Y12", ya)

	p := image.NewPaletted(image.Rect(0, 0, 9, 2), color.Palette{color.Gray{7}, nil})
	p.Pix[8], p.Pix[9] = 1, 7
	add("Paletted9x2/nil+7", p)
	p = image.NewPaletted(image.Rect(0, 0, 16, 2), color.Palette{color.Black})
	p.Pix[8], p.Pix[16] = 5, 7
	add("Paletted16x2/5+7", p)
	p = image.NewPaletted(image.Rect(0, 0, 16, 2), color.Palette{encTestColor{3}, encTestColor{9}})
	p.Pix[12], p.Pix[16] = 5, 7
	add("Paletted16x2/custom/5+7", p)
	p = image.NewPaletted(image.Rect(0, 0, 40, 20), palette.Plan9[:16])
	p.Pix[3*40+33], p.Pix[10*40+2] = 200, 100
	add("Paletted40x20/200+100", p)

	cases["At9x2"] = encBad(image.Rect(0, 0, 9, 2), image.Pt(8, 0), image.Pt(0, 1))
	cases["At16x2"] = encBad(image.Rect(0, 0, 16, 2), image.Pt(12, 0), image.Pt(0, 1))
	cases["At40x40"] = encBad(image.Rect(-3, 5, 37, 45), image.Pt(20, 10), image.Pt(-3, 19), image.Pt(36, 5))

	for name, mk := range cases {
		if !encStrictCompare(t, name, mk, nil) {
			t.Errorf("%s: image/jpeg does not panic", name)
		}
	}
}

// encMalformed returns a random malformed (or panicking) image.
func encMalformed(rng *rand.Rand) (string, func() image.Image) {
	r := image.Rect(rng.Intn(41)-20, rng.Intn(41)-20, 0, 0)
	size := 40
	if rng.Intn(8) == 0 {
		size = 100
	}
	r.Max = r.Min.Add(image.Pt(1+rng.Intn(size), 1+rng.Intn(size)))
	// cut shortens a buffer, keeping its capacity or not; stride returns
	// a broken stride now and then.
	cut := func(b []byte) []byte {
		n := len(b)
		switch rng.Intn(4) {
		case 0:
			n -= 1 + rng.Intn(4)
		case 1:
			n -= 1 + rng.Intn(max(1, len(b)/3))
		case 2:
			n = rng.Intn(n + 1)
		}
		n = max(n, 0)
		if rng.Intn(2) == 0 {
			return b[:n:n]
		}
		return b[:n]
	}
	stride := func(s int) int {
		switch rng.Intn(8) {
		case 0:
			return s + 1 + rng.Intn(8)
		case 1:
			if s > 0 {
				return s - 1 - rng.Intn(s)
			}
		case 2:
			return -s
		}
		return s
	}
	fill := func(b []byte) []byte { rng.Read(b); return b }
	kinds := []string{"Gray", "Gray16", "Alpha", "Alpha16", "RGBA", "NRGBA", "RGBA64", "NRGBA64", "CMYK", "Paletted", "PalettedOdd", "YCbCr", "NYCbCrA", "At"}
	kind := kinds[rng.Intn(len(kinds))]
	var m image.Image
	switch kind {
	case "Gray":
		g := image.NewGray(r)
		g.Pix, g.Stride = cut(fill(g.Pix)), stride(g.Stride)
		m = g
	case "Gray16":
		g := image.NewGray16(r)
		g.Pix, g.Stride = cut(fill(g.Pix)), stride(g.Stride)
		m = g
	case "Alpha":
		a := image.NewAlpha(r)
		a.Pix, a.Stride = cut(fill(a.Pix)), stride(a.Stride)
		m = a
	case "Alpha16":
		a := image.NewAlpha16(r)
		a.Pix, a.Stride = cut(fill(a.Pix)), stride(a.Stride)
		m = a
	case "RGBA":
		a := image.NewRGBA(r)
		a.Pix, a.Stride = cut(fill(a.Pix)), stride(a.Stride)
		m = a
	case "NRGBA":
		a := image.NewNRGBA(r)
		a.Pix, a.Stride = cut(fill(a.Pix)), stride(a.Stride)
		m = a
	case "RGBA64":
		a := image.NewRGBA64(r)
		a.Pix, a.Stride = cut(fill(a.Pix)), stride(a.Stride)
		m = a
	case "NRGBA64":
		a := image.NewNRGBA64(r)
		a.Pix, a.Stride = cut(fill(a.Pix)), stride(a.Stride)
		m = a
	case "CMYK":
		a := image.NewCMYK(r)
		a.Pix, a.Stride = cut(fill(a.Pix)), stride(a.Stride)
		m = a
	case "Paletted", "PalettedOdd":
		var pal color.Palette
		n := 1 + rng.Intn(20)
		for i := 0; i < n; i++ {
			switch {
			case kind == "Paletted":
				pal = append(pal, palette.Plan9[rng.Intn(256)])
			case rng.Intn(6) == 0:
				pal = append(pal, nil)
			default:
				pal = append(pal, encTestColor{uint8(rng.Intn(256))})
			}
		}
		if rng.Intn(10) == 0 {
			pal = nil
		}
		a := image.NewPaletted(r, pal)
		for i := range a.Pix {
			a.Pix[i] = uint8(rng.Intn(n))
			if rng.Intn(60) == 0 {
				a.Pix[i] = uint8(rng.Intn(256))
			}
		}
		if rng.Intn(3) == 0 {
			a.Pix, a.Stride = cut(a.Pix), stride(a.Stride)
		}
		m = a
	case "YCbCr", "NYCbCrA":
		ratio := encRatios[rng.Intn(len(encRatios))]
		var y *image.YCbCr
		var ya *image.NYCbCrA
		func() {
			defer func() { recover() }()
			if kind == "YCbCr" {
				y = image.NewYCbCr(r, ratio)
			} else {
				ya = image.NewNYCbCrA(r, ratio)
				y = &ya.YCbCr
			}
		}()
		if y == nil {
			return encMalformed(rng)
		}
		fill(y.Y)
		fill(y.Cb)
		fill(y.Cr)
		for i := rng.Intn(4); i >= 0; i-- {
			switch rng.Intn(5) {
			case 0:
				y.Y = cut(y.Y)
			case 1:
				y.Cb = cut(y.Cb)
			case 2:
				y.Cr = cut(y.Cr)
			case 3:
				y.YStride = stride(y.YStride)
			case 4:
				y.CStride = stride(y.CStride)
			}
		}
		m = y
		if ya != nil {
			fill(ya.A)
			if rng.Intn(3) == 0 {
				ya.A, ya.AStride = cut(ya.A), stride(ya.AStride)
			}
			m = ya
		}
	case "At":
		var pts []image.Point
		for i := rng.Intn(4); i >= 0; i-- {
			pts = append(pts, image.Pt(r.Min.X+rng.Intn(r.Dx()), r.Min.Y+rng.Intn(r.Dy())))
		}
		return fmt.Sprintf("At/%v/%v", r, pts), encBad(r, pts...)
	}
	return fmt.Sprintf("%s/%v", kind, r), encStatic(m)
}

// TestEncRegressMalformedRandom compares the panic values on random
// malformed images of every kind.
func TestEncRegressMalformedRandom(t *testing.T) {
	n := 6000
	if testing.Short() {
		n = 1500
	}
	rng := rand.New(rand.NewSource(11))
	panics := 0
	for i := 0; i < n; i++ {
		name, mk := encMalformed(rng)
		if encStrictCompare(t, fmt.Sprintf("%d/%s", i, name), mk, encQ(1+rng.Intn(100))) {
			panics++
		}
		if t.Failed() && i > 50 {
			break
		}
	}
	if panics < n/3 {
		t.Errorf("only %d of %d images panic", panics, n)
	}
	t.Logf("%d of %d images panic", panics, n)
}

// FuzzEncStrict compares the panic values on random malformed images.
func FuzzEncStrict(f *testing.F) {
	for i := int64(0); i < 8; i++ {
		f.Add(i)
	}
	f.Fuzz(func(t *testing.T, seed int64) {
		rng := rand.New(rand.NewSource(seed))
		name, mk := encMalformed(rng)
		encStrictCompare(t, name, mk, encQ(1+rng.Intn(100)))
	})
}

// TestEncRegressAtOrder checks that At is called once per pixel, in the
// order of image/jpeg's first call for each pixel, and that the output is
// the same.
func TestEncRegressAtOrder(t *testing.T) {
	sizes := []image.Rectangle{image.Rect(0, 0, 1, 1), image.Rect(0, 0, 9, 2), image.Rect(-5, 3, 12, 20), image.Rect(0, 0, 16, 16), image.Rect(7, -9, 40, 24), image.Rect(0, 0, 100, 75)}
	for _, r := range sizes {
		got := &encBadAt{r: r, log: true}
		want := &encBadAt{r: r, log: true}
		var gb, wb bytes.Buffer
		if err := Encode(&gb, got, nil); err != nil {
			t.Fatal(err)
		}
		if err := std.Encode(&wb, want, nil); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(gb.Bytes(), wb.Bytes()) {
			t.Errorf("%v: output differs", r)
		}
		seen := map[image.Point]bool{}
		var first []image.Point
		for _, p := range want.calls {
			if !seen[p] {
				seen[p] = true
				first = append(first, p)
			}
		}
		if fmt.Sprint(got.calls) != fmt.Sprint(first) {
			t.Errorf("%v: At called %d times in a different order from image/jpeg's first calls (%d)", r, len(got.calls), len(first))
		}
	}
}

// TestEncRegressCustomPalette: a palette of colors of unknown types is read
// through At, in image/jpeg's order, with the same result.
func TestEncRegressCustomPalette(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for _, r := range []image.Rectangle{image.Rect(0, 0, 1, 1), image.Rect(3, 1, 20, 18), image.Rect(0, 0, 64, 48)} {
		pal := color.Palette{encTestColor{1}, color.Gray{9}, encTestColor{200}, color.RGBA{1, 2, 3, 4}}
		m := image.NewPaletted(r, pal)
		for i := range m.Pix {
			m.Pix[i] = uint8(rng.Intn(len(pal)))
		}
		encStrictCompare(t, fmt.Sprint(r), encStatic(m), nil)
		// Short palettes, standard colors.
		m = image.NewPaletted(r, palette.Plan9[:16])
		for i := range m.Pix {
			m.Pix[i] = uint8(rng.Intn(16))
		}
		encStrictCompare(t, fmt.Sprint(r)+"/16", encStatic(m), nil)
	}
}

// encOverflowCases are rectangles near the int limits.
func encOverflowCases() map[string]func() image.Image {
	const maxI, minI = math.MaxInt, math.MinInt
	cases := map[string]func() image.Image{}
	at := func(r image.Rectangle, limit int) func() image.Image {
		return func() image.Image { return &encBadAt{r: r, limit: limit} }
	}
	// Min.X > Max.X with a positive (wrapped) Dx: image/jpeg runs no block.
	xw := image.Rectangle{image.Pt(maxI-5, 0), image.Pt(minI+10, 1)}
	cases["xwrap/RGBA"] = encStatic(&image.RGBA{Pix: make([]byte, 64), Stride: 64, Rect: xw})
	cases["xwrap/Gray"] = encStatic(&image.Gray{Pix: make([]byte, 16), Stride: 16, Rect: xw})
	cases["xwrap/At"] = at(xw, 1000)
	yw := image.Rectangle{image.Pt(0, maxI-5), image.Pt(8, minI)}
	cases["ywrap/RGBA"] = encStatic(&image.RGBA{Pix: make([]byte, 192), Stride: 32, Rect: yw})
	cases["ywrap/Gray"] = encStatic(&image.Gray{Pix: make([]byte, 48), Stride: 8, Rect: yw})
	cases["ywrap/At"] = at(yw, 1000)
	// Min < Max, but Dx or Dy overflows to a negative value.
	cases["negDx/At"] = at(image.Rect(minI, 0, 100, 1), 100000)
	cases["negDx/Gray"] = encStatic(&image.Gray{Pix: make([]byte, 8), Stride: 8, Rect: image.Rectangle{image.Pt(minI, 0), image.Pt(maxI, 1)}})
	cases["negDy/RGBA"] = encStatic(&image.RGBA{Pix: make([]byte, 400), Stride: 4, Rect: image.Rectangle{image.Pt(0, minI), image.Pt(1, maxI)}})
	cases["inf/At"] = at(image.Rectangle{image.Pt(minI, minI), image.Pt(maxI, maxI)}, 100000)
	// image/jpeg's x += 16 (or 8) and p.X+i overflow.
	cases["xmax/RGBA"] = encStatic(image.NewRGBA(image.Rect(maxI-10, 0, maxI, 2)))
	cases["xmax/Gray"] = encStatic(image.NewGray(image.Rect(maxI-1, 0, maxI, 1)))
	cases["xmax/NRGBA-At"] = func() image.Image {
		return &encCountLimit{Image: image.NewNRGBA(image.Rect(maxI-3, 0, maxI, 2)), limit: 1 << 16}
	}
	cases["ymax/RGBA"] = encStatic(image.NewRGBA(image.Rect(0, maxI-10, 8, maxI)))
	cases["ymax/At"] = at(image.Rect(0, maxI-10, 8, maxI), 100000)
	// Near the limits without overflow.
	for _, r := range []image.Rectangle{image.Rect(maxI-30, 0, maxI-14, 3), image.Rect(maxI-40, maxI-20, maxI-15, maxI-15), image.Rect(minI, minI, minI+19, minI+21)} {
		cases[fmt.Sprintf("near/RGBA/%v", r)] = encStatic(imageMakers[0].make(r, fills(4)[0].f))
		cases[fmt.Sprintf("near/NRGBA/%v", r)] = encStatic(imageMakers[2].make(r, fills(4)[1].f))
		cases[fmt.Sprintf("near/Gray/%v", r)] = encStatic(imageMakers[5].make(r, fills(4)[0].f))
		cases[fmt.Sprintf("near/At/%v", r)] = func() image.Image {
			return &encCountLimit{Image: &encBadAt{r: r}, limit: 1 << 16}
		}
	}
	return cases
}

// encCountLimit panics after limit calls of At.
type encCountLimit struct {
	image.Image
	n, limit int
}

func (m *encCountLimit) At(x, y int) color.Color {
	if m.n++; m.n > m.limit {
		panic("encCountLimit: limit")
	}
	return m.Image.At(x, y)
}

// TestEncRegressOverflow: rectangles near the int limits.
func TestEncRegressOverflow(t *testing.T) {
	for name, mk := range encOverflowCases() {
		encStrictCompare(t, name, mk, nil)
	}
}

// TestEncRegressOverflowScan encodes small images in the last and first
// columns and rows of the int range.
func TestEncRegressOverflowScan(t *testing.T) {
	const maxI, minI = math.MaxInt, math.MinInt
	type mk func(r image.Rectangle) image.Image
	kinds := map[string]mk{
		"Gray": func(r image.Rectangle) image.Image { return image.NewGray(r) },
		"RGBA": func(r image.Rectangle) image.Image { return image.NewRGBA(r) },
		"YCbCr444": func(r image.Rectangle) image.Image {
			return image.NewYCbCr(r, image.YCbCrSubsampleRatio444)
		},
		"YCbCr420": func(r image.Rectangle) image.Image {
			return image.NewYCbCr(r, image.YCbCrSubsampleRatio420)
		},
	}
	step := 1
	if testing.Short() {
		step = 3
	}
	for name, f := range kinds {
		for off := 0; off <= 40; off += step {
			for w := 1; w <= 40; w += step {
				rs := []image.Rectangle{
					image.Rect(maxI-off-w, 0, maxI-off, 3),
					image.Rect(0, maxI-off-w, 3, maxI-off),
					image.Rect(minI+off, 1, minI+off+w, 2),
				}
				for _, r := range rs {
					var m image.Image
					func() {
						defer func() { recover() }() // NewYCbCr overflows near MaxInt
						m = f(r)
					}()
					if m == nil {
						continue
					}
					encStrictCompare(t, fmt.Sprintf("%s/%v", name, r), encStatic(m), nil)
				}
			}
		}
	}
}

// Benchmarks of the paths that read pixels in image/jpeg's order.

func BenchmarkEncOrder(b *testing.B) {
	r := image.Rect(0, 0, 1024, 768)
	fill := fills(1)[0].f
	nrgba := image.NewNRGBA(r)
	setAll(nrgba, fill)
	rng := rand.New(rand.NewSource(1))
	pal16 := image.NewPaletted(r, palette.Plan9[:16])
	custom := image.NewPaletted(r, color.Palette{encTestColor{1}, encTestColor{77}, encTestColor{200}, color.Gray{9}})
	for i := range pal16.Pix {
		pal16.Pix[i] = uint8(rng.Intn(16))
		custom.Pix[i] = uint8(rng.Intn(4))
	}
	inputs := []struct {
		name string
		m    image.Image
	}{
		{"At", wrapped{nrgba}},
		{"Paletted16", pal16},
		{"PalettedCustom", custom},
		{"Paletted256", imageMakers[9].make(r, fill)},
	}
	for _, in := range inputs {
		b.Run("img="+in.name, func(b *testing.B) {
			b.SetBytes(int64(r.Dx() * r.Dy()))
			var buf bytes.Buffer
			for b.Loop() {
				buf.Reset()
				if err := Encode(&buf, in.m, nil); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
