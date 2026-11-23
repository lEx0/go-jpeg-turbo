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
	"image"
	"image/color"
	"math"

	"github.com/lEx0/go-jpeg-turbo/internal/ycc"
)

// The strip pipeline. image/jpeg converts every 8x8 block of the image
// separately: it reads the pixel of every position of the MCU, clamping the
// coordinates to the last column and row of the image (edge replication),
// converts it to YCbCr with color.RGBToYCbCr (except for *image.YCbCr and
// *image.Gray, which it reads as they are), and averages the full-resolution
// chroma 2x2 ((sum+2)>>2) for 4:2:0. The encoder computes the same samples,
// but converts a strip of 16 rows (8 for gray) at a time into padded planes,
// so that every block is read straight from a plane by dct.FDCTQuant:
//
//	yp:       the luma of 16 rows, pw (a multiple of 16) samples per row;
//	cbp, crp: the 2x2-averaged chroma of the same rows, 8 rows of pw/2.
//
// Columns and rows past the image repeat the last pixel, which is the same
// as converting the repeated pixel: the conversion is per pixel.

// encSrcKind is how the pixels of a color image are read.
type encSrcKind uint8

const (
	encSrcRGBA     encSrcKind = iota // *image.RGBA: rows of Pix
	encSrcNRGBA                      // *image.NRGBA: rows of Pix
	encSrcRows                       // rows converted to RGBA8 by encRow
	encSrcGray                       // *image.Gray16, *image.Alpha, *image.Alpha16: Y only
	encSrcYCbCr444                   // *image.YCbCr, 4:4:4
	encSrcYCbCr420                   // *image.YCbCr, 4:2:0, Rect.Min even and >= 0
	encSrcYCbCr422                   // *image.YCbCr, 4:2:2, Rect.Min.X even and >= 0
	encSrcYCbCr                      // any other *image.YCbCr
)

// encRowKind is how encRow reads a row of pixels (encSrcRows). Every kind
// but encRowAt and encRowRGBA reads the pixels the way the image's At
// method does and calls the RGBA method of the same color value, without
// the interface conversion; they are used only when every row lies within
// Pix, so that At would not panic either. The images of the other two kinds
// are never pure: encColor reads them in image/jpeg's order (encode_std.go).
type encRowKind uint8

const (
	encRowAt       encRowKind = iota // m.At(x, y).RGBA(), as image/jpeg (encStripAt)
	encRowRGBA                       // *image.RGBA, image/jpeg's rgbaToYCbCr (encStripStd)
	encRowRGBA64                     // *image.RGBA64
	encRowNRGBA64                    // *image.NRGBA64
	encRowGray16                     // *image.Gray16
	encRowCMYK                       // *image.CMYK
	encRowAlpha                      // *image.Alpha
	encRowAlpha16                    // *image.Alpha16
	encRowPaletted                   // *image.Paletted with a palette of standard colors
	encRowNYCbCrA                    // *image.NYCbCrA
)

// encStrip is the pixel stage of the encoder.
type encStrip struct {
	m       image.Image
	rect    image.Rectangle
	width   int // rect.Dx() > 0
	height  int // rect.Dy() > 0
	pw      int // padded width: of yp, and twice that of cbp and crp
	kind    encSrcKind
	rowKind encRowKind
	// pure reports that reading the pixels can neither panic nor have
	// side effects, so that it can stop after a write error. image/jpeg
	// keeps reading (and so keeps calling At, or panics on a malformed
	// image) until the end. When an image is not pure, its pixels are read
	// in image/jpeg's order (encode_std.go), so that the first pixel to
	// fail, and so the panic value, is the same.
	pure bool
	// strip holds the RGBA8 rows of the strip at image row stripY, read by
	// encStripAt; encPair uses them instead of encRow if useStrip.
	strip    []byte
	stripY   int
	useStrip bool
	// palBad is set by encRow (encRowPaletted) on a pixel past the palette.
	palBad bool

	yp, cbp, crp []byte
	// row are RGBA8 rows (encSrcRows).
	row [2][]byte
	// full are full-resolution chroma rows (encSrcYCbCr): Cb and Cr of the
	// two rows of a pair.
	full [4][]byte
	// ccol[x] is the offset of the chroma sample of column rect.Min.X+x
	// from that of column rect.Min.X (encSrcYCbCr, encRowNYCbCrA).
	ccol []int
	// pal is the palette as RGBA8 (encRowPaletted), palN its length.
	pal  [256][4]byte
	palN int

	// Scratch space for the last pixels of a row.
	tail [2][8]byte
	ty   [2][2]byte
	tc   [2][1]byte
}

func (s *encStrip) release() {
	s.m = nil
}

// encGrow returns a slice of length n, reusing buf when it is big enough.
func encGrow(buf []byte, n int) []byte {
	if cap(buf) < n {
		return make([]byte, n)
	}
	return buf[:n]
}

// encFill sets every byte of dst to v.
func encFill(dst []byte, v byte) {
	for i := range dst {
		dst[i] = v
	}
}

// encRowsOK reports whether rows of rowBytes bytes at offsets 0, stride,
// ..., (rows-1)*stride all lie within a buffer of pixLen bytes. It is
// written so that it cannot overflow.
func encRowsOK(pixLen, stride, rowBytes, rows int) bool {
	if rowBytes > pixLen {
		return false
	}
	if rows > 1 && (stride < 0 || stride > (pixLen-rowBytes)/(rows-1)) {
		return false
	}
	return true
}

// encCX and encCY are the column and row parts of image.YCbCr.COffset:
// COffset(x, y) = (encCY(y)-encCY(Min.Y))*CStride + encCX(x)-encCX(Min.X).
func encCX(r image.YCbCrSubsampleRatio, x int) int {
	switch r {
	case image.YCbCrSubsampleRatio422, image.YCbCrSubsampleRatio420:
		return x / 2
	case image.YCbCrSubsampleRatio411, image.YCbCrSubsampleRatio410:
		return x / 4
	}
	return x
}

func encCY(r image.YCbCrSubsampleRatio, y int) int {
	switch r {
	case image.YCbCrSubsampleRatio420, image.YCbCrSubsampleRatio440, image.YCbCrSubsampleRatio410:
		return y / 2
	}
	return y
}

// encChromaOK reports whether every chroma row of m that the pixels of m
// use lies within Cb and Cr, and sets up s.ccol.
func (s *encStrip) encChromaSetup(m *image.YCbCr) bool {
	r, b := m.SubsampleRatio, m.Rect
	if cap(s.ccol) < s.width {
		s.ccol = make([]int, s.width)
	}
	s.ccol = s.ccol[:s.width]
	c0 := encCX(r, b.Min.X)
	for x := range s.ccol {
		s.ccol[x] = encCX(r, b.Min.X+x) - c0
	}
	cw := s.ccol[s.width-1] + 1
	ch := encCY(r, b.Max.Y-1) - encCY(r, b.Min.Y) + 1
	return encRowsOK(len(m.Cb), m.CStride, cw, ch) && encRowsOK(len(m.Cr), m.CStride, cw, ch)
}

// encScan writes the entropy-coded segment of m (whose bounds are b),
// without the final padding.
func (e *encoder) encScan(m image.Image, b image.Rectangle) {
	e.dc = [3]int32{}
	e.nb = 0
	if b.Empty() {
		// image/jpeg's loops run no block (and read no pixel), whatever
		// Dx and Dy are.
		return
	}
	w, h := b.Dx(), b.Dy()
	if w <= 0 || h <= 0 || b.Max.X > math.MaxInt-15 || b.Max.Y > math.MaxInt-15 {
		// Dx or Dy overflowed, or image/jpeg's x += 16 (or 8) and p.X+i
		// would: only its own coordinate loops read what it reads. Below,
		// Min < Max and every coordinate up to Max+15 is an int.
		e.encScanStd(m)
		return
	}
	e.m, e.rect, e.width, e.height = m, b, w, h
	if g, ok := m.(*image.Gray); ok {
		e.encGray(g)
		return
	}
	e.encSetup(m)
	e.encColor()
}

// encSetup chooses how to read the pixels of a color image.
func (e *encoder) encSetup(m image.Image) {
	w, h := e.width, e.height
	e.kind, e.rowKind, e.pure = encSrcRows, encRowAt, false
	// rows selects kind and rowKind if every row of the image lies within
	// its Pix.
	rows := func(pixLen, stride, bpp int, kind encSrcKind, rowKind encRowKind) {
		if encRowsOK(pixLen, stride, bpp*w, h) {
			e.kind, e.rowKind, e.pure = kind, rowKind, true
		}
	}
	switch m := m.(type) {
	case *image.RGBA:
		e.rowKind = encRowRGBA
		rows(len(m.Pix), m.Stride, 4, encSrcRGBA, encRowRGBA)
	case *image.NRGBA:
		rows(len(m.Pix), m.Stride, 4, encSrcNRGBA, encRowAt)
	case *image.YCbCr:
		e.encSetupYCbCr(m)
	case *image.RGBA64:
		rows(len(m.Pix), m.Stride, 8, encSrcRows, encRowRGBA64)
	case *image.NRGBA64:
		rows(len(m.Pix), m.Stride, 8, encSrcRows, encRowNRGBA64)
	case *image.CMYK:
		rows(len(m.Pix), m.Stride, 4, encSrcRows, encRowCMYK)
	// The RGBA of color.Gray16, color.Alpha and color.Alpha16 is v, v, v
	// (and v or 0xffff), so their RGBA8 is v, v, v (v the high byte), which
	// color.RGBToYCbCr maps to v, 128, 128: encSrcGray. encRow converts
	// them too (when the rows are not all within Pix, encRowAt is used).
	case *image.Gray16:
		rows(len(m.Pix), m.Stride, 2, encSrcGray, encRowGray16)
	case *image.Alpha:
		rows(len(m.Pix), m.Stride, 1, encSrcGray, encRowAlpha)
	case *image.Alpha16:
		rows(len(m.Pix), m.Stride, 2, encSrcGray, encRowAlpha16)
	case *image.Paletted:
		if encRowsOK(len(m.Pix), m.Stride, w, h) && e.encSetupPalette(m.Palette) {
			e.rowKind = encRowPaletted
			// A pixel past the palette makes image/jpeg panic.
			e.pure = e.palN == len(e.pal)
		}
	case *image.NYCbCrA:
		// encRow indexes Y, Cb, Cr and A as NYCbCrAAt does. When a plane
		// is too short (not pure), encColor reads the pixels with At
		// instead, in image/jpeg's order.
		e.rowKind = encRowNYCbCrA
		cok := e.encChromaSetup(&m.YCbCr)
		e.pure = cok && encRowsOK(len(m.Y), m.YStride, w, h) && encRowsOK(len(m.A), m.AStride, w, h)
	}
}

// encSetupPalette converts p to RGBA8 if all its colors are of the
// standard color types, whose RGBA methods are pure functions.
func (e *encoder) encSetupPalette(p color.Palette) bool {
	p = p[:min(len(p), len(e.pal))]
	for i, c := range p {
		switch c.(type) {
		case color.RGBA, color.RGBA64, color.NRGBA, color.NRGBA64, color.Alpha, color.Alpha16,
			color.Gray, color.Gray16, color.CMYK, color.YCbCr, color.NYCbCrA:
		default:
			return false
		}
		r, g, b, a := c.RGBA()
		e.pal[i] = [4]byte{uint8(r >> 8), uint8(g >> 8), uint8(b >> 8), uint8(a >> 8)}
	}
	e.palN = len(p)
	return true
}

func (e *encoder) encSetupYCbCr(m *image.YCbCr) {
	b := m.Rect
	yok := encRowsOK(len(m.Y), m.YStride, e.width, e.height)
	cok := e.encChromaSetup(m)
	e.pure = yok && cok
	e.kind = encSrcYCbCr
	if !e.pure {
		return
	}
	switch m.SubsampleRatio {
	case image.YCbCrSubsampleRatio444:
		e.kind = encSrcYCbCr444
	case image.YCbCrSubsampleRatio420:
		// With even, non-negative Rect.Min, both pixels of every pair of
		// columns, and of rows, share a chroma sample.
		if b.Min.X >= 0 && b.Min.Y >= 0 && b.Min.X%2 == 0 && b.Min.Y%2 == 0 {
			e.kind = encSrcYCbCr420
		}
	case image.YCbCrSubsampleRatio422:
		if b.Min.X >= 0 && b.Min.X%2 == 0 {
			e.kind = encSrcYCbCr422
		}
	}
}

// encColor converts and codes the strips of a 3-component image.
func (e *encoder) encColor() {
	w := e.width
	mcuX := (w + 15) >> 4
	pw := mcuX << 4
	cw := pw >> 1
	e.pw = pw
	e.yp = encGrow(e.yp, 16*pw)
	e.cbp = encGrow(e.cbp, 8*cw)
	e.crp = encGrow(e.crp, 8*cw)
	switch e.kind {
	case encSrcRows:
		e.row[0] = encGrow(e.row[0], 4*w)
		e.row[1] = encGrow(e.row[1], 4*w)
	case encSrcYCbCr:
		for i := range e.full {
			e.full[i] = encGrow(e.full[i], pw)
		}
	}
	yp, cbp, crp := e.yp, e.cbp, e.crp
	maxY := e.rect.Max.Y - 1
	for y := e.rect.Min.Y; y <= maxY; y += 16 {
		if e.err != nil && e.pure {
			break
		}
		e.useStrip = false
		switch {
		case e.pure:
			e.encPairs(y)
		case e.kind == encSrcYCbCr || e.rowKind == encRowRGBA:
			// Rows of Pix (or of the planes) past their slices.
			e.encStripStd(y)
		case e.rowKind == encRowPaletted:
			// A palette of fewer than 256 colors.
			e.palBad = false
			e.encPairs(y)
			if e.palBad {
				// A pixel of the strip is past the palette: read the
				// strip again in image/jpeg's order, which panics at its
				// first such pixel.
				e.encStripAt(y)
				e.useStrip = true
				e.encPairs(y)
			}
		default:
			e.encStripAt(y)
			e.useStrip = true
			e.encPairs(y)
		}
		if e.err != nil {
			continue
		}
		for mx := 0; mx < mcuX; mx++ {
			if cap(e.out)-len(e.out) < encOutReserve {
				e.drain(false)
			}
			x := mx << 4
			e.encQuant(yp[x:], pw, 0)
			e.encQuant(yp[x+8:], pw, 0)
			e.encQuant(yp[8*pw+x:], pw, 0)
			e.encQuant(yp[8*pw+x+8:], pw, 0)
			e.encQuant(cbp[mx<<3:], cw, 1)
			e.encQuant(crp[mx<<3:], cw, 2)
			e.encCode()
		}
	}
}

// encPairs converts the rows of the strip at image row y, two by two.
func (e *encoder) encPairs(y int) {
	pw := e.pw
	cw := pw >> 1
	yp, cbp, crp := e.yp, e.cbp, e.crp
	maxY := e.rect.Max.Y - 1
	prevLast := false
	for k := 0; k < 8; k++ {
		y0 := yp[2*k*pw : (2*k+1)*pw]
		y1 := yp[(2*k+1)*pw : (2*k+2)*pw]
		cb := cbp[k*cw : (k+1)*cw]
		cr := crp[k*cw : (k+1)*cw]
		r0, r1 := min(y+2*k, maxY), min(y+2*k+1, maxY)
		last := r0 == maxY
		if last && prevLast {
			// Both rows of this pair and of the previous one are the
			// last row.
			copy(y0, yp[(2*k-1)*pw:2*k*pw])
			copy(y1, y0)
			copy(cb, cbp[(k-1)*cw:k*cw])
			copy(cr, crp[(k-1)*cw:k*cw])
			continue
		}
		prevLast = last
		e.encPair(r0, r1, y0, y1, cb, cr)
	}
}

// encPair converts the image rows r0 and r1 (r1 is r0+1, or r0 at the
// bottom edge) to the luma rows y0, y1 and the chroma rows cb, cr of a
// strip.
func (e *encoder) encPair(r0, r1 int, y0, y1, cb, cr []byte) {
	w := e.width
	switch e.kind {
	case encSrcRGBA:
		m := e.m.(*image.RGBA)
		o0 := (r0 - m.Rect.Min.Y) * m.Stride
		o1 := (r1 - m.Rect.Min.Y) * m.Stride
		e.enc420(m.Pix[o0:o0+4*w:o0+4*w], m.Pix[o1:o1+4*w:o1+4*w], false, y0, y1, cb, cr)
	case encSrcNRGBA:
		m := e.m.(*image.NRGBA)
		o0 := (r0 - m.Rect.Min.Y) * m.Stride
		o1 := (r1 - m.Rect.Min.Y) * m.Stride
		e.enc420(m.Pix[o0:o0+4*w:o0+4*w], m.Pix[o1:o1+4*w:o1+4*w], true, y0, y1, cb, cr)
	case encSrcRows:
		var p0, p1 []byte
		if e.useStrip {
			n := 4 * w
			o0, o1 := (r0-e.stripY)*n, (r1-e.stripY)*n
			p0, p1 = e.strip[o0:o0+n:o0+n], e.strip[o1:o1+n:o1+n]
		} else {
			p0 = e.encRow(0, r0)
			p1 = p0
			if r1 != r0 {
				p1 = e.encRow(1, r1)
			}
		}
		e.enc420(p0, p1, false, y0, y1, cb, cr)
	case encSrcGray:
		e.encGrayRow(r0, y0)
		if r1 != r0 {
			e.encGrayRow(r1, y1)
		} else {
			copy(y1, y0)
		}
		encFill(cb, 128)
		encFill(cr, 128)
	default:
		e.encPairYCbCr(r0, r1, y0, y1, cb, cr)
	}
}

// enc420 converts two RGBA8 (or NRGBA8) rows of w pixels.
func (e *encoder) enc420(p0, p1 []byte, nrgba bool, y0, y1, cb, cr []byte) {
	w, pw := e.width, e.pw
	conv := ycc.RGBAToYCbCr420
	if nrgba {
		conv = ycc.NRGBAToYCbCr420
	}
	n := w &^ 1
	if n > 0 {
		conv(p0, p1, y0, y1, cb, cr, n)
	}
	if n < pw {
		// Every column from n on holds the last pixel (column n, when w
		// is odd, is that pixel itself), so every chroma sample from n/2
		// on is the average of two copies of it in both rows.
		t := &e.tail
		copy(t[0][0:4], p0[4*w-4:4*w])
		copy(t[0][4:8], t[0][0:4])
		copy(t[1][0:4], p1[4*w-4:4*w])
		copy(t[1][4:8], t[1][0:4])
		conv(t[0][:], t[1][:], e.ty[0][:], e.ty[1][:], e.tc[0][:], e.tc[1][:], 2)
		encFill(y0[n:pw], e.ty[0][0])
		encFill(y1[n:pw], e.ty[1][0])
		encFill(cb[n/2:pw/2], e.tc[0][0])
		encFill(cr[n/2:pw/2], e.tc[1][0])
	}
}

// encYRow copies the luma of image row r to dst, padded to e.pw.
func (e *encoder) encYRow(m *image.YCbCr, r int, dst []byte) {
	w := e.width
	o := (r - m.Rect.Min.Y) * m.YStride
	copy(dst[:w], m.Y[o:o+w])
	encFill(dst[w:e.pw], dst[w-1])
}

// encCRow gathers the full-resolution chroma (of plane c of m) of image
// row r into dst, padded to e.pw.
func (e *encoder) encCRow(m *image.YCbCr, c []byte, r int, dst []byte) {
	w := e.width
	o := (encCY(m.SubsampleRatio, r) - encCY(m.SubsampleRatio, m.Rect.Min.Y)) * m.CStride
	ccol := e.ccol[:w]
	dst = dst[:e.pw]
	for x, cx := range ccol {
		dst[x] = c[o+cx]
	}
	encFill(dst[w:], dst[w-1])
}

func (e *encoder) encPairYCbCr(r0, r1 int, y0, y1, cb, cr []byte) {
	m := e.m.(*image.YCbCr)
	w, cw := e.width, e.pw>>1
	e.encYRow(m, r0, y0)
	if r1 != r0 {
		e.encYRow(m, r1, y1)
	} else {
		copy(y1, y0)
	}
	switch e.kind {
	case encSrcYCbCr444:
		o0 := (r0 - m.Rect.Min.Y) * m.CStride
		o1 := (r1 - m.Rect.Min.Y) * m.CStride
		n := w >> 1
		for i, c := range [2][]byte{m.Cb, m.Cr} {
			dst := cb
			if i == 1 {
				dst = cr
			}
			a, b := c[o0:o0+w:o0+w], c[o1:o1+w:o1+w]
			if n > 0 {
				ycc.Downsample2x2(a, b, dst, n)
			}
			if n < cw {
				v := 2*uint(a[w-1]) + 2*uint(b[w-1])
				encFill(dst[n:cw], uint8((v+2)>>2))
			}
		}
	case encSrcYCbCr420:
		// Rows r0 and r1 share a chroma row, columns 2i and 2i+1 a
		// sample: the average of 4 copies of a sample is the sample.
		o := (r0/2 - m.Rect.Min.Y/2) * m.CStride
		n := (w + 1) >> 1
		copy(cb[:n], m.Cb[o:o+n])
		copy(cr[:n], m.Cr[o:o+n])
		encFill(cb[n:cw], cb[n-1])
		encFill(cr[n:cw], cr[n-1])
	case encSrcYCbCr422:
		// Columns 2i and 2i+1 share a sample: (2a+2b+2)>>2 is the
		// rounded-up average of the samples a and b of the two rows.
		o0 := (r0 - m.Rect.Min.Y) * m.CStride
		o1 := (r1 - m.Rect.Min.Y) * m.CStride
		n := (w + 1) >> 1
		encAvg(cb[:n], m.Cb[o0:o0+n], m.Cb[o1:o1+n])
		encAvg(cr[:n], m.Cr[o0:o0+n], m.Cr[o1:o1+n])
		encFill(cb[n:cw], cb[n-1])
		encFill(cr[n:cw], cr[n-1])
	default:
		f := &e.full
		e.encCRow(m, m.Cb, r0, f[0])
		e.encCRow(m, m.Cr, r0, f[2])
		b0, b1 := f[0], f[2]
		if r1 != r0 {
			e.encCRow(m, m.Cb, r1, f[1])
			e.encCRow(m, m.Cr, r1, f[3])
			b0, b1 = f[1], f[3]
		}
		ycc.Downsample2x2(f[0], b0, cb, cw)
		ycc.Downsample2x2(f[2], b1, cr, cw)
	}
}

// encAvg sets dst[i] = (a[i]+b[i]+1)>>1, 8 bytes at a time.
func encAvg(dst, a, b []byte) {
	n := len(dst)
	a, b = a[:n], b[:n]
	i := 0
	for ; i+8 <= n; i += 8 {
		x := encLE64(a[i:])
		y := encLE64(b[i:])
		encPutLE64(dst[i:], (x|y)-((x^y)>>1)&0x7f7f7f7f7f7f7f7f)
	}
	for ; i < n; i++ {
		dst[i] = uint8((uint(a[i]) + uint(b[i]) + 1) >> 1)
	}
}

// encGrayRow copies the gray value (the high byte) of the pixels of image
// row y of an encSrcGray image to dst, padded to e.pw.
func (e *encoder) encGrayRow(y int, dst []byte) {
	var pix []byte
	var o, bpp int
	switch m := e.m.(type) {
	case *image.Gray16:
		pix, o, bpp = m.Pix, (y-m.Rect.Min.Y)*m.Stride, 2
	case *image.Alpha16:
		pix, o, bpp = m.Pix, (y-m.Rect.Min.Y)*m.Stride, 2
	case *image.Alpha:
		pix, o, bpp = m.Pix, (y-m.Rect.Min.Y)*m.Stride, 1
	}
	w := e.width
	s := pix[o : o+bpp*w : o+bpp*w]
	dst = dst[:e.pw]
	if bpp == 1 {
		copy(dst, s)
	} else {
		for x := range dst[:w] {
			dst[x] = s[2*x]
		}
	}
	encFill(dst[w:], dst[w-1])
}

// encRow converts image row y to RGBA8 in s.row[i] (A is not used). The
// rows of encRowAt and encRowRGBA images are read by encStripAt and
// encStripStd instead.
func (e *encoder) encRow(i int, y int) []byte {
	w := e.width
	dst := e.row[i][: 4*w : 4*w]
	switch e.rowKind {
	case encRowRGBA64:
		m := e.m.(*image.RGBA64)
		o := (y - m.Rect.Min.Y) * m.Stride
		s := m.Pix[o : o+8*w : o+8*w]
		for x := 0; x < w; x++ {
			// color.RGBA64.RGBA returns the channels: >>8 is the high
			// byte.
			p := s[8*x : 8*x+8 : 8*x+8]
			d := dst[4*x : 4*x+4 : 4*x+4]
			d[0], d[1], d[2], d[3] = p[0], p[2], p[4], p[6]
		}
	case encRowNRGBA64:
		m := e.m.(*image.NRGBA64)
		o := (y - m.Rect.Min.Y) * m.Stride
		s := m.Pix[o : o+8*w : o+8*w]
		for x := 0; x < w; x++ {
			p := s[8*x : 8*x+8 : 8*x+8]
			r, g, b, a := color.NRGBA64{
				uint16(p[0])<<8 | uint16(p[1]),
				uint16(p[2])<<8 | uint16(p[3]),
				uint16(p[4])<<8 | uint16(p[5]),
				uint16(p[6])<<8 | uint16(p[7]),
			}.RGBA()
			d := dst[4*x : 4*x+4 : 4*x+4]
			d[0], d[1], d[2], d[3] = uint8(r>>8), uint8(g>>8), uint8(b>>8), uint8(a>>8)
		}
	case encRowGray16, encRowAlpha16:
		// color.Gray16 and color.Alpha16: the value in R, G and B.
		m := e.m
		var pix []byte
		var o int
		if g, ok := m.(*image.Gray16); ok {
			pix, o = g.Pix, (y-g.Rect.Min.Y)*g.Stride
		} else {
			a := m.(*image.Alpha16)
			pix, o = a.Pix, (y-a.Rect.Min.Y)*a.Stride
		}
		s := pix[o : o+2*w : o+2*w]
		for x := 0; x < w; x++ {
			v := s[2*x]
			d := dst[4*x : 4*x+4 : 4*x+4]
			d[0], d[1], d[2], d[3] = v, v, v, 0xff
		}
	case encRowAlpha:
		m := e.m.(*image.Alpha)
		o := (y - m.Rect.Min.Y) * m.Stride
		s := m.Pix[o : o+w : o+w]
		for x, v := range s {
			d := dst[4*x : 4*x+4 : 4*x+4]
			d[0], d[1], d[2], d[3] = v, v, v, v
		}
	case encRowCMYK:
		m := e.m.(*image.CMYK)
		o := (y - m.Rect.Min.Y) * m.Stride
		s := m.Pix[o : o+4*w : o+4*w]
		for x := 0; x < w; x++ {
			p := s[4*x : 4*x+4 : 4*x+4]
			r, g, b, _ := color.CMYK{p[0], p[1], p[2], p[3]}.RGBA()
			d := dst[4*x : 4*x+4 : 4*x+4]
			d[0], d[1], d[2], d[3] = uint8(r>>8), uint8(g>>8), uint8(b>>8), 0xff
		}
	case encRowPaletted:
		m := e.m.(*image.Paletted)
		o := (y - m.Rect.Min.Y) * m.Stride
		s := m.Pix[o : o+w : o+w]
		for x, v := range s {
			d := dst[4*x : 4*x+4 : 4*x+4]
			if int(v) < e.palN {
				p := &e.pal[v]
				d[0], d[1], d[2], d[3] = p[0], p[1], p[2], p[3]
			} else {
				// Past the palette: At panics, as in image/jpeg, but
				// maybe at another pixel first; encColor reads the
				// strip again in image/jpeg's order.
				e.palBad = true
			}
		}
	case encRowNYCbCrA:
		m := e.m.(*image.NYCbCrA)
		yo := (y - m.Rect.Min.Y) * m.YStride
		ao := (y - m.Rect.Min.Y) * m.AStride
		co := (encCY(m.SubsampleRatio, y) - encCY(m.SubsampleRatio, m.Rect.Min.Y)) * m.CStride
		ccol := e.ccol[:w]
		for x, cx := range ccol {
			ci := co + cx
			r, g, b, a := color.NYCbCrA{
				YCbCr: color.YCbCr{Y: m.Y[yo+x], Cb: m.Cb[ci], Cr: m.Cr[ci]},
				A:     m.A[ao+x],
			}.RGBA()
			d := dst[4*x : 4*x+4 : 4*x+4]
			d[0], d[1], d[2], d[3] = uint8(r>>8), uint8(g>>8), uint8(b>>8), uint8(a>>8)
		}
	}
	return dst
}

// encGray converts and codes the strips of an *image.Gray (1 component,
// 8x8 blocks).
func (e *encoder) encGray(m *image.Gray) {
	w := e.width
	bx := (w + 7) >> 3
	pw := bx << 3
	e.pw = pw
	e.yp = encGrow(e.yp, 8*pw)
	yp := e.yp
	ok := encRowsOK(len(m.Pix), m.Stride, w, e.height)
	minY, maxY := m.Rect.Min.Y, e.rect.Max.Y-1
	for y := e.rect.Min.Y; y <= maxY; y += 8 {
		if e.err != nil && ok {
			break
		}
		if !ok {
			// The rows do not all lie within Pix: image/jpeg's grayToY,
			// block by block, panics at the same pixel.
			e.encGrayStd(m, y)
		}
		for j := 0; j < 8 && ok; j++ {
			row := yp[j*pw : (j+1)*pw]
			r := min(y+j, maxY)
			if j > 0 && y+j-1 >= maxY {
				copy(row, yp[(j-1)*pw:j*pw])
				continue
			}
			o := (r - minY) * m.Stride
			copy(row[:w], m.Pix[o:o+w])
			encFill(row[w:], row[w-1])
		}
		if e.err != nil {
			continue
		}
		for x := 0; x < pw; x += 8 {
			e.encQuant(yp[x:], pw, 0)
			if e.nb == encBatchMax || x+8 == pw {
				if cap(e.out)-len(e.out) < encOutReserve {
					e.drain(false)
				}
				e.encCode()
			}
		}
	}
}

func encLE64(b []byte) uint64 {
	_ = b[7]
	return uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 |
		uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
}

func encPutLE64(b []byte, v uint64) {
	_ = b[7]
	b[0], b[1], b[2], b[3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
	b[4], b[5], b[6], b[7] = byte(v>>32), byte(v>>40), byte(v>>48), byte(v>>56)
}
