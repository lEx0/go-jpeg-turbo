// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jpeg

import (
	"image"
	"image/color"
)

// image/jpeg's pixel readers and scan loop, for the images whose pixel
// reads can panic or have side effects. image/jpeg reads the pixels MCU by
// MCU (16x16, left to right, top to bottom), each MCU as four 8x8 blocks
// (top left, top right, bottom left, bottom right), each block row by row,
// and for an *image.YCbCr the Y, Cb and Cr samples of a pixel one after the
// other. When several pixels are bad, the first one read in that order is
// the one whose panic escapes, so the encoder reads such images in the same
// order:
//
//   - a malformed *image.RGBA, *image.YCbCr (encStripStd) or *image.Gray
//     (encGrayStd) is read with image/jpeg's own readers below, whose index
//     expressions also fail in the same way on the same pixel;
//   - any other image that is not pure is read with At, in the order of
//     image/jpeg's first read of every pixel (encStripAt);
//   - when image/jpeg's coordinate arithmetic overflows, its whole scan loop
//     runs (encScanStd).
//
// The samples go to the planes of the strip pipeline (or straight to
// encQuant), so the bytes are the same as image/jpeg's: these are its own
// computations.

// toYCbCr converts the 8x8 region of m whose top-left corner is p to its
// YCbCr values.
func toYCbCr(m image.Image, p image.Point, yBlock, cbBlock, crBlock *block) {
	b := m.Bounds()
	xmax := b.Max.X - 1
	ymax := b.Max.Y - 1
	for j := 0; j < 8; j++ {
		for i := 0; i < 8; i++ {
			r, g, b, _ := m.At(min(p.X+i, xmax), min(p.Y+j, ymax)).RGBA()
			yy, cb, cr := color.RGBToYCbCr(uint8(r>>8), uint8(g>>8), uint8(b>>8))
			yBlock[8*j+i] = int32(yy)
			cbBlock[8*j+i] = int32(cb)
			crBlock[8*j+i] = int32(cr)
		}
	}
}

// grayToY stores the 8x8 region of m whose top-left corner is p in yBlock.
func grayToY(m *image.Gray, p image.Point, yBlock *block) {
	b := m.Bounds()
	xmax := b.Max.X - 1
	ymax := b.Max.Y - 1
	pix := m.Pix
	for j := 0; j < 8; j++ {
		for i := 0; i < 8; i++ {
			idx := m.PixOffset(min(p.X+i, xmax), min(p.Y+j, ymax))
			yBlock[8*j+i] = int32(pix[idx])
		}
	}
}

// rgbaToYCbCr is a specialized version of toYCbCr for image.RGBA images.
func rgbaToYCbCr(m *image.RGBA, p image.Point, yBlock, cbBlock, crBlock *block) {
	b := m.Bounds()
	xmax := b.Max.X - 1
	ymax := b.Max.Y - 1
	for j := 0; j < 8; j++ {
		sj := p.Y + j
		if sj > ymax {
			sj = ymax
		}
		offset := (sj-b.Min.Y)*m.Stride - b.Min.X*4
		for i := 0; i < 8; i++ {
			sx := p.X + i
			if sx > xmax {
				sx = xmax
			}
			pix := m.Pix[offset+sx*4:]
			yy, cb, cr := color.RGBToYCbCr(pix[0], pix[1], pix[2])
			yBlock[8*j+i] = int32(yy)
			cbBlock[8*j+i] = int32(cb)
			crBlock[8*j+i] = int32(cr)
		}
	}
}

// yCbCrToYCbCr is a specialized version of toYCbCr for image.YCbCr images.
func yCbCrToYCbCr(m *image.YCbCr, p image.Point, yBlock, cbBlock, crBlock *block) {
	b := m.Bounds()
	xmax := b.Max.X - 1
	ymax := b.Max.Y - 1
	for j := 0; j < 8; j++ {
		sy := p.Y + j
		if sy > ymax {
			sy = ymax
		}
		for i := 0; i < 8; i++ {
			sx := p.X + i
			if sx > xmax {
				sx = xmax
			}
			yi := m.YOffset(sx, sy)
			ci := m.COffset(sx, sy)
			yBlock[8*j+i] = int32(m.Y[yi])
			cbBlock[8*j+i] = int32(m.Cb[ci])
			crBlock[8*j+i] = int32(m.Cr[ci])
		}
	}
}

// scale scales the 16x16 region represented by the 4 src blocks to the 8x8
// dst block.
func scale(dst *block, src *[4]block) {
	for i := 0; i < 4; i++ {
		dstOff := (i&2)<<4 | (i&1)<<2
		for y := 0; y < 4; y++ {
			for x := 0; x < 4; x++ {
				j := 16*y + 2*x
				sum := src[i][j] + src[i][j+1] + src[i][j+8] + src[i][j+9]
				dst[8*y+x+dstOff] = (sum + 2) >> 2
			}
		}
	}
}

// encScanStd is image/jpeg's writeSOS loop, for the rectangles where Dx or
// Dy overflows, or where its coordinates (x += 16, p.X+i) would: the
// strip pipeline counts columns and rows instead, so only these loops read
// the pixels image/jpeg reads (or panic, or call At without end, as it
// does). Each block goes through encQuant and encCode.
func (e *encoder) encScanStd(m image.Image) {
	var (
		// Scratch buffers to hold the YCbCr values.
		// The blocks are in natural (not zig-zag) order.
		b      block
		cb, cr [4]block
	)
	bounds := m.Bounds()
	switch m := m.(type) {
	case *image.Gray:
		for y := bounds.Min.Y; y < bounds.Max.Y; y += 8 {
			for x := bounds.Min.X; x < bounds.Max.X; x += 8 {
				p := image.Pt(x, y)
				grayToY(m, p, &b)
				e.encStdQuant(&b, 0)
				e.encStdCode()
			}
		}
	default:
		rgba, _ := m.(*image.RGBA)
		ycbcr, _ := m.(*image.YCbCr)
		for y := bounds.Min.Y; y < bounds.Max.Y; y += 16 {
			for x := bounds.Min.X; x < bounds.Max.X; x += 16 {
				for i := 0; i < 4; i++ {
					xOff := (i & 1) * 8
					yOff := (i & 2) * 4
					p := image.Pt(x+xOff, y+yOff)
					if rgba != nil {
						rgbaToYCbCr(rgba, p, &b, &cb[i], &cr[i])
					} else if ycbcr != nil {
						yCbCrToYCbCr(ycbcr, p, &b, &cb[i], &cr[i])
					} else {
						toYCbCr(m, p, &b, &cb[i], &cr[i])
					}
					e.encStdQuant(&b, 0)
				}
				scale(&b, &cb)
				e.encStdQuant(&b, 1)
				scale(&b, &cr)
				e.encStdQuant(&b, 2)
				e.encStdCode()
			}
		}
	}
}

// encStdQuant queues the block b of samples (0..255) of component comp.
func (e *encoder) encStdQuant(b *block, comp uint8) {
	var s [blockSize]byte
	for i, v := range b {
		s[i] = uint8(v)
	}
	e.encQuant(s[:], 8, comp)
}

// encStdCode codes the queued blocks, unless writing has failed (image/jpeg
// keeps reading the pixels then, but writes nothing).
func (e *encoder) encStdCode() {
	if e.err != nil {
		e.nb = 0
		return
	}
	if cap(e.out)-len(e.out) < encOutReserve {
		e.drain(false)
	}
	e.encCode()
}

// encPutBlock stores the samples of b (0..255) as the 8x8 block at dst of a
// plane of the given stride.
func encPutBlock(dst []byte, stride int, b *block) {
	for j := 0; j < 8; j++ {
		row := dst[j*stride : j*stride+8 : j*stride+8]
		for i := range row {
			row[i] = uint8(b[8*j+i])
		}
	}
}

// encStripStd fills the planes of the strip at image row y of a malformed
// *image.RGBA or *image.YCbCr as image/jpeg's writeSOS reads it.
func (e *encoder) encStripStd(y int) {
	rgba, _ := e.m.(*image.RGBA)
	ycbcr, _ := e.m.(*image.YCbCr)
	pw, cw := e.pw, e.pw>>1
	var b, c block
	var cb, cr [4]block
	for mx := 0; mx < pw>>4; mx++ {
		x := e.rect.Min.X + mx<<4
		for i := 0; i < 4; i++ {
			xOff := (i & 1) * 8
			yOff := (i & 2) * 4
			p := image.Pt(x+xOff, y+yOff)
			if rgba != nil {
				rgbaToYCbCr(rgba, p, &b, &cb[i], &cr[i])
			} else {
				yCbCrToYCbCr(ycbcr, p, &b, &cb[i], &cr[i])
			}
			encPutBlock(e.yp[yOff*pw+mx<<4+xOff:], pw, &b)
		}
		scale(&c, &cb)
		encPutBlock(e.cbp[mx<<3:], cw, &c)
		scale(&c, &cr)
		encPutBlock(e.crp[mx<<3:], cw, &c)
	}
}

// encGrayStd fills the plane of the strip at image row y of a malformed
// *image.Gray as image/jpeg's writeSOS reads it.
func (e *encoder) encGrayStd(m *image.Gray, y int) {
	var b block
	for bx := 0; bx < e.pw>>3; bx++ {
		grayToY(m, image.Pt(e.rect.Min.X+bx<<3, y), &b)
		encPutBlock(e.yp[bx<<3:], e.pw, &b)
	}
}

// encStripAt reads the pixels of the strip at image row y with At, as
// RGBA8, into e.strip (16 rows of 4*width bytes), in the order of
// image/jpeg's first read of each pixel: MCU by MCU, its four blocks, each
// row by row. image/jpeg reads the pixels of the last column and row again
// for the positions past the image; every such read comes after the first
// read of the same pixel, so At is called once per pixel and, if it
// panics, at the pixel where image/jpeg's first panicking call is.
func (e *encoder) encStripAt(y int) {
	m := e.m
	w, minX := e.width, e.rect.Min.X
	rows := min(16, e.rect.Max.Y-y)
	n := 4 * w
	e.strip = encGrow(e.strip, 16*n)
	e.stripY = y
	s := e.strip
	for x := 0; x < w; x += 16 {
		for i := 0; i < 4; i++ {
			bx, by := x+(i&1)*8, (i&2)*4
			bw, bh := min(8, w-bx), min(8, rows-by)
			if bw <= 0 || bh <= 0 {
				continue
			}
			for j := by; j < by+bh; j++ {
				encAtRun(m, s[j*n+4*bx:j*n+4*(bx+bw)], minX+bx, y+j)
			}
		}
	}
}

// encAtRun reads the len(d)/4 pixels of image row y from column x on with
// At, as RGBA8, into d. It is a function of its own so that the loop keeps
// few values live across the calls of At and RGBA.
//
//go:noinline
func encAtRun(m image.Image, d []byte, x, y int) {
	for k := 0; k+4 <= len(d); k += 4 {
		r, g, b, a := m.At(x, y).RGBA()
		x++
		p := d[k : k+4 : k+4]
		p[0], p[1], p[2], p[3] = uint8(r>>8), uint8(g>>8), uint8(b>>8), uint8(a>>8)
	}
}
