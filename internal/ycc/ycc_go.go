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

import "encoding/binary"

// Portable implementations of the kernels. They are what the generic build
// (and -tags purego) runs, and what the assembly builds use for rows too
// short for a vector block. They produce exactly what the *Ref functions
// produce, with the following arithmetic, proved exhaustively by the tests:
//
//   - color.RGBToYCbCr's chroma is clamp(c>>16, 0, 255) of the pre-shift
//     sum c. With the +257<<15 bias the sum lies in [1<<16, 1<<24] for every
//     input, so only the upper clamp is ever taken (b = 255, r = g = 0 for
//     Cb and r = 255, g = b = 0 for Cr reach 1<<24, which clamps to 255).
//   - color.NRGBA.RGBA() then >>8 is ((c*0x101*a)/0xff)>>8, which equals
//     (c*a*33025)>>23 for every channel c and alpha a (33025 = 0x8101; the
//     product is at most 255*255*33025 < 1<<31, so it fits a uint32).
//
// Pixels are converted two at a time in 64-bit words holding one channel
// of both pixels (pixel 0 in bits 0-31, pixel 1 in bits 32-63): the Y, Cb
// and Cr sums of each pixel, biases included, lie in [0, 1<<25), so the
// 64-bit sums computed modulo 1<<64 (negative coefficients included) are
// exactly sum0 + sum1<<32 and each pixel's sum is read from its own lane.

// lanes selects the low byte of both 32-bit lanes.
const lanes = 0x000000ff_000000ff

// pixel is color.RGBToYCbCr with the chroma clamp reduced to the upper
// bound (see above).
func pixel(r, g, b int32) (uint8, uint8, uint8) {
	yy := (19595*r + 38470*g + 7471*b + 1<<15) >> 16
	cb := (-11056*r - 21712*g + 32768*b + 257<<15) >> 16
	cr := (32768*r - 27440*g - 5328*b + 257<<15) >> 16
	return uint8(yy), uint8(min(cb, 0xff)), uint8(min(cr, 0xff))
}

// pixels2 is pixel for the two pixels whose channels are packed in r, g
// and b (see above). It returns the pre-shift sums, which yOf and cOf
// turn into the samples of each pixel.
func pixels2(r, g, b uint64) (y, cb, cr uint64) {
	const (
		biasY = 1<<15<<32 | 1<<15
		biasC = 257<<15<<32 | 257<<15
	)
	y = 19595*r + 38470*g + 7471*b + biasY
	cb = 32768*b - 11056*r - 21712*g + biasC
	cr = 32768*r - 27440*g - 5328*b + biasC
	return y, cb, cr
}

// yOf returns the luma samples of both pixels of a pixels2 sum.
func yOf(s uint64) (uint8, uint8) { return uint8(s >> 16), uint8(s >> 48) }

// cOf returns the clamped chroma samples of both pixels of a pixels2 sum.
func cOf(s uint64) (uint8, uint8) {
	return uint8(min(uint32(s)>>16, 0xff)), uint8(min(s>>48, 0xff))
}

// unpack2 splits two RGBA pixels (8 bytes, little endian) into packed
// channels.
func unpack2(v uint64) (r, g, b uint64) { return v & lanes, v >> 8 & lanes, v >> 16 & lanes }

// premulMul is the per-pixel factor of the premultiplication: the 8-bit
// premultiplied value of channel c is (c*premulMul(a))>>23.
func premulMul(a uint8) uint32 { return uint32(a) * 33025 }

// premul8 is uint8(color.NRGBA{c, c, c, a}.RGBA() >> 8) for one channel.
func premul8(c uint8, m uint32) int32 { return int32((uint32(c) * m) >> 23) }

// unpack2N is unpack2 for two NRGBA pixels: it premultiplies the channels.
func unpack2N(v uint64) (r, g, b uint64) {
	m0 := uint64(v>>24&0xff) * 33025
	m1 := uint64(v>>56) * 33025
	r = (v&0xff*m0)>>23 | (v>>32&0xff*m1)>>23<<32
	g = (v>>8&0xff*m0)>>23 | (v>>40&0xff*m1)>>23<<32
	b = (v>>16&0xff*m0)>>23 | (v>>48&0xff*m1)>>23<<32
	return r, g, b
}

// pixelN converts the NRGBA pixel p[:4].
func pixelN(p []byte) (uint8, uint8, uint8) {
	m := premulMul(p[3])
	return pixel(premul8(p[0], m), premul8(p[1], m), premul8(p[2], m))
}

func rgbaToYCbCrGo(rgba, y, cb, cr []byte, n int) {
	if n <= 0 {
		return
	}
	y = y[:n:len(y)]
	rgba, cb, cr = rgba[:4*len(y):len(rgba)], cb[:len(y):len(cb)], cr[:len(y):len(cr)]
	for i := 1; i < len(y); i += 2 {
		sy, scb, scr := pixels2(unpack2(binary.LittleEndian.Uint64(rgba[4*i-4 : 4*i+4 : 4*i+4])))
		y[i-1], y[i] = yOf(sy)
		cb[i-1], cb[i] = cOf(scb)
		cr[i-1], cr[i] = cOf(scr)
	}
	if n&1 != 0 {
		p := rgba[4*n-4:]
		y[n-1], cb[n-1], cr[n-1] = pixel(int32(p[0]), int32(p[1]), int32(p[2]))
	}
}

func nrgbaToYCbCrGo(nrgba, y, cb, cr []byte, n int) {
	if n <= 0 {
		return
	}
	y = y[:n:len(y)]
	nrgba, cb, cr = nrgba[:4*len(y):len(nrgba)], cb[:len(y):len(cb)], cr[:len(y):len(cr)]
	for i := 1; i < len(y); i += 2 {
		sy, scb, scr := pixels2(unpack2N(binary.LittleEndian.Uint64(nrgba[4*i-4 : 4*i+4 : 4*i+4])))
		y[i-1], y[i] = yOf(sy)
		cb[i-1], cb[i] = cOf(scb)
		cr[i-1], cr[i] = cOf(scr)
	}
	if n&1 != 0 {
		y[n-1], cb[n-1], cr[n-1] = pixelN(nrgba[4*n-4:])
	}
}

// Downsample2x2 in 16-bit lanes: the four pair sums of 8 bytes of a row
// are (v & m) + (v>>8 & m); with the other row's and the rounding 2 added
// each lane holds at most 1022, and after >>2 only bits 14-15 of a lane
// receive bits of the next one, which the mask m clears.
const (
	evenBytes = 0x00ff00ff_00ff00ff
	twos      = 0x0002000200020002
)

func downsample2x2Go(r0, r1, out []byte, n int) {
	if n <= 0 {
		return
	}
	out = out[:n:len(out)]
	r0, r1 = r0[:2*len(out):len(r0)], r1[:2*len(out):len(r1)]
	i := 0
	for ; i+4 <= len(out); i += 4 {
		a := binary.LittleEndian.Uint64(r0[2*i : 2*i+8 : 2*i+8])
		b := binary.LittleEndian.Uint64(r1[2*i : 2*i+8 : 2*i+8])
		s := a&evenBytes + a>>8&evenBytes + b&evenBytes + b>>8&evenBytes + twos
		s = s >> 2 & evenBytes
		s = (s | s>>8) & 0x0000ffff_0000ffff
		binary.LittleEndian.PutUint32(out[i:i+4:i+4], uint32(s|s>>16))
	}
	for ; i < len(out); i++ {
		a := r0[2*i : 2*i+2 : 2*i+2]
		b := r1[2*i : 2*i+2 : 2*i+2]
		out[i] = avg4(a[0], a[1], b[0], b[1])
	}
}

// avg4 is the rounded average of four chroma samples, as image/jpeg's
// scale computes it.
func avg4(a, b, c, d uint8) uint8 {
	return uint8((uint32(a) + uint32(b) + uint32(c) + uint32(d) + 2) >> 2)
}

// avgC averages the clamped chroma samples of the pixels2 sums s and t
// (the two pixels of a square in each row).
func avgC(s, t uint64) uint8 {
	a, b := cOf(s)
	c, d := cOf(t)
	return avg4(a, b, c, d)
}

func rgbaToYCbCr420Go(rgba0, rgba1, y0, y1, cb, cr []byte, n int) {
	if n <= 0 {
		return
	}
	y0 = y0[:n:len(y0)]
	y1 = y1[:len(y0):len(y1)]
	rgba0, rgba1 = rgba0[:4*len(y0):len(rgba0)], rgba1[:4*len(y0):len(rgba1)]
	cb = cb[: len(y0)/2 : len(cb)]
	cr = cr[:len(cb):len(cr)]
	for i := 1; i < len(y0); i += 2 {
		sy0, scb0, scr0 := pixels2(unpack2(binary.LittleEndian.Uint64(rgba0[4*i-4 : 4*i+4 : 4*i+4])))
		sy1, scb1, scr1 := pixels2(unpack2(binary.LittleEndian.Uint64(rgba1[4*i-4 : 4*i+4 : 4*i+4])))
		y0[i-1], y0[i] = yOf(sy0)
		y1[i-1], y1[i] = yOf(sy1)
		cb[i>>1] = avgC(scb0, scb1)
		cr[i>>1] = avgC(scr0, scr1)
	}
	if n&1 != 0 {
		// The last pixel of an odd row has luma only.
		p, q := rgba0[4*n-4:], rgba1[4*n-4:]
		y0[n-1], _, _ = pixel(int32(p[0]), int32(p[1]), int32(p[2]))
		y1[n-1], _, _ = pixel(int32(q[0]), int32(q[1]), int32(q[2]))
	}
}

func nrgbaToYCbCr420Go(nrgba0, nrgba1, y0, y1, cb, cr []byte, n int) {
	if n <= 0 {
		return
	}
	y0 = y0[:n:len(y0)]
	y1 = y1[:len(y0):len(y1)]
	nrgba0, nrgba1 = nrgba0[:4*len(y0):len(nrgba0)], nrgba1[:4*len(y0):len(nrgba1)]
	cb = cb[: len(y0)/2 : len(cb)]
	cr = cr[:len(cb):len(cr)]
	for i := 1; i < len(y0); i += 2 {
		sy0, scb0, scr0 := pixels2(unpack2N(binary.LittleEndian.Uint64(nrgba0[4*i-4 : 4*i+4 : 4*i+4])))
		sy1, scb1, scr1 := pixels2(unpack2N(binary.LittleEndian.Uint64(nrgba1[4*i-4 : 4*i+4 : 4*i+4])))
		y0[i-1], y0[i] = yOf(sy0)
		y1[i-1], y1[i] = yOf(sy1)
		cb[i>>1] = avgC(scb0, scb1)
		cr[i>>1] = avgC(scr0, scr1)
	}
	if n&1 != 0 {
		y0[n-1], _, _ = pixelN(nrgba0[4*n-4:])
		y1[n-1], _, _ = pixelN(nrgba1[4*n-4:])
	}
}
