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

// Whole decodes with coefficients and quantization tables at the limits
// of what image/jpeg accepts, at the level under test. Extended sequential
// (SOF1) and progressive files may carry 16-bit quantization tables, and
// image/jpeg multiplies them with coefficients of up to 16 bits and lets
// the DC prediction grow without bound, so that the dequantized values
// and the IDCT's intermediates wrap around int32. The kernels take those
// values as int32 and must wrap the same way; their own tests feed them
// such blocks directly, while these tests check that the decoder hands
// the values to them intact (a table narrowed to 16 bits, a clamped DC
// prediction or a saturating multiply would show only here: the random
// files of TestDecCompatHandBuilt keep 16-bit tables at or below 0x2ff).

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"math/rand"
	"path/filepath"
	"testing"
)

// dxRewriteDQT returns data with every DQT segment before the first SOS
// rebuilt from f(zig, orig): 16-bit entries if wide, else 8-bit ones (f's
// value truncated). It returns nil if data has no such segment structure.
func dxRewriteDQT(data []byte, wide bool, f func(zig int, orig uint16) uint16) []byte {
	if len(data) < 4 || data[0] != 0xff || data[1] != soiMarker {
		return nil
	}
	out := append([]byte(nil), data[:2]...)
	for i := 2; i+4 <= len(data); {
		if data[i] != 0xff {
			return nil
		}
		m := data[i+1]
		if m == 0xff {
			out = append(out, 0xff)
			i++
			continue
		}
		n := int(binary.BigEndian.Uint16(data[i+2:]))
		if i+2+n > len(data) {
			return nil
		}
		if m != dqtMarker {
			out = append(out, data[i:i+2+n]...)
		} else {
			var body []byte
			for seg := data[i+4 : i+2+n]; len(seg) > 0; {
				pq, tq := seg[0]>>4, seg[0]&15
				size := 64 * int(pq+1)
				if pq > 1 || len(seg) < 1+size {
					return nil
				}
				var orig [64]uint16
				for z := range orig {
					if pq == 1 {
						orig[z] = binary.BigEndian.Uint16(seg[1+2*z:])
					} else {
						orig[z] = uint16(seg[1+z])
					}
				}
				if wide {
					body = append(body, 0x10|tq)
				} else {
					body = append(body, tq)
				}
				for z := range orig {
					if v := f(z, orig[z]); wide {
						body = binary.BigEndian.AppendUint16(body, v)
					} else {
						body = append(body, byte(v))
					}
				}
				seg = seg[1+size:]
			}
			out = append(out, 0xff, dqtMarker)
			out = binary.BigEndian.AppendUint16(out, uint16(len(body)+2))
			out = append(out, body...)
		}
		i += 2 + n
		if m == sosMarker {
			return append(out, data[i:]...)
		}
	}
	return nil
}

// TestDecExtremeDQT decodes the corpus (every eighth file with -short)
// with its quantization tables replaced by extreme ones: every entry
// 65535, the entries times 257, random 16-bit values (with and without
// zeros), powers of two, a huge DC or huge AC entries, and the 8-bit
// tables 0, 1 and 255. The fast entropy decoder is forced on, so that
// every scan goes through the fused dequantize+IDCT kernels.
func TestDecExtremeDQT(t *testing.T) {
	decForceFast(t)
	rng := rand.New(rand.NewSource(31))
	strategies := []struct {
		name string
		wide bool
		f    func(zig int, o uint16) uint16
	}{
		{"all65535", true, func(zig int, o uint16) uint16 { return 65535 }},
		{"x257", true, func(zig int, o uint16) uint16 { return o * 257 }},
		{"rand16", true, func(zig int, o uint16) uint16 { return uint16(rng.Uint32()) }},
		{"rand16zero", true, func(zig int, o uint16) uint16 {
			if rng.Intn(3) == 0 {
				return 0
			}
			return uint16(rng.Uint32())
		}},
		{"pow2", true, func(zig int, o uint16) uint16 { return 1 << rng.Intn(16) }},
		{"dc65535", true, func(zig int, o uint16) uint16 {
			if zig == 0 {
				return 65535
			}
			return o
		}},
		{"acHuge", true, func(zig int, o uint16) uint16 {
			if zig == 0 {
				return o
			}
			return 32768 + uint16(rng.Intn(32768))
		}},
		{"zeroAC", false, func(zig int, o uint16) uint16 {
			if zig == 0 {
				return o
			}
			return 0
		}},
		{"zero", false, func(zig int, o uint16) uint16 { return 0 }},
		{"one", false, func(zig int, o uint16) uint16 { return 1 }},
		{"255", false, func(zig int, o uint16) uint16 { return 255 }},
	}
	n := 0
	for i, f := range corpusFiles(t) {
		if testing.Short() && i%8 != 0 {
			continue
		}
		data := readFile(t, f)
		for _, s := range strategies {
			if d := dxRewriteDQT(data, s.wide, s.f); d != nil {
				compareDecode(t, filepath.Base(f)+"/"+s.name, d, readers[0].mk)
				n++
			}
		}
		if t.Failed() {
			return
		}
	}
	if n < 100 {
		t.Errorf("only %d rewritten files", n)
	}
}

// dxGrayDC builds a w x h gray JPEG, baseline or progressive (a DC first
// scan with point transform al), with a 16-bit quantization table (q0 for
// DC, qac for AC) and Huffman tables of one DC size (11) and one AC
// symbol (0x0B): every block's DC difference is +2047 (or -2047 if neg),
// so that the DC prediction grows by 2047 per block and, with al and the
// table, its dequantized value wraps around int32; with ac, every
// baseline block also gets one 11-bit AC coefficient.
func dxGrayDC(w, h int, progressive bool, al int, ac bool, q0, qac uint16, neg bool) []byte {
	b := &decJPEGBuilder{}
	b.b = append(b.b, 0xff, soiMarker)
	dqt := []byte{0x10}
	for z := 0; z < 64; z++ {
		q := qac
		if z == 0 {
			q = q0
		}
		dqt = binary.BigEndian.AppendUint16(dqt, q)
	}
	b.segment(dqtMarker, dqt)
	sof := byte(sof0Marker)
	if progressive {
		sof = sof2Marker
	}
	b.segment(sof, []byte{8, byte(h >> 8), byte(h), byte(w >> 8), byte(w), 1, 1, 0x11, 0})
	// DC: the single symbol 11, code "0". AC: EOB "00" and 0x0B "01".
	b.segment(dhtMarker, []byte{
		0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11,
		0x10, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00, 0x0b,
	})
	se, ahal := byte(63), byte(0)
	if progressive {
		se, ahal = 0, byte(al)
	}
	b.segment(sosMarker, []byte{1, 1, 0x00, 0, se, ahal})
	v := uint32(2047) // 11 bits; all zeros is -2047.
	if neg {
		v = 0
	}
	for i := 0; i < ((w+7)/8)*((h+7)/8); i++ {
		b.bits(0, 1)
		b.bits(v, 11)
		if !progressive {
			if ac {
				b.bits(1, 2)
				b.bits(uint32(i*977)&2047|1024, 11)
			}
			b.bits(0, 2)
		}
	}
	b.flush()
	return append(b.b, 0xff, eoiMarker)
}

// TestDecExtremeDCWrap decodes dxGrayDC images, baseline and progressive,
// with the fast entropy decoder forced on and not.
func TestDecExtremeDCWrap(t *testing.T) {
	dims := [][2]int{{8, 8}, {64, 64}, {1023, 517}, {4096, 520}}
	if testing.Short() {
		dims = dims[:3]
	}
	for _, force := range []bool{false, true} {
		t.Run(fmt.Sprintf("force=%v", force), func(t *testing.T) {
			if force {
				decForceFast(t)
			}
			for _, d := range dims {
				for _, prog := range []bool{false, true} {
					for _, al := range []int{0, 5, 13} {
						for _, ac := range []bool{false, true} {
							if (!prog && al != 0) || (prog && ac) {
								continue
							}
							for _, q := range [][2]uint16{{1, 1}, {65535, 65535}, {32768, 3}, {257, 65535}, {0, 0}} {
								for _, neg := range []bool{false, true} {
									name := fmt.Sprintf("%dx%d/prog=%v/al=%d/ac=%v/q=%v/neg=%v", d[0], d[1], prog, al, ac, q, neg)
									compareDecode(t, name, dxGrayDC(d[0], d[1], prog, al, ac, q[0], q[1], neg), readers[0].mk)
								}
							}
						}
					}
				}
			}
		})
	}
	// The streams are valid: image/jpeg decodes them.
	if r := decodeStd(bytes.NewReader(dxGrayDC(64, 64, false, 0, true, 65535, 65535, false))); r.err != nil || r.panic != nil {
		t.Fatalf("image/jpeg rejects the stream: %v", r)
	}
}

// dxExtremeJPEG builds an extended sequential (SOF1) JPEG with 16-bit
// quantization tables (qmode 0: all 65535, 1: random, 2: boundary values,
// 3: all 1) and coefficients of the largest magnitudes the decoder accepts:
// DC differences of 16 bits that accumulate, AC values of 15 bits.
func dxExtremeJPEG(rng *rand.Rand, nComp, hv, width, height, qmode int) []byte {
	w := &decJPEGBuilder{}
	w.b = append(w.b, 0xff, soiMarker)
	var p []byte
	for tq := 0; tq < 2; tq++ {
		p = append(p, byte(0x10|tq))
		for i := 0; i < 64; i++ {
			q := 1
			switch qmode {
			case 0:
				q = 65535
			case 1:
				q = 1 + rng.Intn(65535)
			case 2:
				q = []int{1, 255, 256, 32767, 32768, 65535}[rng.Intn(6)]
			}
			p = append(p, byte(q>>8), byte(q))
		}
	}
	w.segment(dqtMarker, p)
	sof := []byte{8, byte(height >> 8), byte(height), byte(width >> 8), byte(width), byte(nComp)}
	for i := 0; i < nComp; i++ {
		f := 0x11
		if i == 0 {
			f = hv
		}
		sof = append(sof, byte(i+1), byte(f), byte(min(i, 1)))
	}
	w.segment(sof1Marker, sof)
	// DC: sizes 0 to 16, all with 5-bit codes. AC: EOB, ZRL and every
	// run/size with sizes 1 to 15, all with 8-bit codes.
	dcT := &decGenTable{}
	dcT.counts[4] = 17
	for s := 0; s <= 16; s++ {
		dcT.vals = append(dcT.vals, byte(s))
	}
	acT := &decGenTable{vals: []byte{0x00, 0xf0}}
	for r := 0; r < 16; r++ {
		for s := 1; s <= 15; s++ {
			acT.vals = append(acT.vals, byte(r<<4|s))
		}
	}
	acT.counts[7] = byte(len(acT.vals))
	dcT.codes, dcT.lengths = decCodes(dcT.counts)
	acT.codes, acT.lengths = decCodes(acT.counts)
	w.segment(dhtMarker, append(dcT.payload(0, 0), acT.payload(1, 0)...))
	sos := []byte{byte(nComp)}
	for i := 0; i < nComp; i++ {
		sos = append(sos, byte(i+1), 0x00)
	}
	w.segment(sosMarker, append(sos, 0, 63, 0))
	emit := func(t *decGenTable, v byte) {
		for i, x := range t.vals {
			if x == v {
				w.bits(t.codes[i], t.lengths[i])
				return
			}
		}
		panic("no symbol")
	}
	// value writes v (|v| < 1<<s) in s bits, as receiveExtend reads it.
	value := func(v int, s uint) {
		if v < 0 {
			v += 1<<s - 1
		}
		w.bits(uint32(v), s)
	}
	bigs := []int{32767, -32767, 32766, -32767, 16384, -16384, 1, -1, 2047, -2047}
	maxH, maxV := hv>>4, hv&15
	if nComp == 1 {
		maxH, maxV = 1, 1
	}
	mxx, myy := (width+8*maxH-1)/(8*maxH), (height+8*maxV-1)/(8*maxV)
	mode := rng.Intn(4) // The signs of the DC differences, and AC patterns.
	for m := 0; m < mxx*myy; m++ {
		for ci := 0; ci < nComp; ci++ {
			nb := 1
			if ci == 0 && nComp > 1 {
				nb = maxH * maxV
			}
			for j := 0; j < nb; j++ {
				// DC: mostly the largest differences, of one sign.
				s := uint(16)
				if rng.Intn(4) == 0 {
					s = uint(rng.Intn(17))
				}
				emit(dcT, byte(s))
				switch {
				case s == 16:
					// receiveExtend reads 16 bits as they come.
					w.bits(uint32(rng.Intn(1<<16)), 16)
				case s > 0:
					v := 1<<s - 1
					if mode == 1 || (mode == 2 && rng.Intn(2) == 0) {
						v = -v
					}
					value(v, s)
				}
				for zig := 1; zig < 64; zig++ {
					if rng.Intn(16) == 0 {
						emit(acT, 0x00)
						break
					}
					v := bigs[rng.Intn(len(bigs))]
					if mode == 3 {
						v = 32767
						if (zig+m)%2 == 1 {
							v = -32767
						}
					}
					s := uint(0)
					for 1<<s <= max(v, -v) {
						s++
					}
					emit(acT, byte(s))
					value(v, s)
				}
			}
		}
	}
	w.flush()
	return append(w.b, 0xff, eoiMarker)
}

// TestDecExtremeCoefficients decodes dxExtremeJPEG streams of several
// sizes and samplings, with the fast entropy decoder forced on and not.
// image/jpeg accepts them all, and some samples must come out unclamped:
// otherwise the comparison would not see the IDCT's arithmetic.
func TestDecExtremeCoefficients(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	mid := 0
	for _, nComp := range []int{1, 3} {
		for _, hv := range []int{0x11, 0x21, 0x22, 0x12, 0x41} {
			if nComp == 1 && hv != 0x11 {
				continue
			}
			for _, sz := range []image.Point{{8, 8}, {17, 9}, {64, 48}, {129, 31}} {
				for qmode := 0; qmode < 4; qmode++ {
					for k := 0; k < 3; k++ {
						data := dxExtremeJPEG(rng, nComp, hv, sz.X, sz.Y, qmode)
						want := decodeStd(bytes.NewReader(data))
						if want.err != nil || want.panic != nil {
							t.Fatalf("image/jpeg rejects the stream: %v", want)
						}
						mid += dxUnclamped(want.img)
						name := fmt.Sprintf("c%d/%#x/%v/q%d/%d", nComp, hv, sz, qmode, k)
						compareDecode(t, name, data, readers[0].mk)
						old := decMinFastBlocks
						decMinFastBlocks = 0
						compareDecode(t, name+"/force", data, readers[0].mk)
						decMinFastBlocks = old
					}
				}
			}
		}
	}
	if mid == 0 {
		t.Error("every sample is 0 or 255: the test does not see the IDCT")
	}
}

// dxUnclamped counts the samples of m other than 0 and 255.
func dxUnclamped(m image.Image) int {
	var bufs [][]byte
	switch m := m.(type) {
	case *image.YCbCr:
		bufs = [][]byte{m.Y, m.Cb, m.Cr}
	case *image.Gray:
		bufs = [][]byte{m.Pix}
	}
	n := 0
	for _, b := range bufs {
		for _, v := range b {
			if v != 0 && v != 255 {
				n++
			}
		}
	}
	return n
}
