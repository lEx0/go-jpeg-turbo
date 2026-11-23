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
	"math/rand/v2"
	"testing"
	"unsafe"
)

// TestUnaligned runs the kernels at every level with blocks, tables and
// outputs at every offset their types allow (and every offset of the byte
// planes), so that no kernel relies on more alignment than Go guarantees.
func TestUnaligned(t *testing.T) {
	forLevels(t, testUnaligned)
}

func testUnaligned(t *testing.T) {
	r := rand.New(rand.NewPCG(23, 24))
	words := make([]int32, 3*BlockSize)
	for off := 0; off < 16; off++ {
		var b, qt Block
		genBlock(r, off%numKinds, &b)
		genQT(r, r.IntN(numQT), &qt)
		gb := (*Block)(words[off : off+BlockSize])
		gq := (*Block)(words[off+BlockSize+off%3 : off+2*BlockSize+off%3])
		for _, stride := range []int{8, 13, 24} {
			*gb, *gq = b, qt
			checkStore(t, r, "unaligned", DequantIDCTStore, gb, gq, stride, off)
		}
		*gb, *gq = b, qt
		want := b
		DequantIDCTRef(&want, &qt)
		DequantIDCT(gb, gq)
		if *gb != want {
			t.Fatalf("DequantIDCT at word offset %d: output differs", off)
		}
	}
	// Quant holds int64 fields: 8-byte offsets in a []uint64.
	qwords := make([]uint64, 3*int(unsafe.Sizeof(Quant{}))/8)
	halves := make([]int16, 2*BlockSize+16)
	for off := 0; off < 8; off++ {
		gt := (*Quant)(unsafe.Pointer(&qwords[off]))
		q := genTable(r)
		gt.Init(&q)
		gout := (*[BlockSize]int16)(halves[off : off+BlockSize])
		var s [BlockSize]byte
		genSamples(r, off%numSmp, &s)
		for _, stride := range []int{8, 11, 32} {
			for soff := 0; soff < 16; soff += 5 {
				checkFDCT(t, "unaligned", func(src []byte, stride int, _ *Quant, out *[BlockSize]int16) uint64 {
					nz := FDCTQuant(src, stride, gt, gout)
					*out = *gout
					return nz
				}, &s, gt, stride, soff)
			}
		}
	}
}
