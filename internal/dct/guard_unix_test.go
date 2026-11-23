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

//go:build darwin || linux

package dct

import (
	"bytes"
	"math/rand/v2"
	"syscall"
	"testing"
	"unsafe"
)

// pageBytes returns an n-byte slice that ends right before an
// inaccessible page (atEnd) or starts right after one (!atEnd), so that a
// kernel reading or writing past the end (or before the start) of the
// slice faults instead of silently touching neighbouring memory.
func pageBytes(t *testing.T, n int, atEnd bool) []byte {
	t.Helper()
	pg := syscall.Getpagesize()
	data := max((n+pg-1)/pg*pg, pg)
	mem, err := syscall.Mmap(-1, 0, data+2*pg, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatalf("mmap: %v", err)
	}
	t.Cleanup(func() { syscall.Munmap(mem) })
	if err := syscall.Mprotect(mem[:pg], syscall.PROT_NONE); err != nil {
		t.Fatalf("mprotect: %v", err)
	}
	if err := syscall.Mprotect(mem[pg+data:], syscall.PROT_NONE); err != nil {
		t.Fatalf("mprotect: %v", err)
	}
	if atEnd {
		return mem[pg+data-n : pg+data : pg+data]
	}
	return mem[pg : pg+n : pg+n]
}

// pageBlock returns a Block next to an inaccessible page (see pageBytes).
func pageBlock(t *testing.T, atEnd bool) *Block {
	return (*Block)(unsafe.Pointer(&pageBytes(t, int(unsafe.Sizeof(Block{})), atEnd)[0]))
}

// TestGuardPages runs every kernel at every level with each of its inputs
// and outputs next to inaccessible pages, on both sides.
func TestGuardPages(t *testing.T) {
	forLevels(t, testGuardPages)
}

func testGuardPages(t *testing.T) {
	r := rand.New(rand.NewPCG(21, 22))
	for _, atEnd := range []bool{true, false} {
		for i := 0; i < 50; i++ {
			kind := i % numKinds
			var b, qt Block
			if kind == kindWrapZero {
				genWrapZero(r, &b, &qt)
			} else {
				genBlock(r, kind, &b)
				genQT(r, r.IntN(numQT), &qt)
			}
			gb, gq := pageBlock(t, atEnd), pageBlock(t, atEnd)
			for _, stride := range []int{8, 9, 15, 16, 33, 64, 4096} {
				n := 7*stride + 8
				want := make([]byte, n)
				w := b
				DequantIDCTStoreRef(&w, &qt, want, stride)
				dst := pageBytes(t, n, atEnd)
				*gb, *gq = b, qt
				DequantIDCTStore(gb, gq, dst, stride)
				for y := 0; y < 8; y++ {
					if !bytes.Equal(dst[y*stride:y*stride+8], want[y*stride:y*stride+8]) {
						t.Fatalf("DequantIDCTStore stride %d atEnd %v: row %d differs", stride, atEnd, y)
					}
				}
			}
			want := b
			DequantIDCTRef(&want, &qt)
			*gb, *gq = b, qt
			DequantIDCT(gb, gq)
			if *gb != want {
				t.Fatalf("DequantIDCT atEnd %v: output differs", atEnd)
			}
		}

		// The forward DCT with the samples, the table and the output next
		// to the guard pages.
		gt := (*Quant)(unsafe.Pointer(&pageBytes(t, int(unsafe.Sizeof(Quant{})), atEnd)[0]))
		gout := (*[BlockSize]int16)(unsafe.Pointer(&pageBytes(t, 2*BlockSize, atEnd)[0]))
		for i := 0; i < 50; i++ {
			var s [BlockSize]byte
			genSamples(r, i%numSmp, &s)
			q := genTable(r)
			gt.Init(&q)
			for _, stride := range []int{0, 1, 7, 8, 9, 16, 33, 4096} {
				n := 7*stride + 8
				src := pageBytes(t, n, atEnd)
				for y := 0; y < 8; y++ {
					copy(src[y*stride:y*stride+8], s[8*y:8*y+8])
				}
				var want [BlockSize]int16
				wantNZ := FDCTQuantRef(src, stride, gt, &want)
				if nz := FDCTQuant(src, stride, gt, gout); nz != wantNZ || *gout != want {
					t.Fatalf("FDCTQuant stride %d atEnd %v: output differs", stride, atEnd)
				}
			}
		}
	}
}
