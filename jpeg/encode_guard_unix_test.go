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

package jpeg

// The encoder's inputs next to inaccessible pages, at the level under
// test. The color conversion and downsampling kernels read the rows of
// RGBA and NRGBA images and the planes of YCbCr images straight from the
// caller's slices, and an over-read past the end of a row or a plane is
// invisible to the byte comparison with image/jpeg unless the extra bytes
// change the output. Here such a read faults.
//
// This is defense in depth rather than coverage of its own. The kernel
// packages run the same check on the kernels themselves (TestGuardPages),
// and the kernels' Go wrappers reslice the arguments to the length the
// assembly reads, so a call site in this package that passes a slice too
// short for its row panics before any kernel runs and fails every
// comparison test. A defect that only this test would catch takes two: a
// call site here that passes a slice shorter than the kernel reads, and
// a wrapper that no longer checks the length. From the verification of
// the amd64 levels.

import (
	"fmt"
	"image"
	"math/rand"
	"syscall"
	"testing"
)

// guardBytes returns an n-byte slice that ends right before an
// inaccessible page (atEnd) or starts right after one, and a function
// that unmaps it.
func guardBytes(t *testing.T, n int, atEnd bool) ([]byte, func()) {
	t.Helper()
	pg := syscall.Getpagesize()
	data := max((n+pg-1)/pg*pg, pg)
	mem, err := syscall.Mmap(-1, 0, data+2*pg, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatalf("mmap: %v", err)
	}
	free := func() { syscall.Munmap(mem) }
	if err := syscall.Mprotect(mem[:pg], syscall.PROT_NONE); err != nil {
		free()
		t.Fatalf("mprotect: %v", err)
	}
	if err := syscall.Mprotect(mem[pg+data:], syscall.PROT_NONE); err != nil {
		free()
		t.Fatalf("mprotect: %v", err)
	}
	if atEnd {
		return mem[pg+data-n : pg+data : pg+data], free
	}
	return mem[pg : pg+n : pg+n], free
}

// TestEncGuardPages encodes RGBA, NRGBA and Gray images and YCbCr 4:4:4
// and 4:2:0 images whose pixel slices start or end at an inaccessible
// page, of widths 1 to 300 (1 to 80 and some larger ones with -short),
// 511 to 513 and 1023 to 1025, and RGBA images whose rows all are the
// last bytes of such a slice (stride 0), and compares with image/jpeg.
func TestEncGuardPages(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	var widths []int
	for w := 1; w <= 300; w++ {
		if !testing.Short() || w <= 80 || w%64 <= 1 || w%64 == 63 {
			widths = append(widths, w)
		}
	}
	widths = append(widths, 511, 512, 513, 1023, 1024, 1025)
	opts := []*Options{nil, {Quality: 1}, {Quality: 100}, {Quality: 50}, {Quality: 90}, {Quality: 23}}
	for _, atEnd := range []bool{true, false} {
		for _, w := range widths {
			for _, h := range []int{1, 2, 3, 17} {
				if w > 300 && h > 2 {
					continue
				}
				var frees []func()
				buf := func(n int) []byte {
					b, free := guardBytes(t, n, atEnd)
					frees = append(frees, free)
					rng.Read(b)
					return b
				}
				r := image.Rect(0, 0, w, h)
				pix := buf(4 * w * h)
				cw, ch := (w+1)/2, (h+1)/2
				ims := []image.Image{
					&image.RGBA{Pix: pix, Stride: 4 * w, Rect: r},
					&image.NRGBA{Pix: pix, Stride: 4 * w, Rect: r},
					&image.Gray{Pix: buf(w * h), Stride: w, Rect: r},
					&image.YCbCr{
						Y: buf(w * h), Cb: buf(w * h), Cr: buf(w * h),
						YStride: w, CStride: w, SubsampleRatio: image.YCbCrSubsampleRatio444, Rect: r,
					},
					&image.YCbCr{
						Y: buf(w * h), Cb: buf(cw * ch), Cr: buf(cw * ch),
						YStride: w, CStride: cw, SubsampleRatio: image.YCbCrSubsampleRatio420, Rect: r,
					},
					// Rows that overlap: every row is the last 4*w bytes.
					&image.RGBA{Pix: pix[len(pix)-4*w:], Stride: 0, Rect: r},
				}
				o := opts[(w+h)%len(opts)]
				for _, m := range ims {
					compareEncode(t, fmt.Sprintf("%T/%dx%d/atEnd=%v", m, w, h, atEnd), m, o, false)
				}
				for _, free := range frees {
					free()
				}
				if t.Failed() {
					return
				}
			}
		}
	}
}
