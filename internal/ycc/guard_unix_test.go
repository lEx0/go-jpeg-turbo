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

package ycc

import (
	"bytes"
	"math/rand/v2"
	"syscall"
	"testing"
)

// pageBytes returns an n-byte slice that ends right before an
// inaccessible page (atEnd) or starts right after one (!atEnd), so that a
// kernel reading or writing past the end (or before the start) of a slice
// faults instead of silently touching neighbouring memory.
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

// TestGuardPages runs every kernel with all inputs and outputs next to
// inaccessible pages, on both sides.
func TestGuardPages(t *testing.T) { forEachLevel(t, testGuardPages) }

func testGuardPages(t *testing.T) {
	r := rand.New(rand.NewPCG(12, 13))
	var ns []int
	for n := range 40 {
		ns = append(ns, n)
	}
	ns = append(ns, 47, 48, 49, 100, 1000, 1023, 4096)
	for _, atEnd := range []bool{true, false} {
		for _, n := range ns {
			pb := func(size int) []byte { return pageBytes(t, size, atEnd) }
			s0, s1 := pb(4*n), pb(4*n)
			fill(r, s0, 0, true)
			fill(r, s1, 0, true)
			for _, k := range rowKernels {
				wy, wcb, wcr := make([]byte, n), make([]byte, n), make([]byte, n)
				k.ref(s0, wy, wcb, wcr, n)
				y, cb, cr := pb(n), pb(n), pb(n)
				k.fn(s0, y, cb, cr, n)
				if !bytes.Equal(y, wy) || !bytes.Equal(cb, wcb) || !bytes.Equal(cr, wcr) {
					t.Fatalf("%s n=%d atEnd=%v: output differs from the Ref", k.name, n, atEnd)
				}
			}
			for _, k := range rows420Kernels {
				wy0, wy1, wcb, wcr := make([]byte, n), make([]byte, n), make([]byte, n/2), make([]byte, n/2)
				k.ref(s0, s1, wy0, wy1, wcb, wcr, n)
				y0, y1, cb, cr := pb(n), pb(n), pb(n/2), pb(n/2)
				k.fn(s0, s1, y0, y1, cb, cr, n)
				if !bytes.Equal(y0, wy0) || !bytes.Equal(y1, wy1) || !bytes.Equal(cb, wcb) || !bytes.Equal(cr, wcr) {
					t.Fatalf("%s n=%d atEnd=%v: output differs from the Ref", k.name, n, atEnd)
				}
			}
			r0, r1 := pb(2*n), pb(2*n)
			fill(r, r0, 0, false)
			fill(r, r1, 0, false)
			want, out := make([]byte, n), pb(n)
			Downsample2x2Ref(r0, r1, want, n)
			Downsample2x2(r0, r1, out, n)
			if !bytes.Equal(out, want) {
				t.Fatalf("Downsample2x2 n=%d atEnd=%v: output differs from the Ref", n, atEnd)
			}
		}
	}
}
