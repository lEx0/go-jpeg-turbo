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

import (
	"bytes"
	"fmt"
	"runtime"
	"slices"
	"testing"
)

// Regression tests for the slice bounds of the exported kernels: a slice is
// long enough when its len, not its cap, covers what n requires.

// regKernel is an exported kernel with its Ref, as functions of the slice
// arguments in order; need gives the bytes each argument must have for n,
// out which arguments are outputs.
type regKernel struct {
	name      string
	need      func(n int) []int
	out       []bool
	fast, ref func(a [][]byte, n int)
}

var regKernels = []regKernel{
	{"RGBAToYCbCr", func(n int) []int { return []int{4 * n, n, n, n} }, []bool{false, true, true, true},
		func(a [][]byte, n int) { RGBAToYCbCr(a[0], a[1], a[2], a[3], n) },
		func(a [][]byte, n int) { RGBAToYCbCrRef(a[0], a[1], a[2], a[3], n) }},
	{"NRGBAToYCbCr", func(n int) []int { return []int{4 * n, n, n, n} }, []bool{false, true, true, true},
		func(a [][]byte, n int) { NRGBAToYCbCr(a[0], a[1], a[2], a[3], n) },
		func(a [][]byte, n int) { NRGBAToYCbCrRef(a[0], a[1], a[2], a[3], n) }},
	{"Downsample2x2", func(n int) []int { return []int{2 * n, 2 * n, n} }, []bool{false, false, true},
		func(a [][]byte, n int) { Downsample2x2(a[0], a[1], a[2], n) },
		func(a [][]byte, n int) { Downsample2x2Ref(a[0], a[1], a[2], n) }},
	{"RGBAToYCbCr420", func(n int) []int { return []int{4 * n, 4 * n, n, n, n / 2, n / 2} }, []bool{false, false, true, true, true, true},
		func(a [][]byte, n int) { RGBAToYCbCr420(a[0], a[1], a[2], a[3], a[4], a[5], n) },
		func(a [][]byte, n int) { RGBAToYCbCr420Ref(a[0], a[1], a[2], a[3], a[4], a[5], n) }},
	{"NRGBAToYCbCr420", func(n int) []int { return []int{4 * n, 4 * n, n, n, n / 2, n / 2} }, []bool{false, false, true, true, true, true},
		func(a [][]byte, n int) { NRGBAToYCbCr420(a[0], a[1], a[2], a[3], a[4], a[5], n) },
		func(a [][]byte, n int) { NRGBAToYCbCr420Ref(a[0], a[1], a[2], a[3], a[4], a[5], n) }},
}

// regLengths covers the portable short-row code and the vector code (n >=
// 16), even and odd.
var regLengths = []int{1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 100}

// regArgs returns the arguments for need: argument i has len[i] bytes and
// cap[i] bytes of backing array, all of it holding a pattern, so that a
// copy of each backing array shows what a call changed.
func regArgs(lens, caps []int) (args, backing [][]byte) {
	args, backing = make([][]byte, len(lens)), make([][]byte, len(lens))
	for i := range lens {
		b := make([]byte, caps[i])
		for j := range b {
			b[j] = byte(j*29 + 7*i + 3)
		}
		args[i], backing[i] = b[:lens[i]:caps[i]], b
	}
	return args, backing
}

func regClone(bufs [][]byte) [][]byte {
	c := make([][]byte, len(bufs))
	for i, b := range bufs {
		c[i] = bytes.Clone(b)
	}
	return c
}

// regPanic calls f and returns what it panicked with (nil if it returned).
func regPanic(f func()) (v any) {
	defer func() { v = recover() }()
	f()
	return nil
}

// TestRegressLenBelowCap: an argument with len(s) < need <= cap(s) must
// make the kernel panic with a bounds error before it writes anything,
// inside or past the len of any output, as dispatch.go promises and as the
// Ref panics. The wrappers bounded their arguments with s[:k], which Go
// checks against cap(s): the kernels then wrote the outputs past len (into
// the next row of a plane the rows are cut from) and Downsample2x2 read its
// inputs past len, instead of panicking.
func TestRegressLenBelowCap(t *testing.T) { forEachLevel(t, testRegressLenBelowCap) }

func testRegressLenBelowCap(t *testing.T) {
	for _, k := range regKernels {
		for _, n := range regLengths {
			need := k.need(n)
			for i := range need {
				if need[i] == 0 {
					continue
				}
				for _, extra := range []int{1, 64} {
					lens, caps := slices.Clone(need), slices.Clone(need)
					lens[i], caps[i] = need[i]-1, need[i]-1+extra
					kind := "input"
					if k.out[i] {
						kind = "output"
					}
					what := fmt.Sprintf("%s/%s n %d: %s %d len %d cap %d", k.name, implName(), n, kind, i, lens[i], caps[i])

					args, backing := regArgs(lens, caps)
					before := regClone(backing)
					v := regPanic(func() { k.fast(args, n) })
					if _, ok := v.(runtime.Error); !ok {
						t.Errorf("%s: got panic %v, want a bounds error", what, v)
					}
					for j := range backing {
						if !bytes.Equal(backing[j], before[j]) {
							t.Errorf("%s: argument %d changed before the panic", what, j)
						}
					}

					// The specification panics too.
					args, _ = regArgs(lens, caps)
					if v := regPanic(func() { k.ref(args, n) }); v == nil {
						t.Errorf("%s: Ref does not panic", what)
					}
				}
			}
		}
	}
}

// TestRegressCapBeyondLen: slices long enough by len and with spare cap
// give the Ref's output and leave every byte past len alone.
func TestRegressCapBeyondLen(t *testing.T) { forEachLevel(t, testRegressCapBeyondLen) }

func testRegressCapBeyondLen(t *testing.T) {
	for _, k := range regKernels {
		for _, n := range regLengths {
			need := k.need(n)
			caps := make([]int, len(need))
			for j, v := range need {
				caps[j] = v + 64
			}
			args, backing := regArgs(need, caps)
			before := regClone(backing)
			want, _ := regArgs(need, caps)
			k.ref(want, n)
			k.fast(args, n)
			for j := range args {
				if !bytes.Equal(args[j], want[j]) {
					t.Errorf("%s/%s n %d: argument %d differs from the Ref", k.name, implName(), n, j)
				}
				if !bytes.Equal(backing[j][need[j]:], before[j][need[j]:]) {
					t.Errorf("%s/%s n %d: argument %d changed past its len", k.name, implName(), n, j)
				}
			}
		}
	}
}

// TestRegressSharedPlane: rows cut from one plane with two-index slices, as
// the encoder cuts them, and an n one row sample too large. The kernel must
// panic, not write the first samples of the next rows.
func TestRegressSharedPlane(t *testing.T) { forEachLevel(t, testRegressSharedPlane) }

func testRegressSharedPlane(t *testing.T) {
	const stride = 64
	for _, w := range []int{8, 9, 48} {
		plane := bytes.Repeat([]byte{0x11}, 4*stride)
		rgba := bytes.Repeat([]byte{0xff}, 4*(w+2))
		row := func(r, n int) []byte { return plane[r*stride : r*stride+n] }
		y0, y1, cb, cr := row(0, w), row(1, w), row(2, w), row(3, w)
		// Luma rows of w+2 samples and chroma rows of w/2: n = w+2 needs
		// one chroma sample more than the rows hold.
		l0, l1, c0, c1 := row(0, w+2), row(1, w+2), row(2, w/2), row(3, w/2)
		calls := []struct {
			name string
			f    func()
		}{
			{"RGBAToYCbCr", func() { RGBAToYCbCr(rgba, y0, cb, cr, w+1) }},
			{"NRGBAToYCbCr", func() { NRGBAToYCbCr(rgba, y0, cb, cr, w+1) }},
			{"Downsample2x2", func() { Downsample2x2(rgba, rgba, y0, w+1) }},
			{"RGBAToYCbCr420", func() { RGBAToYCbCr420(rgba, rgba, y0, y1, cb, cr, w+1) }},
			{"NRGBAToYCbCr420", func() { NRGBAToYCbCr420(rgba, rgba, y0, y1, cb, cr, w+1) }},
			{"RGBAToYCbCr420 chroma", func() { RGBAToYCbCr420(rgba, rgba, l0, l1, c0, c1, w&^1+2) }},
			{"NRGBAToYCbCr420 chroma", func() { NRGBAToYCbCr420(rgba, rgba, l0, l1, c0, c1, w&^1+2) }},
		}
		for _, c := range calls {
			if regPanic(c.f) == nil {
				t.Errorf("%s/%s w %d: no panic for n past the row", c.name, implName(), w)
			}
			for i, v := range plane {
				if v != 0x11 {
					t.Errorf("%s/%s w %d: plane[%d] changed to %#x (the kernel must panic before writing)", c.name, implName(), w, i, v)
					break
				}
			}
			for i := range plane {
				plane[i] = 0x11
			}
		}
	}
}
