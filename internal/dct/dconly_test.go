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
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
)

// TestDCOnlyExhaustive covers every DC-only block. Its output, dcValue(d)
// with d = b[0]*qt[0] wrapping, depends only on d mod 2^15 (d<<17 keeps
// the low 15 bits), so the test runs all 2^15 low patterns, with random
// high bits, through the kernels' DC path (the arm64 and x86 kernels and
// the portable code each have one) at every level, with the product split
// between b[0] and qt[0] six ways: d*1, 1*d, (-d)*(-1) and three odd
// b[0] times d/b[0] mod 2^32. A DC path that differs from the Ref for any
// d (a rounding term, a wrap of the 32-bit sum, a missing clamp of the
// stored byte, a multiplication done in 16 or 64 bits) fails here, where
// TestDCOnly tries the known wrap points and 20000 random values. From
// the exhaustive DC check of the amd64 kernels' verification.
func TestDCOnlyExhaustive(t *testing.T) { forLevels(t, testDCOnlyExhaustive) }

func testDCOnlyExhaustive(t *testing.T) {
	storeImpls, idctImpls := atLevel(storeImpls), atLevel(idctImpls)
	var fails atomic.Int32
	denseParallel(1<<15, func(w int, lo, hi int) {
		r := rand.New(rand.NewPCG(uint64(w), 103))
		for low := lo; low < hi && fails.Load() < 5; low++ {
			for split := range 6 {
				d := int32(uint32(low) | r.Uint32()<<15)
				var b, qt Block
				for i := range qt {
					qt[i] = int32(r.Uint32()) // ignored: the ACs are zero
				}
				switch split {
				case 0:
					b[0], qt[0] = d, 1
					want := b
					DequantIDCTRef(&want, &qt)
					for i, v := range want {
						if v != dcValue(d) {
							t.Errorf("d %d: Ref output %d = %d, dcValue = %d", d, i, v, dcValue(d))
							fails.Add(1)
							break
						}
					}
				case 1:
					b[0], qt[0] = 1, d
				case 2:
					b[0], qt[0] = -d, -1
				default:
					// b0 odd has an inverse mod 2^32 (Newton's iteration
					// doubles the correct low bits: 3, 6, ..., 48).
					b0 := r.Uint32() | 1
					inv := b0
					for range 4 {
						inv *= 2 - b0*inv
					}
					b[0], qt[0] = int32(b0), int32(uint32(d)*inv)
				}
				if b[0]*qt[0] != d {
					t.Errorf("split %d of %d: %d * %d", split, d, b[0], qt[0])
					fails.Add(1)
				}
				for _, impl := range storeImpls {
					if !checkStore(t, r, impl.name, impl.f, &b, &qt, 8+r.IntN(33), r.IntN(16)) {
						fails.Add(1)
					}
				}
				for _, impl := range idctImpls {
					if !checkIDCT(t, impl.name, impl.f, &b, &qt) {
						fails.Add(1)
					}
				}
			}
		}
	})
}

// denseParallel splits [0, n) into contiguous ranges, one per CPU, and
// runs f(worker, lo, hi) on each.
func denseParallel(n int, f func(w, lo, hi int)) {
	workers := min(runtime.GOMAXPROCS(0), n)
	var wg sync.WaitGroup
	for w := range workers {
		wg.Go(func() { f(w, n*w/workers, n*(w+1)/workers) })
	}
	wg.Wait()
}
