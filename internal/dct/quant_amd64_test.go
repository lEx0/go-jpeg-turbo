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

//go:build amd64 && !purego

package dct

import "github.com/lEx0/go-jpeg-turbo/internal/cpuinfo"

// quantVec runs the vector quantization of the selected level alone
// (quantSSE41, quantAVX2, quantAVX512) on the column-major c.
func quantVec(c *[BlockSize]int16, t *Quant, out *[BlockSize]int16) (uint64, bool) {
	var b Block
	for i, v := range c {
		b[i] = int32(v)
	}
	switch level {
	case cpuinfo.SSE41:
		return quantSSE41(&b, t, out), true
	case cpuinfo.AVX2:
		return quantAVX2(&b, t, out), true
	case cpuinfo.AVX512:
		return quantAVX512(&b, t, out), true
	}
	return 0, false
}
