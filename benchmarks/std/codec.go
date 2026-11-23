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

// Package std adapts the standard library's image/jpeg to the benchmark
// suite: the reference the other builds are compared with.
package std

import (
	"image"
	"image/jpeg"
	"io"
	"runtime"

	"github.com/lEx0/go-jpeg-turbo/benchmarks/suite"
)

// Codec is the image/jpeg adapter.
type Codec struct{}

var _ suite.Codec = Codec{}

func (Codec) Decode(r io.Reader) (image.Image, error) { return jpeg.Decode(r) }

func (Codec) DecodeConfig(r io.Reader) (image.Config, error) { return jpeg.DecodeConfig(r) }

func (Codec) Encode(w io.Writer, m image.Image, quality int) error {
	return jpeg.Encode(w, m, &jpeg.Options{Quality: quality})
}

func (Codec) Describe() string { return "image/jpeg of " + runtime.Version() }
