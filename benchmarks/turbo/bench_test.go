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

package turbo

import (
	"testing"

	"github.com/lEx0/go-jpeg-turbo/benchmarks/suite"
)

func BenchmarkDecode(b *testing.B)       { suite.Decode(b, Codec{}) }
func BenchmarkDecodeConfig(b *testing.B) { suite.DecodeConfig(b, Codec{}) }
func BenchmarkEncode(b *testing.B)       { suite.Encode(b, Codec{}) }
func BenchmarkSatDecode(b *testing.B)    { suite.SatDecode(b, Codec{}) }
func BenchmarkSatEncode(b *testing.B)    { suite.SatEncode(b, Codec{}) }

// TestSameAsStd checks the build on the suite's workloads before run.sh
// measures it; see suite.Check.
func TestSameAsStd(t *testing.T) {
	t.Log(Codec{}.Describe())
	suite.Check(t, Codec{})
}
