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

// Benchmarks of this package ("turbo") against image/jpeg ("std") on the
// same inputs. The MB/s column is megapixels per second (SetBytes is the
// pixel count). Compare with benchstat:
//
//	go test -run '^$' -bench 'Compat' -count 10 ./jpeg > new.txt
//	benchstat -col /impl new.txt

import (
	"bytes"
	"image"
	"image/draw"
	std "image/jpeg"
	"io"
	"testing"
)

var benchDecodeFiles = []struct{ name, path string }{
	{"420q75_960x1280", "../testdata/bench/photo-960x1280-q75-420.jpg"},
	{"444q92_960x1280", "../testdata/photo-960x1280.jpg"},
	{"422q92rst_960x1280", "../testdata/bench/photo-960x1280-q92-422-rst.jpg"},
	{"420q85prog_960x1280", "../testdata/bench/photo-960x1280-q85-420-prog.jpg"},
	{"420q85_1920x2560", "../testdata/bench/photo-1920x2560-q85-420.jpg"},
	{"grayq85_1920x2560", "../testdata/bench/photo-1920x2560-q85-gray.jpg"},
	{"444_1750x697", "../testdata/logo-1750x697.jpg"},
}

var benchImpls = []struct {
	name   string
	decode func(io.Reader) (image.Image, error)
	encode func(io.Writer, image.Image, int) error
}{
	{"std", std.Decode, func(w io.Writer, m image.Image, q int) error { return std.Encode(w, m, &std.Options{Quality: q}) }},
	{"turbo", Decode, func(w io.Writer, m image.Image, q int) error { return Encode(w, m, &Options{Quality: q}) }},
}

func BenchmarkCompatDecode(b *testing.B) {
	for _, f := range benchDecodeFiles {
		data := readFile(b, f.path)
		cfg, err := std.DecodeConfig(bytes.NewReader(data))
		if err != nil {
			b.Fatal(err)
		}
		for _, impl := range benchImpls {
			b.Run("img="+f.name+"/impl="+impl.name, func(b *testing.B) {
				b.SetBytes(int64(cfg.Width * cfg.Height))
				b.ReportAllocs()
				for b.Loop() {
					if _, err := impl.decode(bytes.NewReader(data)); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

func BenchmarkCompatEncode(b *testing.B) {
	src, err := std.Decode(bytes.NewReader(readFile(b, "../testdata/photo-960x1280.jpg")))
	if err != nil {
		b.Fatal(err)
	}
	bounds := src.Bounds()
	rgba := image.NewRGBA(bounds)
	draw.Draw(rgba, bounds, src, bounds.Min, draw.Src)
	nrgba := image.NewNRGBA(bounds)
	draw.Draw(nrgba, bounds, src, bounds.Min, draw.Src)
	gray := image.NewGray(bounds)
	draw.Draw(gray, bounds, src, bounds.Min, draw.Src)
	ycc420, err := std.Decode(bytes.NewReader(readFile(b, "../testdata/bench/photo-960x1280-q75-420.jpg")))
	if err != nil {
		b.Fatal(err)
	}
	inputs := []struct {
		name string
		m    image.Image
	}{
		{"RGBA", rgba},
		{"NRGBA", nrgba},
		{"YCbCr444", src},
		{"YCbCr420", ycc420},
		{"Gray", gray},
	}
	for _, in := range inputs {
		for _, q := range []int{75, 90} {
			for _, impl := range benchImpls {
				b.Run("img="+in.name+"/q="+itoa(q)+"/impl="+impl.name, func(b *testing.B) {
					b.SetBytes(int64(bounds.Dx() * bounds.Dy()))
					b.ReportAllocs()
					var buf bytes.Buffer
					for b.Loop() {
						buf.Reset()
						if err := impl.encode(&buf, in.m, q); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b []byte
	for ; i > 0; i /= 10 {
		b = append([]byte{byte('0' + i%10)}, b...)
	}
	return string(b)
}
