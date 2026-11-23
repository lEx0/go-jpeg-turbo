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

// Package suite holds the benchmarks and the peak-memory tool shared by the
// std (the standard library's image/jpeg) and turbo (this repository)
// adapters in ../std and ../turbo, so that both run exactly the same
// workloads on the same inputs.
//
// The decode and encode workloads, and their names, are those of
// BenchmarkCompatDecode and BenchmarkCompatEncode in the jpeg package
// (jpeg/bench_compat_test.go), where the README's per-picture tables come
// from, plus a 12 MP photo in each direction (see data.go).
//
// See run.sh in the parent directory and the Benchmarks section of the
// repository README.
package suite

import (
	"bytes"
	"fmt"
	"image"
	"image/draw"
	stdjpeg "image/jpeg"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"testing"
)

// Codec adapts one implementation to the suite.
type Codec interface {
	Decode(r io.Reader) (image.Image, error)
	DecodeConfig(r io.Reader) (image.Config, error)
	// Encode encodes m with Options{Quality: quality}.
	Encode(w io.Writer, m image.Image, quality int) error
	// Describe names the implementation and the code it runs (for this
	// repository the kernels of the build), for run.sh's env.txt.
	Describe() string
}

// DecodeFile is a decode workload.
type DecodeFile struct {
	Name string // sub-benchmark name, as in jpeg/bench_compat_test.go
	File string // path in the repository's testdata, or "" for the generated 12 MP photo
}

// Read returns the contents of the file.
func (f DecodeFile) Read() ([]byte, error) {
	if f.File == "" {
		return Photo12JPEG()
	}
	return os.ReadFile(filepath.Join(TestdataDir(), f.File))
}

// DecodeFiles are the pictures of the decode benchmarks.
var DecodeFiles = []DecodeFile{
	{"420q75_960x1280", "bench/photo-960x1280-q75-420.jpg"},
	// Restart markers and 23 KB of Exif, XMP, ICC and Photoshop segments.
	{"444q92_960x1280", "photo-960x1280.jpg"},
	{"422q92rst_960x1280", "bench/photo-960x1280-q92-422-rst.jpg"},
	{"420q85prog_960x1280", "bench/photo-960x1280-q85-420-prog.jpg"},
	{"420q85_1920x2560", "bench/photo-1920x2560-q85-420.jpg"},
	{"grayq85_1920x2560", "bench/photo-1920x2560-q85-gray.jpg"},
	{"444_1750x697", "logo-1750x697.jpg"},
	{Photo12JPEGName, ""},
}

// ConfigFile is the picture of the DecodeConfig benchmark: the photo with
// metadata, whose segments DecodeConfig has to skip to reach the frame
// header.
var ConfigFile = DecodeFiles[1]

// EncodeInputs are the pictures of the encode benchmarks, as named by
// EncodeSource.
var EncodeInputs = []string{"RGBA", "NRGBA", "YCbCr444", "YCbCr420", "Gray", Photo12RGBAName}

// EncodeQualities are the Options.Quality values every encode input runs
// with.
var EncodeQualities = []int{75, 90}

// The workloads of the saturation benchmarks: the decode and the encode of
// an image server's typical call (a 1.2 MP photo, and an *image.RGBA such
// as a resizer makes at the default quality).
var (
	SatDecodeFile    = DecodeFiles[0]
	SatEncodeInput   = "RGBA"
	SatEncodeQuality = 75
)

var sources struct {
	sync.Mutex
	m map[string]image.Image
}

// EncodeSource returns the encode input of the given name, made with the
// standard library only, so that every build encodes the same pixels:
//   - RGBA, NRGBA, Gray: the 960x1280 4:4:4 photo of
//     testdata/photo-960x1280.jpg converted with image/draw;
//   - YCbCr444: that photo as decoded (*image.YCbCr 4:4:4);
//   - YCbCr420: testdata/bench/photo-960x1280-q75-420.jpg as decoded
//     (*image.YCbCr 4:2:0);
//   - RGBA_4000x3000: the generated 12 MP photo (Photo12RGBA).
func EncodeSource(name string) (image.Image, error) {
	sources.Lock()
	defer sources.Unlock()
	if m, ok := sources.m[name]; ok {
		return m, nil
	}
	m, err := loadSource(name)
	if err != nil {
		return nil, err
	}
	if sources.m == nil {
		sources.m = make(map[string]image.Image)
	}
	sources.m[name] = m
	return m, nil
}

func loadSource(name string) (image.Image, error) {
	decodeFile := func(file string) (image.Image, error) {
		data, err := os.ReadFile(filepath.Join(TestdataDir(), file))
		if err != nil {
			return nil, err
		}
		return stdjpeg.Decode(bytes.NewReader(data))
	}
	convert := func(dst draw.Image) (image.Image, error) {
		src, err := decodeFile("photo-960x1280.jpg")
		if err != nil {
			return nil, err
		}
		draw.Draw(dst, dst.Bounds(), src, src.Bounds().Min, draw.Src)
		return dst, nil
	}
	r := image.Rect(0, 0, 960, 1280)
	switch name {
	case "RGBA":
		return convert(image.NewRGBA(r))
	case "NRGBA":
		return convert(image.NewNRGBA(r))
	case "Gray":
		return convert(image.NewGray(r))
	case "YCbCr444":
		return decodeFile("photo-960x1280.jpg")
	case "YCbCr420":
		return decodeFile("bench/photo-960x1280-q75-420.jpg")
	case Photo12RGBAName:
		return Photo12RGBA()
	}
	return nil, fmt.Errorf("suite: unknown encode input %q", name)
}

func mustRead(b *testing.B, f DecodeFile) (data []byte, pixels int) {
	b.Helper()
	data, err := f.Read()
	if err != nil {
		b.Fatal(err)
	}
	cfg, err := stdjpeg.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		b.Fatalf("%s: %v", f.Name, err)
	}
	return data, cfg.Width * cfg.Height
}

func mustSource(b *testing.B, name string) (m image.Image, pixels int) {
	b.Helper()
	m, err := EncodeSource(name)
	if err != nil {
		b.Fatal(err)
	}
	return m, m.Bounds().Dx() * m.Bounds().Dy()
}

type counter struct{ n int }

func (c *counter) Write(p []byte) (int, error) { c.n += len(p); return len(p), nil }

// encodedSize encodes m into a writer that only counts the bytes and
// returns the count. Like image/jpeg, Encode wraps such a writer in a
// bufio.Writer of its own.
func encodedSize(c Codec, m image.Image, quality int) (int, error) {
	var w counter
	err := c.Encode(&w, m, quality)
	return w.n, err
}

// timed runs loop as the measured part of b and reports, besides the wall
// time and the allocations:
//   - the CPU time per operation as cpu-ns/op (user + system, all threads
//     of the process, so the garbage collector's work on other threads
//     included) where the platform has getrusage;
//   - with pixels > 0, the throughput in megapixels per second as MP/s
//     (of all goroutines together in the saturation benchmarks), the unit
//     of the README tables.
func timed(b *testing.B, pixels int, loop func()) {
	b.ReportAllocs()
	b.ResetTimer()
	c0, ok := cpuTime()
	loop()
	c1, _ := cpuTime()
	b.StopTimer()
	if ok {
		b.ReportMetric(float64((c1-c0).Nanoseconds())/float64(b.N), "cpu-ns/op")
	}
	if s := b.Elapsed().Seconds(); pixels > 0 && s > 0 {
		b.ReportMetric(float64(pixels)*float64(b.N)/s/1e6, "MP/s")
	}
}

// Decode benchmarks one Decode of an in-memory file per operation:
// Decode/img=<name>.
func Decode(b *testing.B, c Codec) {
	for _, f := range DecodeFiles {
		b.Run("img="+f.Name, func(b *testing.B) {
			data, px := mustRead(b, f)
			timed(b, px, func() {
				for i := 0; i < b.N; i++ {
					if _, err := c.Decode(bytes.NewReader(data)); err != nil {
						b.Fatal(err)
					}
				}
			})
		})
	}
}

// DecodeConfig benchmarks one DecodeConfig per operation:
// DecodeConfig/img=<name>.
func DecodeConfig(b *testing.B, c Codec) {
	f := ConfigFile
	b.Run("img="+f.Name, func(b *testing.B) {
		data, _ := mustRead(b, f)
		timed(b, 0, func() {
			for i := 0; i < b.N; i++ {
				if _, err := c.DecodeConfig(bytes.NewReader(data)); err != nil {
					b.Fatal(err)
				}
			}
		})
	})
}

// Encode benchmarks one Encode per operation:
// Encode/img=<input>/q=<quality>. It reports the encoded size as bytes,
// the same for every build, since the encoders write the same bytes.
func Encode(b *testing.B, c Codec) {
	for _, name := range EncodeInputs {
		for _, q := range EncodeQualities {
			b.Run("img="+name+"/q="+strconv.Itoa(q), func(b *testing.B) {
				m, px := mustSource(b, name)
				var n int
				timed(b, px, func() {
					for i := 0; i < b.N; i++ {
						var err error
						if n, err = encodedSize(c, m, q); err != nil {
							b.Fatal(err)
						}
					}
				})
				b.ReportMetric(float64(n), "bytes")
			})
		}
	}
}

// SatDecode runs GOMAXPROCS decodes at a time (saturation, as on a busy
// server): ns/op is the inverse of the throughput, MP/s the throughput of
// all goroutines together.
func SatDecode(b *testing.B, c Codec) {
	f := SatDecodeFile
	b.Run("img="+f.Name, func(b *testing.B) {
		data, px := mustRead(b, f)
		timed(b, px, func() {
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					if _, err := c.Decode(bytes.NewReader(data)); err != nil {
						b.Error(err)
						return
					}
				}
			})
		})
	})
}

// SatEncode runs GOMAXPROCS encodes at a time (saturation).
func SatEncode(b *testing.B, c Codec) {
	name, q := SatEncodeInput, SatEncodeQuality
	b.Run("img="+name+"/q="+strconv.Itoa(q), func(b *testing.B) {
		m, px := mustSource(b, name)
		timed(b, px, func() {
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					if _, err := encodedSize(c, m, q); err != nil {
						b.Error(err)
						return
					}
				}
			})
		})
	})
}
