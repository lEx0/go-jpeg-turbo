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

package suite

// The 12 MP photo of the workloads 420q90_4000x3000 (decode) and
// RGBA_4000x3000 (encode, peak memory). 4000x3000 is what a 12 MP camera
// writes; the repository's testdata has no picture of that size (it would
// add megabytes to every clone), so the suite makes one, deterministically
// and with the standard library only:
//
//  1. decode testdata/bench/photo-1920x2560-q85-420.jpg with image/jpeg
//     and convert it to an *image.RGBA with image/draw;
//  2. tile that 1920x2560 picture into 4000x3000, every other copy
//     mirrored so that the copies join without edges. Upscaling instead
//     would smooth the picture and leave the encoder fewer non-zero
//     coefficients than a photo has; plain tiling would add sharp edges
//     at the joins;
//  3. encode the result with image/jpeg at quality 90 (4:2:0, as
//     image/jpeg writes every color picture).
//
// The RGBA pixels and the JPEG are cached in DataDir (benchmarks/data,
// git-ignored), so that the peak-memory tool reads its input instead of
// making it (which would raise the peak it measures) and later runs skip
// the work. The files are checked against the SHA-256 sums below, which
// also show that every machine measures the same bytes. A cached file that
// differs is made again; a newly made file that differs (another Go
// release's image/jpeg or image/draw, a changed testdata file) is reported
// on stderr, and run.sh records the sums in env.txt.

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"image"
	"image/draw"
	stdjpeg "image/jpeg"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sync"
)

const (
	// Photo12JPEGName is the decode workload of the generated photo.
	Photo12JPEGName = "420q90_4000x3000"
	// Photo12RGBAName is the encode input of the generated photo.
	Photo12RGBAName = "RGBA_4000x3000"

	photo12W, photo12H = 4000, 3000
	photo12Quality     = 90
	photo12Source      = "bench/photo-1920x2560-q85-420.jpg"
	photo12RGBAFile    = "photo-4000x3000.rgba" // the Pix of the *image.RGBA (stride 4*4000)
	photo12JPEGFile    = "photo-4000x3000-q90-420.jpg"
	photo12RGBASum     = "a116eb1e1dc66d437f6100fa018a10095f55fb1d9c5d8e6252a0dc14f7540acb"
	photo12JPEGSum     = "e6ad935feafc718e5373c136f6f94b7f0d60399a182d7252077a8db809a62760"
)

// TestdataDir returns the repository's testdata directory:
// $JPEG_BENCH_TESTDATA, or ../../testdata from this source file.
func TestdataDir() string {
	if d := os.Getenv("JPEG_BENCH_TESTDATA"); d != "" {
		return d
	}
	return filepath.Join(sourceDir(), "..", "..", "testdata")
}

// DataDir returns the directory of the generated inputs: $JPEG_BENCH_DATA,
// or benchmarks/data.
func DataDir() string {
	if d := os.Getenv("JPEG_BENCH_DATA"); d != "" {
		return d
	}
	return filepath.Join(sourceDir(), "..", "data")
}

// sourceDir returns the directory of this file as compiled, so that the
// test binaries and the peak-memory tool find the pictures from any
// working directory.
func sourceDir() string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Dir(file)
}

// DataFile describes a generated input file.
type DataFile struct {
	Path string
	Size int64
	Sum  string // SHA-256 of the contents, hex
	Want string // the reference SHA-256
}

var data struct {
	once  sync.Once
	files []DataFile
	err   error
}

// Generate makes the generated inputs in DataDir unless they are there
// with the reference contents, and describes them: the RGBA pixels first,
// then the JPEG.
func Generate() ([]DataFile, error) {
	data.once.Do(func() { data.files, data.err = makeData() })
	return data.files, data.err
}

// Photo12JPEG returns the generated 12 MP JPEG.
func Photo12JPEG() ([]byte, error) {
	files, err := Generate()
	if err != nil {
		return nil, err
	}
	return os.ReadFile(files[1].Path)
}

// Photo12RGBA returns the generated 12 MP photo as an *image.RGBA. It reads
// the pixels straight into the image, so that the image is the only memory
// the call leaves allocated or touched.
func Photo12RGBA() (*image.RGBA, error) {
	files, err := Generate()
	if err != nil {
		return nil, err
	}
	f, err := os.Open(files[0].Path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	m := image.NewRGBA(image.Rect(0, 0, photo12W, photo12H))
	if _, err := io.ReadFull(f, m.Pix); err != nil {
		return nil, fmt.Errorf("suite: %s: %v", files[0].Path, err)
	}
	return m, nil
}

func makeData() ([]DataFile, error) {
	dir := DataDir()
	files := []DataFile{
		{Path: filepath.Join(dir, photo12RGBAFile), Want: photo12RGBASum},
		{Path: filepath.Join(dir, photo12JPEGFile), Want: photo12JPEGSum},
	}
	cached := true
	for i := range files {
		f := &files[i]
		var err error
		if f.Sum, f.Size, err = fileSum(f.Path); err != nil || f.Sum != f.Want {
			cached = false
			break
		}
	}
	if cached {
		return files, nil
	}
	m, jpg, err := makePhoto12()
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	for i, b := range [][]byte{m.Pix, jpg} {
		f := &files[i]
		if err := writeFile(f.Path, b); err != nil {
			return nil, err
		}
		s := sha256.Sum256(b)
		f.Sum, f.Size = hex.EncodeToString(s[:]), int64(len(b))
		if f.Sum != f.Want {
			fmt.Fprintf(os.Stderr, "suite: made %s with SHA-256 %s, not the reference %s: "+
				"other machines measure other bytes (another Go release's image/jpeg or image/draw, or a changed testdata/%s)\n",
				f.Path, f.Sum, f.Want, photo12Source)
		}
	}
	return files, nil
}

// makePhoto12 makes the 12 MP photo (see the comment at the top of the
// file) and its JPEG.
func makePhoto12() (*image.RGBA, []byte, error) {
	src, err := os.ReadFile(filepath.Join(TestdataDir(), photo12Source))
	if err != nil {
		return nil, nil, err
	}
	img, err := stdjpeg.Decode(bytes.NewReader(src))
	if err != nil {
		return nil, nil, fmt.Errorf("suite: %s: %v", photo12Source, err)
	}
	sb := img.Bounds()
	tile := image.NewRGBA(image.Rect(0, 0, sb.Dx(), sb.Dy()))
	draw.Draw(tile, tile.Rect, img, sb.Min, draw.Src)
	m := image.NewRGBA(image.Rect(0, 0, photo12W, photo12H))
	tw, th := tile.Rect.Dx(), tile.Rect.Dy()
	for y := range photo12H {
		srow := tile.Pix[mirror(y, th)*tile.Stride:]
		drow := m.Pix[y*m.Stride:]
		for x := range photo12W {
			sx := 4 * mirror(x, tw)
			copy(drow[4*x:4*x+4], srow[sx:sx+4])
		}
	}
	var buf bytes.Buffer
	if err := stdjpeg.Encode(&buf, m, &stdjpeg.Options{Quality: photo12Quality}); err != nil {
		return nil, nil, err
	}
	return m, buf.Bytes(), nil
}

// mirror maps i >= 0 into [0, n) through copies of [0, n) of which every
// other one is reversed: 0 ... n-1, n-1 ... 0, 0 ... n-1, and so on.
func mirror(i, n int) int {
	i %= 2 * n
	if i >= n {
		i = 2*n - 1 - i
	}
	return i
}

func fileSum(path string) (sum string, size int64, err error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()
	h := sha256.New()
	if size, err = io.Copy(h, f); err != nil {
		return "", 0, err
	}
	return hex.EncodeToString(h.Sum(nil)), size, nil
}

// writeFile writes the file through a temporary file and a rename, so that
// a benchmark process never reads a half-written file of another one (two
// processes make the same bytes).
func writeFile(path string, b []byte) error {
	f, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".*.tmp")
	if err != nil {
		return err
	}
	// CreateTemp makes the file readable by its owner only.
	if err := f.Chmod(0o644); err != nil {
		f.Close()
		os.Remove(f.Name())
		return err
	}
	if _, err := f.Write(b); err != nil {
		f.Close()
		os.Remove(f.Name())
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(f.Name())
		return err
	}
	if err := os.Rename(f.Name(), path); err != nil {
		os.Remove(f.Name())
		return err
	}
	return nil
}
