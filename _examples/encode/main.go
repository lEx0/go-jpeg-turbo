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

// Command encode converts images to JPEG with
// github.com/lEx0/go-jpeg-turbo/jpeg at a few qualities and prints the size
// and the encoding time of each file. The inputs can be JPEG, PNG or GIF:
// image.Decode picks the decoder from the first bytes of the file.
//
// Run from the repository root:
//
//	go run ./_examples/encode                                # testdata/ files
//	go run ./_examples/encode -q 60,85 -out /tmp/jpeg a.png  # your own files
package main

import (
	"bytes"
	"flag"
	"fmt"
	"image"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	// image.Decode only knows the formats whose packages are linked in.
	_ "image/gif"
	_ "image/png"

	// This import also registers the "jpeg" format with package image, so
	// image.Decode reads JPEG input with this package. A program that only
	// calls image.Decode needs no more than the blank import
	// _ "github.com/lEx0/go-jpeg-turbo/jpeg" in place of _ "image/jpeg".
	"github.com/lEx0/go-jpeg-turbo/jpeg" // was: "image/jpeg"
)

// defaultFiles are a JPEG to re-encode and a PNG to convert.
var defaultFiles = []string{
	"testdata/photo-960x1280.jpg", // a JPEG photo, decoded to *image.YCbCr
	"testdata/video-001.png",      // a PNG, decoded to *image.RGBA
}

func main() {
	out := flag.String("out", filepath.Join(os.TempDir(), "go-jpeg-turbo-encode"), "directory for the JPEG files")
	qualityList := flag.String("q", "50,75,90", "comma-separated qualities, 1 to 100")
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "usage: encode [flags] [image ...]\n\n"+
			"Without arguments it encodes two files from testdata/ (run it from the repository root).\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()
	files := flag.Args()
	if len(files) == 0 {
		files = defaultFiles
	}
	var qualities []int
	for _, s := range strings.Split(*qualityList, ",") {
		q, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil || q < 1 || q > 100 {
			log.Fatalf("-q: %q is not a quality from 1 to 100", s)
		}
		qualities = append(qualities, q)
	}
	if err := os.MkdirAll(*out, 0o755); err != nil {
		log.Fatal(err)
	}

	for _, name := range files {
		if err := encodeFile(name, *out, qualities); err != nil {
			log.Fatalf("%s: %v", name, err)
		}
	}
	fmt.Println("JPEG files in", *out)
}

// encodeFile decodes the image in the file name and writes it to outDir as
// JPEG once per quality.
func encodeFile(name, outDir string, qualities []int) error {
	data, err := os.ReadFile(name)
	if err != nil {
		return err
	}
	img, format, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return err
	}
	b := img.Bounds()
	pixels := float64(b.Dx()) * float64(b.Dy())
	fmt.Printf("%s: %s, %dx%d, %T, %.1f KiB\n", name, format, b.Dx(), b.Dy(), img, float64(len(data))/1024)

	base := strings.TrimSuffix(filepath.Base(name), filepath.Ext(name))
	var buf bytes.Buffer
	for _, q := range qualities {
		buf.Reset()
		// Like image/jpeg, Encode writes baseline JPEG: 4:2:0 for color
		// images, a single component for *image.Gray. A nil *Options
		// means jpeg.DefaultQuality (75).
		start := time.Now()
		if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: q}); err != nil {
			return err
		}
		elapsed := time.Since(start)

		dst := fmt.Sprintf("%s-q%d.jpg", base, q)
		if err := os.WriteFile(filepath.Join(outDir, dst), buf.Bytes(), 0o644); err != nil {
			return err
		}
		fmt.Printf("  quality %3d: %7.1f KiB  %5.2f bits/pixel  %7.2f ms  %s\n",
			q, float64(buf.Len())/1024, float64(8*buf.Len())/pixels, elapsed.Seconds()*1e3, dst)
	}
	return nil
}
