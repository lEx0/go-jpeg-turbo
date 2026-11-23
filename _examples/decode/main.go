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

// Command decode decodes JPEG files with github.com/lEx0/go-jpeg-turbo/jpeg,
// prints what each header declares and what Decode returned, and saves a PNG
// copy of every picture. The package is a drop-in replacement for
// image/jpeg: with "image/jpeg" in the import list this program compiles
// and decodes the same pixels, only more slowly.
//
// Run from the repository root:
//
//	go run ./_examples/decode                      # a few files from testdata/
//	go run ./_examples/decode -out /tmp/png a.jpg  # your own files
package main

import (
	"bytes"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/lEx0/go-jpeg-turbo/jpeg" // was: "image/jpeg"
)

// defaultFiles cover the image types Decode returns.
var defaultFiles = []string{
	"testdata/photo-960x1280.jpg",                    // baseline, 4:4:4
	"testdata/bench/photo-960x1280-q85-420-prog.jpg", // progressive, 4:2:0
	"testdata/video-005.gray.jpeg",                   // one component
	"testdata/video-001.rgb.jpeg",                    // RGB components, no YCbCr transform
	"testdata/video-001.cmyk.jpeg",                   // Adobe CMYK
}

func main() {
	out := flag.String("out", filepath.Join(os.TempDir(), "go-jpeg-turbo-decode"), "directory for the PNG copies")
	maxPixels := flag.Int64("max-pixels", 64<<20, "reject images with more pixels than this before decoding them")
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "usage: decode [flags] [file.jpg ...]\n\n"+
			"Without arguments it decodes a few files from testdata/ (run it from the repository root).\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()
	files := flag.Args()
	if len(files) == 0 {
		files = defaultFiles
	}
	if err := os.MkdirAll(*out, 0o755); err != nil {
		log.Fatal(err)
	}

	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "file\theader\tdecoded to\tdecode time")
	var failures []string
	for _, name := range files {
		row, err := decodeFile(name, *out, *maxPixels)
		if err != nil {
			// As with image/jpeg, the decoder fails with a
			// jpeg.FormatError (not valid JPEG, or cut short in the
			// compressed pixels), a jpeg.UnsupportedError (valid JPEG
			// using a feature image/jpeg does not implement either, such
			// as arithmetic coding) or io.ErrUnexpectedEOF (cut short in
			// the header).
			failures = append(failures, fmt.Sprintf("%s: %v", name, err))
			continue
		}
		fmt.Fprintln(tw, row)
	}
	tw.Flush()
	for _, f := range failures {
		fmt.Println(f)
	}
	fmt.Println("PNG copies in", *out)
	if len(failures) > 0 {
		os.Exit(1)
	}
}

// decodeFile decodes one file and saves it as PNG in outDir. It returns a
// tab-separated row for the summary table.
func decodeFile(name, outDir string, maxPixels int64) (string, error) {
	data, err := os.ReadFile(name)
	if err != nil {
		return "", err
	}

	// DecodeConfig reads only the header. Checking the dimensions first
	// keeps a small hostile file from making Decode allocate gigabytes.
	cfg, err := jpeg.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return "", err
	}
	if int64(cfg.Width)*int64(cfg.Height) > maxPixels {
		return "", fmt.Errorf("%dx%d is more than -max-pixels", cfg.Width, cfg.Height)
	}

	start := time.Now()
	img, err := jpeg.Decode(bytes.NewReader(data))
	elapsed := time.Since(start)
	if err != nil {
		return "", err
	}

	base := strings.TrimSuffix(filepath.Base(name), filepath.Ext(name))
	if err := savePNG(filepath.Join(outDir, base+".png"), img); err != nil {
		return "", err
	}
	return fmt.Sprintf("%s\t%dx%d %s\t%s\t%.2f ms", name, cfg.Width, cfg.Height,
		modelName(cfg.ColorModel), imageKind(img), elapsed.Seconds()*1e3), nil
}

// modelName names the color models DecodeConfig reports.
func modelName(m color.Model) string {
	switch m {
	case color.YCbCrModel:
		return "YCbCr"
	case color.GrayModel:
		return "gray"
	case color.RGBAModel:
		return "RGB"
	case color.CMYKModel:
		return "CMYK"
	}
	return "unknown"
}

var ratios = map[image.YCbCrSubsampleRatio]string{
	image.YCbCrSubsampleRatio444: "4:4:4",
	image.YCbCrSubsampleRatio422: "4:2:2",
	image.YCbCrSubsampleRatio420: "4:2:0",
	image.YCbCrSubsampleRatio440: "4:4:0",
	image.YCbCrSubsampleRatio411: "4:1:1",
	image.YCbCrSubsampleRatio410: "4:1:0",
}

// imageKind names the concrete type Decode returned: *image.YCbCr (with
// the file's chroma subsampling), *image.Gray, *image.RGBA or *image.CMYK.
func imageKind(img image.Image) string {
	if m, ok := img.(*image.YCbCr); ok {
		return fmt.Sprintf("%T %s", img, ratios[m.SubsampleRatio])
	}
	return fmt.Sprintf("%T", img)
}

func savePNG(path string, img image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	// The copies are for looking at, so favor speed over size.
	enc := png.Encoder{CompressionLevel: png.BestSpeed}
	if err := enc.Encode(f, img); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
