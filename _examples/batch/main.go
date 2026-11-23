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

// Command batch re-encodes every JPEG file under a directory at one quality,
// on several goroutines at once, and prints a summary. Like image/jpeg's,
// Decode, DecodeConfig and Encode are safe for concurrent use: each call
// works on its own state, and Encode takes its buffers from a sync.Pool.
//
// Files that Decode rejects are skipped and counted by error message.
// testdata/corpus has such files on purpose: arithmetic coding, 12-bit
// samples, lossless JPEG and sampling layouts that image/jpeg rejects too.
//
// Run from the repository root:
//
//	go run ./_examples/batch                          # everything under testdata/
//	go run ./_examples/batch -dir ~/Pictures -q 80 -workers 4
//	go run ./_examples/batch -out ''                  # encode, but write nothing
//	go run -race ./_examples/batch                    # with the race detector
package main

import (
	"bytes"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"maps"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/lEx0/go-jpeg-turbo/jpeg" // was: "image/jpeg"
)

type result struct {
	path     string
	pixels   int64 // pixels in the decoded image
	in, out  int64 // file sizes in bytes
	err      error // the first error, if any
	rejected bool  // err comes from Decode, so the file's contents are at fault
}

func main() {
	dir := flag.String("dir", "testdata", "directory to search for .jpg and .jpeg files, subdirectories included")
	out := flag.String("out", filepath.Join(os.TempDir(), "go-jpeg-turbo-batch"),
		"directory for the re-encoded files, laid out like -dir; empty to discard them")
	quality := flag.Int("q", 80, "JPEG quality, 1 to 100")
	workers := flag.Int("workers", runtime.GOMAXPROCS(0), "number of goroutines")
	flag.Parse()
	if *quality < 1 || *quality > 100 {
		log.Fatal("-q must be from 1 to 100")
	}
	if *workers < 1 {
		log.Fatal("-workers must be at least 1")
	}

	var paths []string
	err := filepath.WalkDir(*dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		switch strings.ToLower(filepath.Ext(path)) {
		case ".jpg", ".jpeg":
			if d.Type().IsRegular() {
				paths = append(paths, path)
			}
		}
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}

	start := time.Now()
	jobs := make(chan string)
	results := make(chan result)
	var wg sync.WaitGroup
	for range *workers {
		wg.Go(func() {
			// One output buffer per goroutine, reused for every file.
			var buf bytes.Buffer
			for path := range jobs {
				results <- reencode(path, *dir, *out, *quality, &buf)
			}
		})
	}
	go func() {
		for _, p := range paths {
			jobs <- p
		}
		close(jobs)
		wg.Wait()
		close(results)
	}()

	var done, failed int
	var total result                  // pixels and sizes of the re-encoded files
	rejected := map[string][]string{} // decoder error text -> files
	for r := range results {
		switch {
		case r.err == nil:
			done++
			total.pixels += r.pixels
			total.in += r.in
			total.out += r.out
		case r.rejected:
			rejected[r.err.Error()] = append(rejected[r.err.Error()], r.path)
		default:
			failed++
			fmt.Fprintf(os.Stderr, "%s: %v\n", r.path, r.err)
		}
	}
	elapsed := time.Since(start)

	fmt.Printf("%d JPEG files under %s, quality %d, workers %d\n", len(paths), *dir, *quality, *workers)
	fmt.Printf("re-encoded %d: %.1f megapixels, %.1f MiB -> %.1f MiB, in %v (%.0f megapixels/s)\n",
		done, float64(total.pixels)/1e6, float64(total.in)/(1<<20), float64(total.out)/(1<<20),
		elapsed.Round(time.Millisecond), float64(total.pixels)/1e6/elapsed.Seconds())
	for _, msg := range slices.Sorted(maps.Keys(rejected)) {
		files := rejected[msg]
		slices.Sort(files)
		fmt.Printf("skipped %d: %s (%s, ...)\n", len(files), msg, files[0])
	}
	if *out != "" {
		fmt.Println("output in", *out)
	}
	if failed > 0 {
		os.Exit(1)
	}
}

// reencode decodes the file at path and encodes it again into buf. Unless
// outDir is empty, the result goes to the same path relative to outDir as
// path has relative to dir.
func reencode(path, dir, outDir string, quality int, buf *bytes.Buffer) result {
	r := result{path: path}
	data, err := os.ReadFile(path)
	if err != nil {
		r.err = err
		return r
	}
	r.in = int64(len(data))

	img, err := jpeg.Decode(bytes.NewReader(data))
	if err != nil {
		// The reader holds the whole file, so any error is about its
		// contents: a jpeg.FormatError (not valid JPEG, or cut short in
		// the compressed pixels), a jpeg.UnsupportedError (valid JPEG
		// using a feature that image/jpeg does not implement either) or
		// io.ErrUnexpectedEOF (cut short in the header).
		r.err, r.rejected = err, true
		return r
	}
	b := img.Bounds()
	r.pixels = int64(b.Dx()) * int64(b.Dy())

	buf.Reset()
	if err := jpeg.Encode(buf, img, &jpeg.Options{Quality: quality}); err != nil {
		r.err = err
		return r
	}
	r.out = int64(buf.Len())
	if outDir == "" {
		return r
	}

	rel, err := filepath.Rel(dir, path)
	if err != nil {
		r.err = err
		return r
	}
	if rel == "." { // -dir names the file itself
		rel = filepath.Base(path)
	}
	dst := filepath.Join(outDir, rel)
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		r.err = err
		return r
	}
	if err := os.WriteFile(dst, buf.Bytes(), 0o644); err != nil {
		r.err = err
	}
	return r
}
