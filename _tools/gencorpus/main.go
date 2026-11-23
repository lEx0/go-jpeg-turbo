// Command gencorpus regenerates testdata/corpus: JPEG files made by
// libjpeg-turbo's cjpeg (must be in PATH) from crops of the pictures in
// testdata, covering every subsampling layout, progressive and restart
// modes, Huffman optimization, grayscale, RGB and the unsupported variants
// (arithmetic coding, 12-bit, lossless) that the decoder must reject exactly
// like image/jpeg does.
//
// Run from the repository root:
//
//	go run ./_tools/gencorpus
package main

import (
	"fmt"
	"image"
	"image/draw"
	"image/jpeg"
	"image/png"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

type variant struct {
	name string
	args []string
}

func main() {
	out := filepath.Join("testdata", "corpus")
	if err := os.RemoveAll(out); err != nil {
		log.Fatal(err)
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		log.Fatal(err)
	}
	tmp, err := os.MkdirTemp("", "gencorpus")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(tmp)

	photo := load("testdata/photo-960x1280.jpg")
	video := load("testdata/video-001.png")

	type src struct {
		name string
		img  image.Image
	}
	var small []src
	for _, sz := range []image.Point{{1, 1}, {2, 3}, {7, 5}, {8, 8}, {9, 17}, {16, 16}, {17, 15}, {33, 31}, {64, 1}, {1, 64}} {
		small = append(small, src{fmt.Sprintf("photo%dx%d", sz.X, sz.Y), crop(photo, image.Rect(300, 400, 300+sz.X, 400+sz.Y))})
	}
	small = append(small, src{"video150x103", video})
	large := []src{
		{"photo257x131", crop(photo, image.Rect(100, 200, 357, 331))},
		{"photo640x480", crop(photo, image.Rect(160, 300, 800, 780))},
	}

	samplings := []variant{
		{"444", []string{"-sample", "1x1"}},
		{"422", []string{"-sample", "2x1"}},
		{"420", []string{"-sample", "2x2"}},
		{"440", []string{"-sample", "1x2"}},
		{"411", []string{"-sample", "4x1"}},
		{"410", []string{"-sample", "4x2"}},
		{"221122", []string{"-sample", "2x2,1x1,2x2"}},
		{"211211", []string{"-sample", "2x1,1x2,1x1"}},
		{"121121", []string{"-sample", "1x2,1x1,2x1"}},
		{"222112", []string{"-sample", "2x2,2x1,1x2"}},
		{"412111", []string{"-sample", "4x1,2x1,1x1"}},
		{"141211", []string{"-sample", "1x4,1x2,1x1"}}, // v == 4
		{"422211", []string{"-sample", "4x2,2x1,1x1"}}, // total sampling factors 11 > 10
		{"311111", []string{"-sample", "3x1,1x1,1x1"}}, // h == 3: unsupported
	}
	modes := []variant{
		{"base", nil},
		{"prog", []string{"-progressive"}},
		{"opt", []string{"-optimize"}},
		{"rst1", []string{"-restart", "1B"}},
		{"rst3", []string{"-restart", "3B"}},
		{"progrst", []string{"-progressive", "-restart", "2B"}},
	}
	extra := []variant{
		{"gray", []string{"-grayscale"}},
		{"grayprog", []string{"-grayscale", "-progressive"}},
		{"grayrst", []string{"-grayscale", "-restart", "1B"}},
		{"gray22", []string{"-grayscale", "-sample", "2x2"}},
		{"rgb", []string{"-rgb"}},
		{"rgbprog", []string{"-rgb", "-progressive"}},
		{"rgb420", []string{"-rgb", "-sample", "2x2"}},
		{"q1", []string{"-quality", "1"}},
		{"q10", []string{"-quality", "10"}},
		{"q50", []string{"-quality", "50"}},
		{"q95", []string{"-quality", "95"}},
		{"q100", []string{"-quality", "100"}},
		{"q100prog", []string{"-quality", "100", "-progressive"}},
		{"q100_444", []string{"-quality", "100", "-sample", "1x1"}},
		{"fast", []string{"-dct", "fast"}},
		{"float", []string{"-dct", "float"}},
		{"smooth", []string{"-smooth", "40"}},
		{"arith", []string{"-arithmetic"}},
		{"p12", []string{"-precision", "12"}},
		{"lossless", []string{"-lossless", "1"}},
	}

	n := 0
	run := func(s src, name string, args []string) {
		in := filepath.Join(tmp, s.name+".png")
		if _, err := os.Stat(in); err != nil {
			save(in, s.img)
		}
		dst := filepath.Join(out, s.name+"_"+name+".jpeg")
		cmd := exec.Command("cjpeg", append(append([]string{}, args...), "-outfile", dst, in)...)
		if b, err := cmd.CombinedOutput(); err != nil {
			log.Printf("skip %s %s: %v %s", s.name, name, err, strings.TrimSpace(string(b)))
			os.Remove(dst)
			return
		}
		n++
	}
	for _, s := range small {
		for _, sm := range samplings {
			for _, m := range modes {
				run(s, sm.name+"_"+m.name, append(append([]string{"-quality", "80"}, sm.args...), m.args...))
			}
		}
		for _, e := range extra {
			run(s, e.name, e.args)
		}
	}
	for _, s := range large {
		for _, sm := range samplings[:6] {
			run(s, sm.name, append([]string{"-quality", "85"}, sm.args...))
			run(s, sm.name+"_prog", append([]string{"-quality", "85", "-progressive"}, sm.args...))
		}
		for _, e := range extra {
			switch e.name {
			case "gray", "grayprog", "rgb", "q50", "q100", "q100prog", "arith":
				run(s, e.name, e.args)
			}
		}
		run(s, "rst8", []string{"-restart", "8B"})
		run(s, "opt", []string{"-optimize"})
	}
	log.Printf("wrote %d files to %s", n, out)
}

func load(path string) image.Image {
	f, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	var m image.Image
	if strings.HasSuffix(path, ".png") {
		m, err = png.Decode(f)
	} else {
		m, err = jpeg.Decode(f)
	}
	if err != nil {
		log.Fatal(err)
	}
	return m
}

func crop(m image.Image, r image.Rectangle) image.Image {
	dst := image.NewNRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
	draw.Draw(dst, dst.Bounds(), m, r.Min, draw.Src)
	return dst
}

func save(path string, m image.Image) {
	f, err := os.Create(path)
	if err != nil {
		log.Fatal(err)
	}
	if err := png.Encode(f, m); err != nil {
		log.Fatal(err)
	}
	if err := f.Close(); err != nil {
		log.Fatal(err)
	}
}
