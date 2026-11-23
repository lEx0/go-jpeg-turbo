// Command gencorpus regenerates testdata/corpus: JPEG files made by
// libjpeg-turbo's cjpeg and jpegtran (must be in PATH) from crops of the
// pictures in testdata, covering every subsampling layout, progressive and
// restart modes, Huffman optimization, grayscale, RGB, scan scripts
// (sequential files with a scan per component, progressive files with
// random scripts) and the unsupported variants (arithmetic coding, 12-bit,
// lossless) that the decoder must reject exactly like image/jpeg does.
//
// Run from the repository root:
//
//	go run ./_tools/gencorpus
package main

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"image/png"
	"log"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
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

	// Sequential files with one scan per component, or two groups, in
	// permuted orders, from cjpeg scan scripts, some with restart
	// intervals. image/jpeg counts the restart interval of a
	// non-interleaved scan in groups of h*v blocks of the component, not in
	// blocks, so that those of a subsampled luma component are rejected
	// (by both decoders alike): the files pin that quirk with a real
	// encoder's output.
	seqScans := []struct {
		src    src
		name   string
		script string
		args   []string
	}{
		{small[7], "420_seqscans0", "0;1;2", []string{"-sample", "2x2"}},
		{small[7], "420_seqscans1", "2;0;1", []string{"-sample", "2x2"}},
		{small[7], "420_seqscans2", "0;1,2", []string{"-sample", "2x2", "-optimize"}},
		{small[7], "420_seqscans3", "0;1;2", []string{"-sample", "2x2", "-restart", "1B"}},
		{small[7], "420_seqscans4", "1,2;0", []string{"-sample", "2x2", "-restart", "2B"}},
		{small[7], "444_seqscans5", "2;1;0", []string{"-sample", "1x1", "-restart", "1B"}},
		{small[7], "rgb_seqscans6", "0;1;2", []string{"-rgb", "-restart", "3B"}},
		{small[7], "441111_seqscans7", "0;1;2", []string{"-sample", "4x4,1x1,1x1"}},
		{small[10], "422_seqscans8", "1;2;0", []string{"-sample", "2x1", "-restart", "2B"}},
		{small[10], "422_seqscans9", "0;2;1", []string{"-sample", "2x1", "-restart", "1"}},
		{small[10], "444_seqscans10", "0;1;2", []string{"-sample", "1x1", "-restart", "1", "-optimize"}},
		{small[10], "420_seqscans11", "2;1;0", []string{"-sample", "2x2", "-restart", "5B"}},
	}
	for _, v := range seqScans {
		script := filepath.Join(tmp, v.name+".txt")
		var b strings.Builder
		for _, group := range strings.Split(v.script, ";") {
			fmt.Fprintf(&b, "%s: 0-63, 0, 0;\n", group)
		}
		if err := os.WriteFile(script, []byte(b.String()), 0o644); err != nil {
			log.Fatal(err)
		}
		run(v.src, v.name, append([]string{"-quality", "80", "-scans", script}, v.args...))
	}

	// Progressive files from jpegtran with random scan scripts
	// (scanScript), made from small files above: per-coefficient and
	// left-out bands, left-out components, Al up to 10, split and
	// incomplete refinements, DC refinements in any grouping, and, for
	// the files without subsampling, restart intervals (image/jpeg
	// rejects those of the AC scans of a subsampled component, see
	// above). cjpeg's -progressive always writes the same script.
	scanSrcs := []string{
		"photo33x31_444_base", "photo33x31_422_base", "photo33x31_420_base", "photo33x31_440_base",
		"photo33x31_411_base", "photo33x31_410_base", "photo33x31_221122_base", "photo33x31_211211_base",
		"photo33x31_121121_base", "photo33x31_222112_base", "photo33x31_412111_base", "photo33x31_141211_base",
		"photo33x31_gray", "photo33x31_rgb", "photo33x31_q100",
		"video150x103_420_base", "video150x103_422_base", "video150x103_444_base", "video150x103_221122_base",
		"video150x103_412111_base", "video150x103_gray", "../video-001.cmyk",
	}
	for i := 0; i < 64; i++ {
		srcName := scanSrcs[i%len(scanSrcs)]
		in := filepath.Join(out, srcName+".jpeg")
		cfg, err := jpeg.DecodeConfig(bytes.NewReader(readFile(in)))
		if err != nil {
			log.Fatalf("%s: %v", in, err)
		}
		nComp := map[color.Model]int{color.GrayModel: 1, color.YCbCrModel: 3, color.RGBAModel: 3, color.CMYKModel: 4}[cfg.ColorModel]
		rng := rand.New(rand.NewSource(int64(i)))
		script := filepath.Join(tmp, fmt.Sprintf("scans%d.txt", i))
		if err := os.WriteFile(script, []byte(scanScript(rng, nComp)), 0o644); err != nil {
			log.Fatal(err)
		}
		args := []string{"-scans", script}
		if strings.Contains(srcName, "444") || strings.Contains(srcName, "gray") || strings.Contains(srcName, "rgb") || strings.Contains(srcName, "cmyk") {
			// Every block, every row, every 3 blocks, in the 3 files
			// made from each.
			args = append(args, "-restart", []string{"1B", "1", "3B"}[i/len(scanSrcs)])
		}
		name := strings.TrimSuffix(strings.TrimPrefix(srcName, "../video-001."), "_base")
		if name == "cmyk" {
			name = "video001cmyk"
		}
		dst := filepath.Join(out, fmt.Sprintf("%s_scans%02d.jpeg", name, i))
		cmd := exec.Command("jpegtran", append(args, "-outfile", dst, in)...)
		if b, err := cmd.CombinedOutput(); err != nil {
			log.Fatalf("jpegtran %s: %v %s", dst, err, strings.TrimSpace(string(b)))
		}
		n++
	}
	log.Printf("wrote %d files to %s", n, out)
}

// scanScript returns a random progressive scan script for nComp
// components, in the format of cjpeg's and jpegtran's -scans, that
// libjpeg's validation accepts: DC first scans (interleaved, one per
// component in a random order, or two groups), AC bands of each component
// (some components and bands left out, now and then one band per
// coefficient) with successive approximation, Al up to 10, refinements
// that stop early or are split into two scans, all interleaved at random,
// and DC refinements in random groupings, possibly late.
func scanScript(rng *rand.Rand, nComp int) string {
	type scan struct {
		comps          []int
		ss, se, ah, al int
	}
	pick := func(v ...int) int { return v[rng.Intn(len(v))] }
	for {
		var scans []scan
		dcAl := pick(0, 0, 1, 2, 3, 5, 8, 10)
		comps := make([]int, nComp)
		for c := range comps {
			comps[c] = c
		}
		var groups [][]int
		switch rng.Intn(4) {
		case 0:
			groups = [][]int{comps}
		case 1:
			for _, c := range rng.Perm(nComp) {
				groups = append(groups, []int{c})
			}
		case 2:
			groups = [][]int{comps}
			if nComp > 1 {
				groups = [][]int{{0}, comps[1:]}
			}
		default:
			p := rng.Perm(nComp)
			k := 1 + rng.Intn(nComp)
			groups = [][]int{p[:k]}
			if k < nComp {
				groups = append(groups, p[k:])
			}
		}
		for _, g := range groups {
			g = append([]int(nil), g...)
			sort.Ints(g)
			scans = append(scans, scan{comps: g, al: dcAl})
		}
		// The AC steps of each band, in order.
		type queue struct {
			c     int
			steps []scan
		}
		var queues []queue
		for _, c := range comps {
			if rng.Intn(6) == 0 {
				continue // No AC at all.
			}
			cuts := rng.Perm(62)[:pick(0, 0, 1, 2, 3, 5, 8)]
			bounds := []int{1, 64}
			for _, x := range cuts {
				bounds = append(bounds, x+2)
			}
			sort.Ints(bounds)
			var bands [][2]int
			for i := 0; i+1 < len(bounds); i++ {
				bands = append(bands, [2]int{bounds[i], bounds[i+1] - 1})
			}
			if rng.Intn(8) == 0 {
				// One band per coefficient over a range.
				lo := 1 + rng.Intn(63)
				hi := min(63, lo+rng.Intn(31))
				bands = nil
				for k := lo; k <= hi; k++ {
					bands = append(bands, [2]int{k, k})
				}
				if lo > 1 {
					bands = append(bands, [2]int{1, lo - 1})
				}
				if hi < 63 {
					bands = append(bands, [2]int{hi + 1, 63})
				}
				rng.Shuffle(len(bands), func(i, j int) { bands[i], bands[j] = bands[j], bands[i] })
			}
			for _, b := range bands {
				if rng.Intn(10) == 0 {
					continue // A band left out.
				}
				al := pick(0, 0, 1, 1, 2, 3, 4, 6, 9, 10)
				steps := []scan{{comps: []int{c}, ss: b[0], se: b[1], al: al}}
				stop := 0
				if al > 0 {
					stop = min(pick(0, 0, 0, 1, 2), al)
				}
				for a := al; a > stop; a-- {
					steps = append(steps, scan{comps: []int{c}, ss: b[0], se: b[1], ah: a, al: a - 1})
				}
				queues = append(queues, queue{c, steps})
			}
		}
		for {
			var live []int
			for i, q := range queues {
				if len(q.steps) > 0 {
					live = append(live, i)
				}
			}
			if len(live) == 0 {
				break
			}
			q := &queues[live[rng.Intn(len(live))]]
			s := q.steps[0]
			q.steps = q.steps[1:]
			if s.ah != 0 && s.se > s.ss && rng.Intn(3) == 0 {
				// A refinement split in two.
				m := s.ss + rng.Intn(s.se-s.ss)
				a, b := s, s
				a.se, b.ss = m, m+1
				scans = append(scans, a, b)
			} else {
				scans = append(scans, s)
			}
		}
		// The DC refinements, after the DC first scans, in order.
		stop := 0
		if dcAl > 0 {
			stop = min(pick(0, 0, 0, 1), dcAl)
		}
		var tail []scan
		for a := dcAl; a > stop; a-- {
			if rng.Intn(2) == 0 {
				tail = append(tail, scan{comps: comps, ah: a, al: a - 1})
			} else {
				for _, c := range comps {
					tail = append(tail, scan{comps: []int{c}, ah: a, al: a - 1})
				}
			}
		}
		pos := len(groups)
		for _, s := range tail {
			pos += rng.Intn(len(scans) - pos + 1)
			scans = append(scans[:pos], append([]scan{s}, scans[pos:]...)...)
			pos++
		}
		if len(scans) > 100 {
			continue
		}
		var b strings.Builder
		for _, s := range scans {
			cs := make([]string, len(s.comps))
			for i, c := range s.comps {
				cs[i] = strconv.Itoa(c)
			}
			fmt.Fprintf(&b, "%s: %d-%d, %d, %d;\n", strings.Join(cs, ","), s.ss, s.se, s.ah, s.al)
		}
		return b.String()
	}
}

func readFile(path string) []byte {
	b, err := os.ReadFile(path)
	if err != nil {
		log.Fatal(err)
	}
	return b
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
