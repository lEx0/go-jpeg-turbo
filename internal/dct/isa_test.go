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

package dct

import (
	"bytes"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// TestKernelISA audits the machine code of the amd64 kernels of this
// package and of internal/ycc against the instructions the level of each
// kernel (the suffix of its name: SSE41, AVX2 or AVX512) may use. The
// runtime tests cannot see these defects: on a machine with AVX2 an AVX
// instruction in an SSE4.1 kernel runs fine at JPEG_TURBO_LEVEL=sse41 (a
// CPU without AVX would raise SIGILL), a missing VZEROUPPER only costs
// time, and a constant that is aligned by luck in one binary may not be
// in the next. Whatever the host, the test builds both packages for
// linux/amd64, takes the object of their assembly out of the package
// archive and disassembles it with go tool objdump, which prints the raw
// bytes (a C4 or C5 byte after the legacy and REX prefixes starts a VEX
// prefix, 62 an EVEX one).
//
// The rules, from the claims at the top of dct_amd64.s and ycc_amd64.s
// and the CPU features internal/cpuinfo checks for each level:
//   - SSE41 kernels: legacy encodings only (no VEX or EVEX prefix, so no
//     ymm, zmm or mask register) and nothing beyond SSE4.1 (no POPCNT,
//     SSE4.2 or TZCNT);
//   - AVX2 kernels: no EVEX prefix (no zmm, mask or x/ymm16-31 register),
//     nothing beyond AVX2, BMI1, BMI2 and FMA (no F16C, AVX-VNNI, LZCNT);
//   - AVX512 kernels: nothing beyond AVX-512 F, BW, CD, DQ and VL, and
//     AVX-512 VNNI (EVEX VPDPWSSD and VPDPBUSD) only in a package whose
//     AVX512 kernels run only with it (internal/ycc, hasAVX512Kernels);
//   - AVX2 and AVX512 kernels: no legacy SSE instruction (an SSE/AVX
//     transition while the upper halves are dirty), and VZEROUPPER right
//     before every RET of a kernel that touches a ymm or zmm register;
//   - every kernel: no BP, R14 or R15 (apart from the frame the
//     assembler adds), no CALL;
//   - memory operands that must be aligned (those of legacy SSE
//     instructions other than the unaligned loads and the 4- and 8-byte
//     ones, and of VMOVDQA and the like) are RIP-relative constants at an
//     offset that is a multiple of the alignment, in a symbol the linker
//     aligns at least that much (linkerAlign): Go aligns stack frames to
//     8 bytes and the kernels' arguments to their element size only.
//
// Adapted from the llvm-objdump encoding audits run on the amd64 kernels
// during their verification; this version needs the go command only.
func TestKernelISA(t *testing.T) {
	goTool, err := exec.LookPath("go")
	if err != nil {
		t.Skip("no go command")
	}
	for _, p := range isaPackages {
		t.Run(path.Base(p.path), func(t *testing.T) {
			objs := isaObjects(t, goTool, p.path)
			if len(objs) == 0 {
				t.Fatalf("no assembly object in the linux/amd64 archive of %s", p.path)
			}
			levels := map[string]int{}
			for _, obj := range objs {
				funcs, err := parseObjdump(string(isaGo(t, goTool, "tool", "objdump", obj)))
				if err != nil {
					t.Fatalf("%s: %v", filepath.Base(obj), err)
				}
				text, data, err := parseNm(string(isaGo(t, goTool, "tool", "nm", "-size", obj)))
				if err != nil {
					t.Fatalf("%s: %v", filepath.Base(obj), err)
				}
				for name := range text {
					if !slices.ContainsFunc(funcs, func(f *isaFunc) bool { return f.name == name }) {
						t.Errorf("%s: function %s not in the disassembly", filepath.Base(obj), name)
					}
				}
				for _, f := range funcs {
					level := isaLevel(f.name)
					if level == "" {
						t.Errorf("%s: the name does not end in SSE41, AVX2 or AVX512, so its rules are unknown", f.name)
						continue
					}
					levels[level]++
					probs := auditKernel(f, level, p.vnni, text[f.name], data)
					for i, pr := range probs {
						if i == 10 {
							t.Errorf("%s: %d more", f.name, len(probs)-i)
							break
						}
						t.Errorf("%s: %s", f.name, pr)
					}
					t.Logf("%-28s %-6s %s", f.name[strings.LastIndex(f.name, ".")+1:], level, encodingCounts(f))
				}
			}
			for _, l := range []string{"SSE41", "AVX2", "AVX512"} {
				if levels[l] == 0 {
					t.Errorf("no %s kernel found: the audit may not have seen the kernels", l)
				}
			}
		})
	}
}

// isaPackages are the packages with amd64 kernels. vnni reports that
// their AVX512 kernels run only on machines with AVX-512 VNNI.
var isaPackages = []struct {
	path string
	vnni bool
}{
	{"github.com/lEx0/go-jpeg-turbo/internal/dct", false},
	{"github.com/lEx0/go-jpeg-turbo/internal/ycc", true},
}

// isaGo runs the go command for linux/amd64 and returns its output.
func isaGo(t *testing.T, goTool string, args ...string) []byte {
	t.Helper()
	cmd := exec.Command(goTool, args...)
	// The kernels are built for linux/amd64 whatever the host, and
	// without the flags of this test run (-tags purego drops them; the
	// -tags= of isaObjects also overrides a GOFLAGS of the go env file).
	cmd.Env = append(os.Environ(), "GOOS=linux", "GOARCH=amd64", "CGO_ENABLED=0", "GOFLAGS=")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("go %s: %v\n%s", strings.Join(args, " "), err, stderr.Bytes())
	}
	return out
}

// isaObjects builds pkg for linux/amd64 and writes the objects of its
// assembly files (the members of the package archive other than the
// export data and the compiled Go code) to a temporary directory.
//
// It also reads the source directory of that build and its source files.
// go test caches a passing result against the test binary and the files
// the test opens inside the module, not against the build cache, where
// the archive is: the test binary of a non-amd64 host does not contain
// the amd64 kernels, and that of this package never contains
// internal/ycc's, so without these reads a cached pass would outlive an
// edit of the assembly it audited. For a file the test opens, go test
// records its size and modification time; for a directory, the names,
// sizes and modification times of its entries. Reading the directory
// therefore also covers files added, removed or renamed since the cached
// run, which that run's file list cannot name, and edits of files that
// the build constraints exclude for now. The reads of the files stay
// because go test does not cache a result that depends on a file
// modified too recently for its modification time to be trusted, a
// check it does not make for the entries of a directory.
func isaObjects(t *testing.T, goTool, pkg string) []string {
	t.Helper()
	list := strings.Split(string(isaGo(t, goTool, "list", "-tags=", "-export", "-f",
		"{{.Export}}\n{{.Dir}}\n{{range .GoFiles}}{{.}}\n{{end}}{{range .SFiles}}{{.}}\n{{end}}{{range .HFiles}}{{.}}\n{{end}}", pkg)), "\n")
	if len(list) < 3 {
		t.Fatalf("go list %s: unexpected output %q", pkg, list)
	}
	archive, src := list[0], list[1]
	if _, err := os.ReadDir(src); err != nil {
		t.Fatal(err)
	}
	for _, f := range list[2:] {
		if f == "" {
			continue
		}
		if _, err := os.ReadFile(filepath.Join(src, f)); err != nil {
			t.Fatal(err)
		}
	}
	b, err := os.ReadFile(archive)
	if err != nil {
		t.Fatal(err)
	}
	members, err := arMembers(b)
	if err != nil {
		t.Fatalf("%s: %v", archive, err)
	}
	dir := t.TempDir()
	var objs []string
	for _, m := range members {
		if m.name == "__.PKGDEF" || m.name == "_go_.o" {
			continue
		}
		obj := filepath.Join(dir, m.name)
		if err := os.WriteFile(obj, m.data, 0o644); err != nil {
			t.Fatal(err)
		}
		objs = append(objs, obj)
	}
	return objs
}

type arMember struct {
	name string
	data []byte
}

// arMembers splits a Unix ar archive, the format of Go package archives.
func arMembers(b []byte) ([]arMember, error) {
	const magic = "!<arch>\n"
	if !bytes.HasPrefix(b, []byte(magic)) {
		return nil, errors.New("not an ar archive")
	}
	var ms []arMember
	for p := b[len(magic):]; len(p) > 0; {
		if len(p) < 60 || string(p[58:60]) != "`\n" {
			return nil, errors.New("bad member header")
		}
		size, err := strconv.Atoi(strings.TrimSpace(string(p[48:58])))
		if err != nil || size < 0 || size > len(p)-60 {
			return nil, fmt.Errorf("bad member size %q", p[48:58])
		}
		ms = append(ms, arMember{strings.TrimRight(string(p[:16]), " /"), p[60 : 60+size]})
		p = p[60+size:]
		if size%2 == 1 && len(p) > 0 {
			p = p[1:] // members start at even offsets
		}
	}
	return ms, nil
}

type isaFunc struct {
	name  string
	insns []isaInsn
}

type isaInsn struct {
	pc     uint64
	raw    []byte
	op     string // mnemonic, Go syntax
	args   string
	relocs []isaReloc
}

type isaReloc struct {
	end int    // end of the relocated field in the instruction
	typ string // R_PCREL, R_CALL...
	sym string
	add int64
}

func (in isaInsn) String() string { return strings.TrimSpace(in.op + " " + in.args) }

var relocRE = regexp.MustCompile(`^\[(-?\d+):(-?\d+)\](\w+)(?::(.*))?$`)

// parseObjdump parses the output of go tool objdump: "TEXT name(SB) file"
// headers, then one line per instruction with tab-separated fields
// "file:line", pc, the hex bytes, the instruction and its relocations.
// go tool objdump decodes VZEROUPPER (C5 F8 77) with one byte too many,
// so that VZEROUPPER; RET shows as one 4-byte VZEROUPPER: such a line is
// split in two.
func parseObjdump(out string) ([]*isaFunc, error) {
	var funcs []*isaFunc
	var cur *isaFunc
	for _, line := range strings.Split(out, "\n") {
		if rest, ok := strings.CutPrefix(line, "TEXT "); ok {
			name, _, _ := strings.Cut(rest, "(SB)")
			cur = &isaFunc{name: name}
			funcs = append(funcs, cur)
			continue
		}
		if strings.TrimSpace(line) == "" {
			continue
		}
		var f []string
		for _, s := range strings.Split(line, "\t") {
			if s = strings.TrimSpace(s); s != "" {
				f = append(f, s)
			}
		}
		if cur == nil || len(f) < 4 {
			return nil, fmt.Errorf("unexpected line %q", line)
		}
		pc, err := strconv.ParseUint(strings.TrimPrefix(f[1], "0x"), 16, 64)
		if err != nil {
			return nil, fmt.Errorf("line %q: %v", line, err)
		}
		raw, err := hex.DecodeString(f[2])
		if err != nil || len(raw) == 0 {
			return nil, fmt.Errorf("line %q: bad bytes", line)
		}
		in := isaInsn{pc: pc, raw: raw}
		in.op, in.args, _ = strings.Cut(f[3], " ")
		for _, s := range strings.Fields(strings.Join(f[4:], " ")) {
			m := relocRE.FindStringSubmatch(s)
			if m == nil {
				return nil, fmt.Errorf("line %q: bad relocation %q", line, s)
			}
			r := isaReloc{typ: m[3], sym: m[4]}
			r.end, _ = strconv.Atoi(m[2])
			if i := strings.LastIndex(r.sym, "+"); i >= 0 {
				if add, err := strconv.ParseInt(r.sym[i+1:], 10, 64); err == nil {
					r.sym, r.add = r.sym[:i], add
				}
			}
			in.relocs = append(in.relocs, r)
		}
		if n := len(raw); in.op == "VZEROUPPER" && n > 3 && raw[n-2] == 0x77 && raw[n-1] == 0xc3 {
			cur.insns = append(cur.insns, isaInsn{pc: pc, raw: raw[:n-1], op: "VZEROUPPER"})
			in = isaInsn{pc: pc + uint64(n-1), raw: raw[n-1:], op: "RET"}
		}
		cur.insns = append(cur.insns, in)
	}
	return funcs, nil
}

// isaSym is a symbol of an object file.
type isaSym struct{ addr, size int64 }

// parseNm parses the output of go tool nm -size: the text symbols and the
// sizes of the data symbols.
func parseNm(out string) (text map[string]isaSym, data map[string]int64, err error) {
	text, data = map[string]isaSym{}, map[string]int64{}
	for _, line := range strings.Split(out, "\n") {
		f := strings.Fields(line)
		if len(f) != 4 {
			continue // undefined symbols have no address
		}
		addr, err1 := strconv.ParseInt(f[0], 16, 64)
		size, err2 := strconv.ParseInt(f[1], 10, 64)
		if err1 != nil || err2 != nil {
			return nil, nil, fmt.Errorf("unexpected line %q", line)
		}
		switch f[2] {
		case "T", "t":
			text[f[3]] = isaSym{addr, size}
		case "R", "r", "D", "d", "B", "b":
			data[f[3]] = size
		}
	}
	return text, data, nil
}

// isaLevel returns the level of a kernel from its name, "" if none.
func isaLevel(name string) string {
	for _, l := range []string{"SSE41", "AVX2", "AVX512"} {
		if strings.HasSuffix(name, l) {
			return l
		}
	}
	return ""
}

// isaEncoding returns "legacy", "VEX" or "EVEX": the byte after the
// legacy prefixes and a REX prefix starts a VEX (C4, C5) or EVEX (62)
// prefix, which 64-bit mode does not decode as LES, LDS or BOUND.
func isaEncoding(raw []byte) string {
	i := 0
	for i < len(raw) && strings.IndexByte("\x26\x2e\x36\x3e\x64\x65\x66\x67\xf0\xf2\xf3", raw[i]) >= 0 {
		i++
	}
	if i < len(raw) && raw[i]&0xf0 == 0x40 {
		i++
	}
	if i < len(raw) {
		switch raw[i] {
		case 0xc4, 0xc5:
			return "VEX"
		case 0x62:
			return "EVEX"
		}
	}
	return "legacy"
}

var (
	vecRegRE   = regexp.MustCompile(`\b([XYZ])([0-9]+)\b`)
	upperRegRE = regexp.MustCompile(`\b[YZ][0-9]+\b`)
	frameRegRE = regexp.MustCompile(`\b(BP|R14|R15)B?\b`)
)

// isaDenied returns why an instruction may not be used at level (its
// extension has a CPUID bit no level of internal/cpuinfo checks, or one
// above the level), or "" if it may. vnni allows AVX-512 VNNI.
func isaDenied(level, op, enc string, vnni bool) string {
	has := func(prefixes ...string) bool {
		return slices.ContainsFunc(prefixes, func(p string) bool { return strings.HasPrefix(op, p) })
	}
	switch {
	case has("POPCNT", "LZCNT", "MOVBE", "ADCX", "ADOX", "RDRAND", "RDSEED"):
		return "POPCNT, LZCNT, MOVBE, ADX, RDRAND and RDSEED have CPUID bits that no level checks"
	case has("CRC32", "PCMPESTR", "PCMPISTR", "PCMPGTQ"):
		return "SSE4.2, which no level checks"
	case has("AES", "VAES", "PCLMUL", "VPCLMUL", "SHA", "VSHA", "GF2P8", "VGF2P8", "VSM3", "VSM4"):
		return "AES, PCLMULQDQ, SHA, GFNI and SM3/SM4 have CPUID bits that no level checks"
	case has("VCVTPH2PS", "VCVTPS2PH") && enc == "VEX":
		return "F16C, which no level checks"
	case has("VPDPBUSD", "VPDPWSSD"):
		if enc == "VEX" {
			return "AVX-VNNI (the VEX form), which no level checks"
		}
		if !vnni {
			return "AVX-512 VNNI: this package's AVX512 kernels also run without it"
		}
	case has("VPDP"):
		return "AVX-VNNI-INT8/INT16, which no level checks"
	case has("VPMADD52"):
		return "AVX-512 IFMA, which no level checks"
	case has("VPERMB", "VPERMI2B", "VPERMT2B", "VPMULTISHIFTQB"):
		return "AVX-512 VBMI, which no level checks"
	case has("VPCOMPRESSB", "VPCOMPRESSW", "VPEXPANDB", "VPEXPANDW", "VPSHLD", "VPSHRD"):
		return "AVX-512 VBMI2, which no level checks"
	case has("VPOPCNT", "VPSHUFBITQMB"):
		return "AVX-512 BITALG or VPOPCNTDQ, which no level checks"
	case has("VCVTNE", "VDPBF16", "VBCSTNE", "VP2INTERSECT", "V4F", "VP4D", "VEXP2", "VRCP28", "VRSQRT28", "VGATHERPF", "VSCATTERPF"):
		return "an extension (BF16, VP2INTERSECT, Xeon Phi) that no level checks"
	case level == "SSE41" && has("TZCNT"):
		return "TZCNT is BMI1, part of the AVX2 level"
	}
	return ""
}

// isaUnaligned lists the legacy SSE instructions whose memory operand
// need not be 16-byte aligned (unaligned loads and stores, 1- to 8-byte
// operands); the scalar SS and SD instructions and PMOVZX/PMOVSX are
// recognized by name in needAlign.
var isaUnaligned = map[string]bool{
	"MOVDQU": true, "MOVUPS": true, "MOVUPD": true, "LDDQU": true,
	"MOVQ": true, "MOVD": true, "MOVL": true, "MOVSS": true, "MOVSD": true,
	"MOVHPS": true, "MOVLPS": true, "MOVHPD": true, "MOVLPD": true, "MOVDDUP": true,
	"INSERTPS": true, "EXTRACTPS": true,
	"PINSRB": true, "PINSRW": true, "PINSRD": true, "PINSRQ": true,
	"PEXTRB": true, "PEXTRW": true, "PEXTRD": true, "PEXTRQ": true,
	"LEAQ": true, "LEAL": true,
}

// needAlign returns the alignment a memory operand of in must have: 16
// for a legacy SSE instruction outside isaUnaligned, the vector width for
// VMOVDQA and the other aligned VEX and EVEX moves, 0 for the others.
func needAlign(in isaInsn, enc string) int64 {
	regs := vecRegRE.FindAllStringSubmatch(in.args, -1)
	if len(regs) == 0 {
		return 0
	}
	if enc == "legacy" {
		op := in.op
		if isaUnaligned[op] || strings.HasPrefix(op, "PMOVZX") || strings.HasPrefix(op, "PMOVSX") ||
			!strings.HasPrefix(op, "P") && (strings.HasSuffix(op, "SS") || strings.HasSuffix(op, "SD")) {
			return 0
		}
		return 16
	}
	switch in.op {
	case "VMOVDQA", "VMOVDQA32", "VMOVDQA64", "VMOVAPS", "VMOVAPD", "VMOVNTDQ", "VMOVNTDQA", "VMOVNTPS", "VMOVNTPD":
		w := int64(16)
		for _, r := range regs {
			w = max(w, map[string]int64{"X": 16, "Y": 32, "Z": 64}[r[1]])
		}
		return w
	}
	return 0
}

// linkerAlign is the alignment cmd/link gives a data symbol of size bytes
// that asks for none (symalign in cmd/link/internal/ld/data.go): the
// largest power of two not above size, at most 32 on amd64.
func linkerAlign(size int64) int64 {
	a := int64(32)
	for a > size && a > 1 {
		a >>= 1
	}
	return a
}

// auditKernel applies the rules of TestKernelISA to f, a kernel of level
// whose symbol is sym, and returns the violations. data holds the sizes
// of the data symbols.
func auditKernel(f *isaFunc, level string, vnni bool, sym isaSym, data map[string]int64) []string {
	var probs []string
	bad := func(in isaInsn, format string, args ...any) {
		probs = append(probs, fmt.Sprintf("+%#x %s: %s", int64(in.pc)-sym.addr, in, fmt.Sprintf(format, args...)))
	}
	ins := f.insns
	if len(ins) == 0 {
		return []string{"no instructions"}
	}

	// The disassembly covers the symbol exactly, instruction after
	// instruction (VZEROUPPER followed by anything but RET would not).
	pc := uint64(sym.addr)
	for _, in := range ins {
		if in.pc != pc {
			bad(in, "at %#x, want %#x: the disassembly is not contiguous", in.pc, pc)
			break
		}
		pc += uint64(len(in.raw))
		if in.op == "?" || strings.Contains(in.op, "(bad)") {
			bad(in, "not decoded")
		}
		if in.op == "VZEROUPPER" && len(in.raw) != 3 {
			bad(in, "go tool objdump decoded VZEROUPPER with the byte after it (% x): only VZEROUPPER; RET can be audited", in.raw)
		}
	}
	if end := uint64(sym.addr + sym.size); pc != end {
		probs = append(probs, fmt.Sprintf("disassembly ends at %#x, the symbol at %#x", pc, end))
	}

	// The frame the assembler adds for a nonzero frame size: PUSHQ BP;
	// MOVQ SP, BP; SUBQ $n, SP at the entry, ADDQ $n, SP; POPQ BP before
	// each RET.
	frame := make([]bool, len(ins))
	if len(ins) > 2 && ins[0].String() == "PUSHQ BP" && ins[1].String() == "MOVQ SP, BP" {
		frame[0], frame[1] = true, true
		if ins[2].op == "SUBQ" && strings.HasSuffix(ins[2].args, ", SP") {
			frame[2] = true
		}
	}
	for k, in := range ins {
		if in.op == "RET" && k >= 2 && ins[k-1].String() == "POPQ BP" {
			frame[k-1] = true
			if ins[k-2].op == "ADDQ" && strings.HasSuffix(ins[k-2].args, ", SP") {
				frame[k-2] = true
			}
		}
	}

	upper := slices.ContainsFunc(ins, func(in isaInsn) bool { return upperRegRE.MatchString(in.args) })
	for k, in := range ins {
		enc := isaEncoding(in.raw)
		switch {
		case level == "SSE41" && enc != "legacy":
			bad(in, "%s encoding in an SSE4.1 kernel", enc)
		case level == "AVX2" && enc == "EVEX":
			bad(in, "EVEX encoding (AVX-512) in an AVX2 kernel")
		case level != "SSE41" && enc == "legacy" && vecRegRE.MatchString(in.args):
			bad(in, "legacy SSE encoding in an %s kernel (SSE/AVX transition)", level)
		}
		if why := isaDenied(level, in.op, enc, vnni); why != "" {
			bad(in, "%s", why)
		}
		if !frame[k] && frameRegRE.MatchString(in.args) {
			bad(in, "uses BP, R14 or R15")
		}
		if strings.HasPrefix(in.op, "CALL") {
			bad(in, "a call from a kernel")
		}
		if in.op == "RET" && upper {
			j := k - 1
			for j >= 0 && frame[j] {
				j--
			}
			if j < 0 || ins[j].op != "VZEROUPPER" {
				bad(in, "RET without VZEROUPPER right before it in a kernel that uses ymm or zmm registers")
			}
		}
		need := needAlign(in, enc)
		if need == 0 {
			continue
		}
		i := slices.IndexFunc(in.relocs, func(r isaReloc) bool { return r.typ == "R_PCREL" })
		if i < 0 {
			if strings.Contains(in.args, "(") {
				bad(in, "a memory operand that must be %d-byte aligned, through a pointer whose alignment the kernel does not control", need)
			}
			continue
		}
		pcrel := in.relocs[i]
		size, ok := data[pcrel.sym]
		if !ok {
			bad(in, "RIP-relative operand %s not a data symbol of the object", pcrel.sym)
			continue
		}
		// The assembler subtracts the bytes after the relocated field
		// (an immediate) from the addend, so that the linker can compute
		// the displacement from the end of the field.
		off := pcrel.add + int64(len(in.raw)-pcrel.end)
		if off%need != 0 || linkerAlign(size) < need {
			bad(in, "%s+%d must be %d-byte aligned; the linker aligns the %d-byte symbol to %d",
				pcrel.sym, off, need, size, linkerAlign(size))
		}
	}
	return probs
}

// encodingCounts summarizes the encodings of f's instructions.
func encodingCounts(f *isaFunc) string {
	n := map[string]int{}
	for _, in := range f.insns {
		n[isaEncoding(in.raw)]++
	}
	return fmt.Sprintf("%4d instructions: %4d legacy, %4d VEX, %4d EVEX", len(f.insns), n["legacy"], n["VEX"], n["EVEX"])
}
