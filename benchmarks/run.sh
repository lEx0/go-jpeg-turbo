#!/usr/bin/env bash
# Compares this repository with the standard library's image/jpeg on the
# same workloads (suite/suite.go); see the Benchmarks section of
# ../README.md.
#
#   benchmarks/run.sh                                # everything (about 20 minutes)
#   ROUNDS=3 RSS_ROUNDS=1 SAT=0 benchmarks/run.sh    # a quick look (about 5 minutes)
#   BUILDS="std turbo-sse41 turbo-avx2 turbo-avx512" benchmarks/run.sh    # amd64 levels (25 minutes)
#   PIN=2 benchmarks/run.sh                          # Linux: single calls on CPU 2
#
# Environment:
#   BUILDS      builds to compare, in table order; the first one is the
#               reference column of the benchstat tables (default
#               "std turbo-purego turbo"):
#               std = the standard library's image/jpeg;
#               turbo-purego = this repository built with -tags purego (the
#               portable Go code that other architectures run);
#               turbo = this repository as built by default (NEON on arm64,
#               on amd64 the best of SSE4.1, AVX2 and AVX-512 the CPU has);
#               turbo-LEVEL = the default build with JPEG_TURBO_LEVEL=LEVEL
#               (go, sse41, avx2 or avx512, up to what the CPU has): the
#               kernels of that level on amd64 (arm64 accepts only go and
#               still runs NEON)
#   ROUNDS      benchmark rounds (default 10); the builds alternate within
#               every round, in a rotating order, so that a change of the
#               machine's state affects all of them alike
#   BENCHTIME   -test.benchtime of each single-call benchmark (default 1s)
#   SAT         1 (default) also runs the saturation benchmarks (GOMAXPROCS
#               calls at a time), 0 skips them
#   SATTIME     -test.benchtime of each saturation benchmark (default 2s)
#   RSS_ROUNDS  peak-memory rounds (default 5; 0 skips them)
#   PIN         Linux: a CPU list for taskset -c that the single-call
#               benchmarks and the peak-memory runs are confined to (PIN=2:
#               one CPU, so GOMAXPROCS is 1 and the garbage collector shares
#               it with the call); the saturation benchmarks keep every CPU
#               that run.sh may use (default: no pinning)
#   OUT         output directory (default benchmarks/out; a relative path is
#               taken from the current directory)
#
# Before measuring, every build of this repository must give the same
# results as image/jpeg on all the workloads (TestSameAsStd), and the 12 MP
# inputs are generated into benchmarks/data/ unless they are there
# (suite/data.go).
#
# Results: $OUT/benchstat.txt (medians, and the differences from the first
# build), $OUT/benchstat-saturation.txt, $OUT/rss.txt (peak resident memory,
# medians), the raw outputs in $OUT/raw/, the machine, the builds' kernels
# and the inputs' SHA-256 sums in $OUT/env.txt.
set -euo pipefail

case ${OUT:-} in
"" | /*) ;;
*) OUT=$PWD/$OUT ;;
esac
cd "$(dirname "$0")"
BUILDS=${BUILDS:-"std turbo-purego turbo"}
ROUNDS=${ROUNDS:-10}
BENCHTIME=${BENCHTIME:-1s}
SAT=${SAT:-1}
SATTIME=${SATTIME:-2s}
RSS_ROUNDS=${RSS_ROUNDS:-5}
PIN=${PIN:-}
OUT=${OUT:-out}
mkdir -p "$OUT/bin" "$OUT/raw"
OUT=$(cd "$OUT" && pwd)
BIN=$OUT/bin
RAW=$OUT/raw
rm -f "$OUT"/benchstat*.txt "$OUT/rss.txt"

if [ "$ROUNDS" -gt 0 ]; then
	BENCHSTAT=${BENCHSTAT:-$(command -v benchstat || echo "$(go env GOPATH)/bin/benchstat")}
	if [ ! -x "$BENCHSTAT" ]; then
		echo "benchstat not found: go install golang.org/x/perf/cmd/benchstat@latest" >&2
		exit 1
	fi
fi
if [ -n "$PIN" ] && ! command -v taskset >/dev/null; then
	echo "PIN needs taskset (Linux, util-linux)" >&2
	exit 1
fi

load() { uptime | sed 's/.*load averages*: *//'; }

# level prints the JPEG_TURBO_LEVEL of build $1 (empty: the CPU's best).
level() {
	case $1 in
	std | turbo | turbo-purego) ;;
	turbo-*) echo "${1#turbo-}" ;;
	esac
}

# run runs build $1's command "$2 ..." with its JPEG_TURBO_LEVEL.
run() {
	local b=$1
	shift
	JPEG_TURBO_LEVEL=$(level "$b") "$@"
}

# pinned is run on the CPUs of $PIN (on every CPU without it).
pinned() {
	if [ -n "$PIN" ]; then
		local b=$1
		shift
		run "$b" taskset -c "$PIN" "$@"
	else
		run "$@"
	fi
}

{
	echo "date: $(date)"
	echo "go: $(go version)"
	echo "os: $(uname -srm)"
	case "$(uname -s)" in
	Darwin)
		echo "cpu: $(sysctl -n machdep.cpu.brand_string), $(sysctl -n hw.ncpu) cores, $(($(sysctl -n hw.memsize) / 1073741824)) GiB"
		echo "macos: $(sw_vers -productVersion)"
		pmset -g 2>/dev/null | awk '$1 == "lowpowermode" || $1 == "powermode" { print "power: " $1 " " $2 }' || true
		;;
	Linux)
		# arm64 kernels have no "model name" in /proc/cpuinfo; lscpu decodes the part numbers.
		model=$(grep -m1 '^model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //' || true)
		[ -n "$model" ] || model=$(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1 || true)
		if [ "$model" = - ]; then model=; fi
		echo "cpu: ${model:-unknown}, $(nproc --all) cpus ($(nproc) available to run.sh), $(awk '/^MemTotal/ { printf "%d GiB", $2 / 1048576 }' /proc/meminfo)"
		if [ -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
			echo "governor: $(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c | awk '{ printf "%s%s (%d cpus)", sep, $2, $1; sep = ", " }')"
		fi
		if [ -r /sys/devices/system/cpu/cpufreq/boost ]; then
			echo "boost: $(cat /sys/devices/system/cpu/cpufreq/boost)"
		fi
		;;
	esac
	# GOARCH can differ from the machine's (amd64 binaries under Rosetta 2).
	case "$(go env GOARCH)" in
	amd64) echo "GOARCH: amd64, GOAMD64: $(go env GOAMD64)" ;;
	arm64) echo "GOARCH: arm64, GOARM64: $(go env GOARM64)" ;;
	*) echo "GOARCH: $(go env GOARCH)" ;;
	esac
	echo "GOMAXPROCS: ${GOMAXPROCS:-default}; GOGC: ${GOGC:-default}; GOMEMLIMIT: ${GOMEMLIMIT:-default}"
	if rev=$(git -C .. rev-parse --short HEAD 2>/dev/null); then
		if [ -n "$(git -C .. status --porcelain --untracked-files=no 2>/dev/null)" ]; then
			rev="$rev (modified)"
		fi
		echo "repository: $rev"
	fi
	echo "builds: $BUILDS; rounds: $ROUNDS; benchtime: $BENCHTIME; saturation: $SAT ($SATTIME); rss rounds: $RSS_ROUNDS; pin: ${PIN:-none}"
	echo "load at start: $(load)"
} >"$OUT/env.txt"

echo "building..."
for b in $BUILDS; do
	case $b in
	std) dir=./std tags= ;;
	turbo-purego) dir=./turbo tags=purego ;;
	turbo | turbo-*) dir=./turbo tags= ;;
	*)
		echo "unknown build $b" >&2
		exit 1
		;;
	esac
	if ! go test -c -tags "$tags" -o "$BIN/$b.test" "$dir" 2>"$RAW/build-$b.log" ||
		! go build -tags "$tags" -o "$BIN/rss-$b" "$dir/rss" 2>>"$RAW/build-$b.log"; then
		cat "$RAW/build-$b.log" >&2
		exit 1
	fi
	info=$(run "$b" "$BIN/rss-$b" -mode info) # fails for a level the CPU lacks
	echo "build $b: $info" >>"$OUT/env.txt"
done

echo "generating the 12 MP inputs..."
set -- $BUILDS
"$BIN/rss-$1" -mode gen | sed 's/^/data: /' >>"$OUT/env.txt"
cat "$OUT/env.txt"

for b in $BUILDS; do
	[ "$b" = std ] && continue
	echo "checking $b against image/jpeg..."
	if ! run "$b" "$BIN/$b.test" -test.run '^TestSameAsStd$' -test.v >"$RAW/check-$b.txt" 2>&1; then
		cat "$RAW/check-$b.txt" >&2
		exit 1
	fi
done

if [ "$ROUNDS" -gt 0 ]; then
	set -- $BUILDS
	n=$#
	for b in $BUILDS; do : >"$RAW/$b.txt"; : >"$RAW/sat-$b.txt"; done
	for r in $(seq 1 "$ROUNDS"); do
		for k in $(seq 0 $((n - 1))); do
			i=$(((k + r) % n + 1))
			b=${!i}
			echo "round $r/$ROUNDS $b (load $(load))"
			pinned "$b" "$BIN/$b.test" -test.run '^$' -test.bench '^Benchmark(Decode|DecodeConfig|Encode)$' \
				-test.benchtime "$BENCHTIME" -test.benchmem -test.count 1 -test.timeout 0 >>"$RAW/$b.txt"
			if [ "$SAT" = 1 ]; then
				run "$b" "$BIN/$b.test" -test.run '^$' -test.bench '^BenchmarkSat' \
					-test.benchtime "$SATTIME" -test.benchmem -test.count 1 -test.timeout 0 >>"$RAW/sat-$b.txt"
			fi
		done
	done
	files=()
	satfiles=()
	for b in $BUILDS; do
		files+=("$b.txt")
		satfiles+=("sat-$b.txt")
	done
	# The package paths differ between the builds' outputs.
	(cd "$RAW" && "$BENCHSTAT" -ignore pkg "${files[@]}") >"$OUT/benchstat.txt"
	if [ "$SAT" = 1 ]; then
		(cd "$RAW" && "$BENCHSTAT" -ignore pkg "${satfiles[@]}") >"$OUT/benchstat-saturation.txt"
	fi
fi

if [ "$RSS_ROUNDS" -gt 0 ]; then
	WORKLOADS="Decode/img=420q90_4000x3000 Encode/img=RGBA_4000x3000/q=90"
	: >"$RAW/rss.log"
	rss() {
		local b=$1 w=$2 line
		shift 2
		line=$(pinned "$b" "$@" | tail -1)
		printf '%s\t%s\t%s\n' "$b" "$w" "$line" >>"$RAW/rss.log"
	}
	for r in $(seq 1 "$RSS_ROUNDS"); do
		for b in $BUILDS; do
			echo "rss round $r/$RSS_ROUNDS $b (load $(load))"
			rss "$b" Decode/img=420q90_4000x3000 "$BIN/rss-$b" -mode dec12
			rss "$b" Encode/img=RGBA_4000x3000/q=90 "$BIN/rss-$b" -mode enc12 -q 90
		done
	done
	# Medians (and ranges) per build and workload: the peak resident memory
	# of the process, the peak before the call (the Go runtime and the
	# input) and the wall time of the call.
	tab=$(printf '\t')
	awk -F'\t' '{
		n = split($3, kv, " ")
		for (i = 1; i <= n; i++) {
			split(kv[i], p, "=")
			print $1 "\t" $2 "\t" p[1] "\t" p[2]
		}
	}' "$RAW/rss.log" | sort -t "$tab" -k1,1 -k2,2 -k3,3 -k4,4n | awk -F'\t' -v builds="$BUILDS" -v works="$WORKLOADS" '
	function flush() {
		if (key == "") return
		med[key] = v[int((c + 1) / 2)]; lo[key] = v[1]; hi[key] = v[c]; cnt[key] = c
	}
	{
		k = $1 SUBSEP $2 SUBSEP $3
		if (k != key) { flush(); key = k; c = 0 }
		v[++c] = $4
	}
	END {
		flush()
		nb = split(builds, B, " "); nw = split(works, W, " ")
		for (j = 1; j <= nw; j++) for (i = 1; i <= nb; i++) {
			k = B[i] SUBSEP W[j]
			if (!((k SUBSEP "peak_mb") in med)) continue
			printf "%-14s %-32s peak %7.1f MB (%.1f-%.1f)  before the call %6.1f MB  wall %7.1f ms  (%d runs)\n",
				B[i], W[j], med[k SUBSEP "peak_mb"], lo[k SUBSEP "peak_mb"], hi[k SUBSEP "peak_mb"],
				med[k SUBSEP "base_mb"], med[k SUBSEP "wall_ms"], cnt[k SUBSEP "peak_mb"]
		}
	}' >"$OUT/rss.txt"
fi

echo "load at end: $(load)" >>"$OUT/env.txt"
echo
[ -f "$OUT/benchstat.txt" ] && cat "$OUT/benchstat.txt"
[ -f "$OUT/benchstat-saturation.txt" ] && cat "$OUT/benchstat-saturation.txt"
[ -f "$OUT/rss.txt" ] && cat "$OUT/rss.txt"
echo "results in $OUT"
