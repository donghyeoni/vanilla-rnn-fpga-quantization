"""
합성 리포트 수집 (HLS synthesis reports -> CSV).

``run_hls.tcl``의 sweep 모드가 만든 solution들의 ``csynth.xml``을 읽어 면적 대
지연시간 표를 만든다. UNROLL_H를 바꿀 때 무엇이 늘고 무엇이 줄어드는지가
이 프로젝트의 하드웨어 쪽 성과 지표다.

주의: 여기 나오는 LUT/DSP/BRAM은 **HLS의 추정치**다. Vivado 구현(place & route)을
거치면 특히 LUT이 상당히 달라진다. 최종 수치로 쓸 것은 post-route 리포트다.

사용 예:
    python hls/collect_reports.py
    python hls/collect_reports.py --proj hls/hls_proj --out results/hls/sweep.csv
"""

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HLS_DIR = Path(__file__).resolve().parent


def text(node, path, default=""):
    found = node.find(path)
    return found.text.strip() if found is not None and found.text else default


def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_csynth(xml_path):
    """csynth.xml -> 한 행짜리 dict."""
    root = ET.parse(xml_path).getroot()

    area = root.find("AreaEstimates/Resources")
    avail = root.find("AreaEstimates/AvailableResources")
    timing = root.find("PerformanceEstimates/SummaryOfTimingAnalysis")
    latency = root.find("PerformanceEstimates/SummaryOfOverallLatency")

    row = {
        "solution":     xml_path.parent.parent.parent.name,
        "part":         text(root, "UserAssignments/Part"),
        "target_ns":    text(root, "UserAssignments/TargetClockPeriod"),
        "estimated_ns": text(timing, "EstimatedClockPeriod") if timing is not None else "",
        "latency_min":  text(latency, "Best-caseLatency") if latency is not None else "",
        "latency_max":  text(latency, "Worst-caseLatency") if latency is not None else "",
        "interval_max": text(latency, "Interval-max") if latency is not None else "",
    }
    for res in ("LUT", "FF", "DSP", "BRAM_18K", "URAM"):
        row[res] = text(area, res) if area is not None else ""
        row[f"{res}_avail"] = text(avail, res) if avail is not None else ""

    # solution 이름(u16)에서 UNROLL 값을 뽑는다
    m = re.fullmatch(r"u(\d+)", row["solution"])
    row["unroll"] = int(m.group(1)) if m else None

    # 사용률(%)과 Fmax
    for res in ("LUT", "FF", "DSP", "BRAM_18K"):
        used, total = as_int(row[res]), as_int(row[f"{res}_avail"])
        row[f"{res}_pct"] = round(100.0 * used / total, 2) if used and total else ""

    est = row["estimated_ns"]
    try:
        row["fmax_mhz"] = round(1000.0 / float(est), 1) if float(est) > 0 else ""
    except (TypeError, ValueError):
        row["fmax_mhz"] = ""

    return row


def main():
    ap = argparse.ArgumentParser(description="HLS sweep 리포트를 CSV로 모은다")
    ap.add_argument("--proj", default=str(HLS_DIR / "hls_proj"),
                    help="run_hls.tcl이 만든 프로젝트 디렉터리")
    ap.add_argument("--out", default=str(HLS_DIR / "build" / "sweep.csv"))
    args = ap.parse_args()

    proj = Path(args.proj)
    if not proj.is_dir():
        print(f"[FAIL] 프로젝트를 찾을 수 없다: {proj}")
        print("  vitis_hls -f hls/run_hls.tcl -tclargs sweep")
        return 1

    xmls = sorted(proj.glob("*/syn/report/csynth.xml"))
    if not xmls:
        print(f"[FAIL] csynth.xml이 없다: {proj}/*/syn/report/")
        print("  csynth_design이 실행된 solution이 있어야 한다.")
        return 1

    rows = [parse_csynth(x) for x in xmls]
    rows.sort(key=lambda r: (r["unroll"] is None, r["unroll"]))

    cols = ["solution", "unroll", "part", "target_ns", "estimated_ns", "fmax_mhz",
            "latency_min", "latency_max", "interval_max",
            "LUT", "LUT_pct", "FF", "FF_pct",
            "DSP", "DSP_pct", "BRAM_18K", "BRAM_18K_pct"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # 콘솔 표 (README에 붙일 형태)
    head = f"{'UNROLL':>7}{'cycles':>10}{'Fmax(MHz)':>11}{'LUT':>9}{'FF':>9}{'DSP':>7}{'BRAM':>7}"
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{str(r['unroll'] or r['solution']):>7}"
              f"{str(r['latency_max']):>10}{str(r['fmax_mhz']):>11}"
              f"{str(r['LUT']):>9}{str(r['FF']):>9}"
              f"{str(r['DSP']):>7}{str(r['BRAM_18K']):>7}")

    print(f"\n{len(rows)} solutions -> {out}")
    print("주의: 위 수치는 HLS 추정치다. 최종 수치는 Vivado post-route 리포트를 쓸 것.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
