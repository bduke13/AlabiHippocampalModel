"""
Unified-compatible HTML report entrypoint (non-fast alias).

This module delegates to hmap_html_report_fast so both scripts produce
consistent outputs for unified and per-scale architectures.
"""

from hmap_html_report_fast import (
    discover_available_scales,
    generate_multi_scale_reports,
)


if __name__ == "__main__":
    print("Starting unified-compatible place cell HTML report generation...")

    reports = generate_multi_scale_reports(
        scales=discover_available_scales() or [0, 1, 2],
        activation_threshold=0.2,
        open_browser=True,
        gridsize=50,
        dpi=300,
        n_workers=1,  # keep this entrypoint deterministic/non-parallel by default
    )

    print("\nAll reports generated!")
    for i, p in enumerate(reports, 1):
        print(f"Report {i}: {p}")
