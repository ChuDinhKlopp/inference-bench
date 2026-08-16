#!/home/ducct/repos/vllm/.venv/bin/python
"""Render one RIVF26 run as aligned HBM/KV/scheduler SVG panels."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path


WIDTH = 1400
HEIGHT = 980
LEFT = 115
RIGHT = 125
TOP = 105
PANEL_HEIGHT = 225
PANEL_GAP = 38
PLOT_WIDTH = WIDTH - LEFT - RIGHT
COLORS = {
    "hbm": "#2563eb",
    "kv": "#d64545",
    "running": "#15803d",
    "waiting": "#d97706",
    "preemptions": "#7c3aed",
    "grid": "#dbe3ec",
    "axis": "#52606d",
    "text": "#17212b",
    "muted": "#667788",
}


def nice_upper(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    for step in (1, 2, 5, 10):
        if normalized <= step:
            return step * magnitude
    return 10 * magnitude


def tick_values(upper: float, count: int = 5) -> list[float]:
    return [upper * index / count for index in range(count + 1)]


def x_ticks(last_index: int, target: int = 7) -> list[int]:
    raw_step = max(1, last_index / target)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    step = next(value for value in (1, 2, 5, 10) if normalized <= value) * magnitude
    ticks = list(range(0, last_index + 1, int(step)))
    if ticks[-1] != last_index:
        ticks.append(last_index)
    return ticks


def points(values: list[float], y_top: float, y_upper: float) -> list[tuple[float, float]]:
    last = max(1, len(values) - 1)
    return [
        (
            LEFT + index / last * PLOT_WIDTH,
            y_top + PANEL_HEIGHT - value / y_upper * PANEL_HEIGHT,
        )
        for index, value in enumerate(values)
    ]


def path(values: list[float], y_top: float, y_upper: float) -> str:
    coords = points(values, y_top, y_upper)
    return "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in coords)


def area(values: list[float], y_top: float, y_upper: float) -> str:
    coords = points(values, y_top, y_upper)
    baseline = y_top + PANEL_HEIGHT
    body = " L ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
    return (
        f"M {coords[0][0]:.2f},{baseline:.2f} L {body} "
        f"L {coords[-1][0]:.2f},{baseline:.2f} Z"
    )


def text_node(x: float, y: float, value: str, **attrs: object) -> str:
    rendered = " ".join(f'{key.replace("_", "-")}="{item}"' for key, item in attrs.items())
    return f'<text x="{x}" y="{y}" {rendered}>{html.escape(value)}</text>'


def panel_axes(
    y_top: float,
    y_upper: float,
    ylabel: str,
    xticks: list[int],
    last_index: int,
    show_x_labels: bool,
) -> list[str]:
    items = [
        f'<rect x="{LEFT}" y="{y_top}" width="{PLOT_WIDTH}" height="{PANEL_HEIGHT}" '
        'fill="#ffffff" stroke="#aebcca" stroke-width="1"/>'
    ]
    for value in tick_values(y_upper):
        y = y_top + PANEL_HEIGHT - value / y_upper * PANEL_HEIGHT
        items.append(
            f'<line x1="{LEFT}" x2="{LEFT + PLOT_WIDTH}" y1="{y:.2f}" y2="{y:.2f}" '
            f'stroke="{COLORS["grid"]}" stroke-width="1"/>'
        )
        label = f"{value:.0f}" if y_upper >= 10 else f"{value:.1f}"
        items.append(
            text_node(
                LEFT - 12,
                y + 5,
                label,
                text_anchor="end",
                fill=COLORS["muted"],
                font_size="14",
            )
        )
    for value in xticks:
        x = LEFT + value / max(1, last_index) * PLOT_WIDTH
        items.append(
            f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{y_top}" y2="{y_top + PANEL_HEIGHT}" '
            f'stroke="{COLORS["grid"]}" stroke-width="1"/>'
        )
        if show_x_labels:
            items.append(
                text_node(
                    x,
                    y_top + PANEL_HEIGHT + 25,
                    str(value),
                    text_anchor="middle",
                    fill=COLORS["muted"],
                    font_size="14",
                )
            )
    cy = y_top + PANEL_HEIGHT / 2
    items.append(
        text_node(
            28,
            cy,
            ylabel,
            text_anchor="middle",
            fill=COLORS["text"],
            font_size="15",
            font_weight="600",
            transform=f"rotate(-90 28 {cy})",
        )
    )
    return items


def locate_run(data: dict, run_id: str | None) -> tuple[str, str, dict]:
    matches = []
    for series_name, series in data["TS"].items():
        for candidate, values in series.get("runs", {}).items():
            if run_id is None or candidate == run_id:
                matches.append((series_name, candidate, values))
    if not matches:
        raise ValueError(f"run_id not found in plot data: {run_id}")
    if len(matches) != 1:
        raise ValueError("plot data contains multiple runs; pass --run-id")
    return matches[0]


def render(data: dict, run_id: str | None) -> str:
    series_name, selected_run_id, run = locate_run(data, run_id)
    required = ("hbm", "kv", "run", "wait", "pre")
    lengths = {name: len(run[name]) for name in required}
    if any(length == 0 for length in lengths.values()) or len(set(lengths.values())) != 1:
        raise ValueError(f"required aligned series are missing or unequal: {lengths}")

    hbm = [float(value) for value in run["hbm"]]
    kv = [float(value) * 100 for value in run["kv"]]
    running = [float(value) for value in run["run"]]
    waiting = [float(value) for value in run["wait"]]
    preemptions = [float(value) for value in run["pre"]]
    count = len(hbm)
    last_index = count - 1
    bin_seconds = float(data["run"]["bin_seconds"])
    xticks = x_ticks(last_index)
    scheduler_upper = nice_upper(max(running + waiting))
    preemption_upper = nice_upper(max(preemptions))
    y_positions = [TOP + index * (PANEL_HEIGHT + PANEL_GAP) for index in range(3)]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        f'aria-label="Stacked inference timeline for {html.escape(selected_run_id)}">',
        '<rect width="100%" height="100%" fill="#f7f9fc"/>',
        text_node(LEFT, 40, "RIVF26 Part 1 — aligned inference timeline", fill=COLORS["text"], font_size="25", font_weight="700"),
        text_node(LEFT, 68, f"{selected_run_id} · {series_name} · {count} samples × {bin_seconds:g} s", fill=COLORS["muted"], font_size="15"),
    ]

    svg.extend(panel_axes(y_positions[0], 100, "HBM BW utilization (%)", xticks, last_index, False))
    svg.append(f'<path d="{area(hbm, y_positions[0], 100)}" fill="{COLORS["hbm"]}" opacity="0.13"/>')
    svg.append(f'<path d="{path(hbm, y_positions[0], 100)}" fill="none" stroke="{COLORS["hbm"]}" stroke-width="2"/>')
    svg.append(text_node(LEFT + PLOT_WIDTH - 8, y_positions[0] + 23, "normalized aggregate HBM bandwidth", text_anchor="end", fill=COLORS["hbm"], font_size="14", font_weight="600"))

    svg.extend(panel_axes(y_positions[1], 100, "KV-cache utilization (%)", xticks, last_index, False))
    svg.append(f'<path d="{area(kv, y_positions[1], 100)}" fill="{COLORS["kv"]}" opacity="0.12"/>')
    svg.append(f'<path d="{path(kv, y_positions[1], 100)}" fill="none" stroke="{COLORS["kv"]}" stroke-width="2"/>')
    svg.append(text_node(LEFT + PLOT_WIDTH - 8, y_positions[1] + 23, "KV cache used / capacity", text_anchor="end", fill=COLORS["kv"], font_size="14", font_weight="600"))

    svg.extend(panel_axes(y_positions[2], scheduler_upper, "scheduled requests", xticks, last_index, True))
    svg.append(f'<path d="{area(running, y_positions[2], scheduler_upper)}" fill="{COLORS["running"]}" opacity="0.10"/>')
    svg.append(f'<path d="{path(running, y_positions[2], scheduler_upper)}" fill="none" stroke="{COLORS["running"]}" stroke-width="2.2"/>')
    svg.append(f'<path d="{path(waiting, y_positions[2], scheduler_upper)}" fill="none" stroke="{COLORS["waiting"]}" stroke-width="2" stroke-dasharray="7 5"/>')

    pre_points = points(preemptions, y_positions[2], preemption_upper)
    pre_path = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in pre_points)
    svg.append(f'<path d="{pre_path}" fill="none" stroke="{COLORS["preemptions"]}" stroke-width="1.8" stroke-dasharray="3 5"/>')
    svg.append(text_node(LEFT + PLOT_WIDTH + 16, y_positions[2] + 5, f"preemptions: {int(max(preemptions))}", fill=COLORS["preemptions"], font_size="13"))

    legend_y = y_positions[2] + 23
    legend_x = LEFT + PLOT_WIDTH - 360
    for offset, label, color, dash in (
        (0, "running", COLORS["running"], ""),
        (120, "waiting", COLORS["waiting"], "7 5"),
        (235, "preemptions", COLORS["preemptions"], "3 5"),
    ):
        x = legend_x + offset
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(f'<line x1="{x}" x2="{x + 34}" y1="{legend_y}" y2="{legend_y}" stroke="{color}" stroke-width="2.2"{dash_attr}/>')
        svg.append(text_node(x + 40, legend_y + 5, label, fill=color, font_size="13", font_weight="600"))

    svg.append(
        text_node(
            LEFT + PLOT_WIDTH / 2,
            y_positions[2] + PANEL_HEIGHT + 58,
            f"sampled timestep ({bin_seconds:g} seconds per sample)",
            text_anchor="middle",
            fill=COLORS["text"],
            font_size="16",
            font_weight="600",
        )
    )
    summary = (
        f"peak HBM {max(hbm):.1f}% · peak KV {max(kv):.1f}% · "
        f"peak running {max(running):.1f} · peak waiting {max(waiting):.1f} · "
        f"preemptions {int(max(preemptions))}"
    )
    svg.append(text_node(LEFT, HEIGHT - 25, summary, fill=COLORS["muted"], font_size="14"))
    svg.append("</svg>")
    return "\n".join(svg)


def render_png(data: dict, run_id: str | None, output: Path) -> None:
    from PIL import Image, ImageColor, ImageDraw, ImageFont

    series_name, selected_run_id, run = locate_run(data, run_id)
    hbm = [float(value) for value in run["hbm"]]
    kv = [float(value) * 100 for value in run["kv"]]
    running = [float(value) for value in run["run"]]
    waiting = [float(value) for value in run["wait"]]
    preemptions = [float(value) for value in run["pre"]]
    arrays = (hbm, kv, running, waiting, preemptions)
    if any(not values for values in arrays) or len({len(values) for values in arrays}) != 1:
        raise ValueError("PNG inputs must be non-empty aligned series")

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_path, 14)
    small = ImageFont.truetype(font_path, 13)
    label_font = ImageFont.truetype(bold_path, 15)
    title_font = ImageFont.truetype(bold_path, 25)
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f7f9fc")
    draw = ImageDraw.Draw(image)

    def rgb(name: str) -> tuple[int, int, int]:
        return ImageColor.getrgb(COLORS[name])

    def faded(name: str, opacity: float = 0.12) -> tuple[int, int, int]:
        color = rgb(name)
        return tuple(round(255 * (1 - opacity) + channel * opacity) for channel in color)

    def dashed_line(coords: list[tuple[float, float]], color: str, width: int, dash: int, gap: int) -> None:
        for start, end in zip(coords, coords[1:]):
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            if length == 0:
                continue
            position = 0.0
            while position < length:
                stop = min(position + dash, length)
                first = (start[0] + dx * position / length, start[1] + dy * position / length)
                second = (start[0] + dx * stop / length, start[1] + dy * stop / length)
                draw.line((first, second), fill=rgb(color), width=width)
                position += dash + gap

    count = len(hbm)
    last_index = count - 1
    bin_seconds = float(data["run"]["bin_seconds"])
    xticks = x_ticks(last_index)
    y_positions = [TOP + index * (PANEL_HEIGHT + PANEL_GAP) for index in range(3)]
    scheduler_upper = nice_upper(max(running + waiting))

    draw.text((LEFT, 30), "RIVF26 Part 1 — aligned inference timeline", fill=rgb("text"), font=title_font)
    draw.text(
        (LEFT, 65),
        f"{selected_run_id} · {series_name} · {count} samples × {bin_seconds:g} s",
        fill=rgb("muted"),
        font=font,
    )

    def axes(y_top: float, upper: float, ylabel: str, show_x: bool) -> None:
        draw.rectangle((LEFT, y_top, LEFT + PLOT_WIDTH, y_top + PANEL_HEIGHT), fill="white", outline="#aebcca")
        for value in tick_values(upper):
            y = y_top + PANEL_HEIGHT - value / upper * PANEL_HEIGHT
            draw.line((LEFT, y, LEFT + PLOT_WIDTH, y), fill=rgb("grid"), width=1)
            label = f"{value:.0f}" if upper >= 10 else f"{value:.1f}"
            box = draw.textbbox((0, 0), label, font=font)
            draw.text((LEFT - 12 - (box[2] - box[0]), y - 8), label, fill=rgb("muted"), font=font)
        for value in xticks:
            x = LEFT + value / max(1, last_index) * PLOT_WIDTH
            draw.line((x, y_top, x, y_top + PANEL_HEIGHT), fill=rgb("grid"), width=1)
            if show_x:
                label = str(value)
                box = draw.textbbox((0, 0), label, font=font)
                draw.text((x - (box[2] - box[0]) / 2, y_top + PANEL_HEIGHT + 7), label, fill=rgb("muted"), font=font)
        label_image = Image.new("RGBA", (PANEL_HEIGHT, 28), (0, 0, 0, 0))
        label_draw = ImageDraw.Draw(label_image)
        label_draw.text((PANEL_HEIGHT / 2, 3), ylabel, anchor="ma", fill=rgb("text"), font=label_font)
        rotated = label_image.rotate(90, expand=True)
        image.paste(rotated, (8, round(y_top + (PANEL_HEIGHT - rotated.height) / 2)), rotated)

    axes(y_positions[0], 100, "HBM BW utilization (%)", False)
    hbm_points = points(hbm, y_positions[0], 100)
    draw.polygon([(hbm_points[0][0], y_positions[0] + PANEL_HEIGHT), *hbm_points, (hbm_points[-1][0], y_positions[0] + PANEL_HEIGHT)], fill=faded("hbm"))
    draw.line(hbm_points, fill=rgb("hbm"), width=2, joint="curve")
    draw.text((LEFT + PLOT_WIDTH - 285, y_positions[0] + 10), "normalized aggregate HBM bandwidth", fill=rgb("hbm"), font=small)

    axes(y_positions[1], 100, "KV-cache utilization (%)", False)
    kv_points = points(kv, y_positions[1], 100)
    draw.polygon([(kv_points[0][0], y_positions[1] + PANEL_HEIGHT), *kv_points, (kv_points[-1][0], y_positions[1] + PANEL_HEIGHT)], fill=faded("kv"))
    draw.line(kv_points, fill=rgb("kv"), width=2, joint="curve")
    draw.text((LEFT + PLOT_WIDTH - 200, y_positions[1] + 10), "KV cache used / capacity", fill=rgb("kv"), font=small)

    axes(y_positions[2], scheduler_upper, "scheduled requests", True)
    running_points = points(running, y_positions[2], scheduler_upper)
    waiting_points = points(waiting, y_positions[2], scheduler_upper)
    preemption_upper = nice_upper(max(preemptions))
    preemption_points = points(preemptions, y_positions[2], preemption_upper)
    draw.polygon([(running_points[0][0], y_positions[2] + PANEL_HEIGHT), *running_points, (running_points[-1][0], y_positions[2] + PANEL_HEIGHT)], fill=faded("running"))
    draw.line(running_points, fill=rgb("running"), width=2, joint="curve")
    dashed_line(waiting_points, "waiting", 2, 8, 5)
    dashed_line(preemption_points, "preemptions", 2, 3, 5)
    draw.text((LEFT + PLOT_WIDTH - 335, y_positions[2] + 10), "running", fill=rgb("running"), font=small)
    draw.text((LEFT + PLOT_WIDTH - 245, y_positions[2] + 10), "waiting", fill=rgb("waiting"), font=small)
    draw.text((LEFT + PLOT_WIDTH - 155, y_positions[2] + 10), f"preemptions ({int(max(preemptions))})", fill=rgb("preemptions"), font=small)

    xlabel = f"sampled timestep ({bin_seconds:g} seconds per sample)"
    xlabel_box = draw.textbbox((0, 0), xlabel, font=label_font)
    draw.text((LEFT + (PLOT_WIDTH - (xlabel_box[2] - xlabel_box[0])) / 2, y_positions[2] + PANEL_HEIGHT + 43), xlabel, fill=rgb("text"), font=label_font)
    summary = (
        f"peak HBM {max(hbm):.1f}% · peak KV {max(kv):.1f}% · peak running {max(running):.1f} · "
        f"peak waiting {max(waiting):.1f} · preemptions {int(max(preemptions))}"
    )
    draw.text((LEFT, HEIGHT - 28), summary, fill=rgb("muted"), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plot_data", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--output-svg", type=Path, required=True)
    parser.add_argument("--output-html", type=Path)
    parser.add_argument("--output-png", type=Path)
    args = parser.parse_args()

    data = json.loads(args.plot_data.read_text())
    svg = render(data, args.run_id)
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.write_text(svg + "\n")
    if args.output_html:
        args.output_html.parent.mkdir(parents=True, exist_ok=True)
        args.output_html.write_text(
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>RIVF26 stacked inference timeline</title>"
            "<style>body{margin:0;background:#f7f9fc;font-family:Inter,Arial,sans-serif}"
            "main{max-width:1400px;margin:auto}svg{display:block;width:100%;height:auto}</style>"
            f"</head><body><main>{svg}</main></body></html>\n"
        )
    if args.output_png:
        render_png(data, args.run_id, args.output_png)
    print(f"wrote {args.output_svg}")
    if args.output_html:
        print(f"wrote {args.output_html}")
    if args.output_png:
        print(f"wrote {args.output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
