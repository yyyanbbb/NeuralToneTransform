from pathlib import Path


COLORS = {
    "ink": "#263238",
    "muted": "#607D8B",
    "line": "#90A4AE",
    "paper": "#FFFFFF",
    "panel": "#F6F8FA",
    "blue": "#0F4D92",
    "blue_fill": "#E7F0FA",
    "teal": "#2C7A7B",
    "teal_fill": "#E5F4F4",
    "gold": "#A76F16",
    "gold_fill": "#F8EBCB",
    "violet": "#5B4DA3",
    "violet_fill": "#ECE9F8",
    "rose": "#A6403A",
    "rose_fill": "#F7E5E2",
}


def esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def pdf_text_escape(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))


class SVG:
    def __init__(self, path, w, h):
        self.path = Path(path)
        self.w = w
        self.h = h
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
            """<style>
            text{font-family:Arial,Helvetica,sans-serif;fill:#263238}
            .title{font-size:20px;font-weight:700}
            .subtitle{font-size:12px;fill:#607D8B}
            .label{font-size:13px;font-weight:700}
            .body{font-size:12px;fill:#455A64}
            .panel{font-size:15px;font-weight:700}
            </style>""",
        ]

    def rect(self, x, y, w, h, fill, stroke=None, sw=1.0, rx=0):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke or fill}" stroke-width="{sw}"/>'
        )

    def line(self, x1, y1, x2, y2, color=COLORS["line"], sw=1.4):
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" stroke-linecap="round"/>'
        )

    def circle(self, x, y, r, fill, stroke=None, sw=1.0):
        self.parts.append(
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke or fill}" stroke-width="{sw}"/>'
        )

    def arrow(self, x1, y1, x2, y2, color=COLORS["line"]):
        self.line(x1, y1, x2, y2, color, 1.6)
        if abs(x2 - x1) >= abs(y2 - y1):
            s = 1 if x2 >= x1 else -1
            pts = [(x2, y2), (x2 - s * 9, y2 - 5), (x2 - s * 9, y2 + 5)]
        else:
            s = 1 if y2 >= y1 else -1
            pts = [(x2, y2), (x2 - 5, y2 - s * 9), (x2 + 5, y2 - s * 9)]
        self.poly(pts, color, color)

    def poly(self, pts, fill, stroke=None, sw=1):
        points = " ".join(f"{x},{y}" for x, y in pts)
        self.parts.append(f'<polygon points="{points}" fill="{fill}" stroke="{stroke or fill}" stroke-width="{sw}"/>')

    def text(self, x, y, text, cls="body", anchor="start"):
        self.parts.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{esc(text)}</text>')

    def save(self):
        self.parts.append("</svg>")
        self.path.write_text("\n".join(self.parts), encoding="utf-8")


class PDF:
    def __init__(self, path, w, h):
        self.path = Path(path)
        self.w = w
        self.h = h
        self.ops = []

    def _fill(self, color):
        self.ops.append("%.3f %.3f %.3f rg" % rgb(color))

    def _stroke(self, color):
        self.ops.append("%.3f %.3f %.3f RG" % rgb(color))

    def rect(self, x, y, w, h, fill, stroke=None, sw=0.6):
        y = self.h - y - h
        self._fill(fill)
        self._stroke(stroke or fill)
        self.ops.append(f"{sw:.2f} w {x:.2f} {y:.2f} {w:.2f} {h:.2f} re B")

    def line(self, x1, y1, x2, y2, color=COLORS["line"], sw=0.8):
        y1, y2 = self.h - y1, self.h - y2
        self._stroke(color)
        self.ops.append(f"{sw:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def circle(self, x, y, r, fill, stroke=None, sw=0.6):
        y = self.h - y
        k = 0.5522847498 * r
        self._fill(fill)
        self._stroke(stroke or fill)
        self.ops.append(
            f"{sw:.2f} w {x+r:.2f} {y:.2f} m "
            f"{x+r:.2f} {y+k:.2f} {x+k:.2f} {y+r:.2f} {x:.2f} {y+r:.2f} c "
            f"{x-k:.2f} {y+r:.2f} {x-r:.2f} {y+k:.2f} {x-r:.2f} {y:.2f} c "
            f"{x-r:.2f} {y-k:.2f} {x-k:.2f} {y-r:.2f} {x:.2f} {y-r:.2f} c "
            f"{x+k:.2f} {y-r:.2f} {x+r:.2f} {y-k:.2f} {x+r:.2f} {y:.2f} c B"
        )

    def poly(self, pts, fill, stroke=None, sw=0.6):
        pts = [(x, self.h - y) for x, y in pts]
        self._fill(fill)
        self._stroke(stroke or fill)
        first, rest = pts[0], pts[1:]
        cmd = f"{sw:.2f} w {first[0]:.2f} {first[1]:.2f} m "
        cmd += " ".join(f"{x:.2f} {y:.2f} l" for x, y in rest)
        self.ops.append(cmd + " h B")

    def arrow(self, x1, y1, x2, y2, color=COLORS["line"]):
        self.line(x1, y1, x2, y2, color, 0.9)
        if abs(x2 - x1) >= abs(y2 - y1):
            s = 1 if x2 >= x1 else -1
            pts = [(x2, y2), (x2 - s * 9, y2 - 5), (x2 - s * 9, y2 + 5)]
        else:
            s = 1 if y2 >= y1 else -1
            pts = [(x2, y2), (x2 - 5, y2 - s * 9), (x2 + 5, y2 - s * 9)]
        self.poly(pts, color, color, 0.4)

    def text(self, x, y, text, size=8, color=COLORS["ink"], bold=False, align="left"):
        self._fill(color)
        font = "/F2" if bold else "/F1"
        approx = len(text) * size * 0.47
        if align == "center":
            x -= approx / 2
        elif align == "right":
            x -= approx
        y = self.h - y
        self.ops.append(f"BT {font} {size:.1f} Tf {x:.2f} {y:.2f} Td ({pdf_text_escape(text)}) Tj ET")

    def save(self):
        content = "\n".join(self.ops).encode("latin-1")
        objs = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.w} {self.h}] "
                f"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
            ).encode("latin-1"),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
            b"<< /Length " + str(len(content)).encode("latin-1") + b" >>\nstream\n" + content + b"\nendstream",
        ]
        out = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
        offsets = []
        for i, obj in enumerate(objs, 1):
            offsets.append(sum(len(p) for p in out))
            out.append(f"{i} 0 obj\n".encode("latin-1"))
            out.append(obj)
            out.append(b"\nendobj\n")
        xref = sum(len(p) for p in out)
        out.append(f"xref\n0 {len(objs)+1}\n".encode("latin-1"))
        out.append(b"0000000000 65535 f \n")
        for off in offsets:
            out.append(f"{off:010d} 00000 n \n".encode("latin-1"))
        out.append(f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("latin-1"))
        self.path.write_bytes(b"".join(out))


def box(svg, pdf, x, y, w, h, title, lines, fill, stroke, num):
    svg.rect(x, y, w, h, fill, stroke, 1.2, 10)
    pdf.rect(x, y, w, h, fill, stroke, 0.7)
    svg.circle(x + 20, y + 22, 12, COLORS["paper"], stroke, 1)
    pdf.circle(x + 20, y + 22, 12, COLORS["paper"], stroke, 0.7)
    svg.text(x + 20, y + 26, str(num), "label", "middle")
    pdf.text(x + 20, y + 26, str(num), 8, COLORS["ink"], True, "center")
    svg.text(x + 42, y + 26, title, "label")
    pdf.text(x + 42, y + 26, title, 8.5, COLORS["ink"], True)
    for i, line in enumerate(lines):
        svg.text(x + 42, y + 48 + i * 16, line, "body")
        pdf.text(x + 42, y + 48 + i * 16, line, 7.2, COLORS["muted"])


def pipeline(base):
    w, h = 900, 330
    svg, pdf = SVG(base.with_suffix(".svg"), w, h), PDF(base.with_suffix(".pdf"), w, h)
    svg.rect(0, 0, w, h, COLORS["paper"])
    pdf.rect(0, 0, w, h, COLORS["paper"])
    svg.text(30, 38, "a", "panel")
    svg.text(58, 38, "Black-box tone-modelling pipeline", "title")
    svg.text(58, 62, "A paired dry/wet workflow links data capture, alignment, learning and evaluation.", "subtitle")
    pdf.text(30, 38, "a", 10, COLORS["ink"], True)
    pdf.text(58, 38, "Black-box tone-modelling pipeline", 13, COLORS["ink"], True)
    pdf.text(58, 62, "A paired dry/wet workflow links data capture, alignment, learning and evaluation.", 8, COLORS["muted"])

    top = [
        (45, 105, 145, 78, "Dry input", ["source performance", "or test signal"], COLORS["blue_fill"], COLORS["blue"]),
        (225, 105, 155, 78, "Target device", ["amplifier, pedal,", "cabinet or plugin"], COLORS["gold_fill"], COLORS["gold"]),
        (415, 105, 145, 78, "Wet output", ["recorded target", "tone response"], COLORS["rose_fill"], COLORS["rose"]),
        (595, 105, 190, 78, "Alignment", ["sample delay", "estimated and trimmed"], COLORS["teal_fill"], COLORS["teal"]),
    ]
    bottom = [
        (120, 225, 150, 72, "Chunking", ["causal windows", "and data splits"], COLORS["panel"], COLORS["muted"]),
        (315, 225, 160, 72, "Training", ["LSTM, TCN,", "WaveNet or NAM"], COLORS["violet_fill"], COLORS["violet"]),
        (520, 225, 150, 72, "Inference", ["latency-aware", "causal processing"], COLORS["blue_fill"], COLORS["blue"]),
        (715, 225, 135, 72, "Evaluation", ["ESR, spectra,", "latency and ABX"], COLORS["rose_fill"], COLORS["rose"]),
    ]
    for i, spec in enumerate(top + bottom, 1):
        box(svg, pdf, *spec, i)

    arrows = [(190, 144, 225, 144), (380, 144, 415, 144), (560, 144, 595, 144),
              (690, 183, 195, 225), (270, 261, 315, 261), (475, 261, 520, 261), (670, 261, 715, 261)]
    for a in arrows:
        svg.arrow(*a)
        pdf.arrow(*a)

    svg.text(45, 318, "Design note: this is a planned methodology diagram, not a claim of completed system performance.", "subtitle")
    pdf.text(45, 318, "Design note: planned methodology, not completed system performance.", 7.2, COLORS["muted"])
    svg.save()
    pdf.save()


def dilation(base):
    w, h = 900, 360
    svg, pdf = SVG(base.with_suffix(".svg"), w, h), PDF(base.with_suffix(".pdf"), w, h)
    svg.rect(0, 0, w, h, COLORS["paper"])
    pdf.rect(0, 0, w, h, COLORS["paper"])
    svg.text(30, 38, "b", "panel")
    svg.text(58, 38, "Dilated causal convolution", "title")
    svg.text(58, 62, "Each row highlights the input samples used by one causal convolution layer.", "subtitle")
    pdf.text(30, 38, "b", 10, COLORS["ink"], True)
    pdf.text(58, 38, "Dilated causal convolution", 13, COLORS["ink"], True)
    pdf.text(58, 62, "Each row highlights the input samples used by one causal convolution layer.", 8, COLORS["muted"])

    x0, dx = 170, 28
    xs = [x0 + i * dx for i in range(17)]
    row_ys = {8: 105, 4: 155, 2: 205, 1: 255}
    colors = {1: COLORS["blue"], 2: COLORS["teal"], 4: COLORS["gold"], 8: COLORS["violet"]}

    svg.text(72, 90, "dilation", "label")
    pdf.text(72, 90, "dilation", 8, COLORS["ink"], True)
    svg.text(166, 322, "older samples", "subtitle")
    svg.text(596, 322, "current sample", "subtitle", "end")
    pdf.text(166, 322, "older samples", 7, COLORS["muted"])
    pdf.text(596, 322, "current sample", 7, COLORS["muted"], False, "right")

    for dilation, y in row_ys.items():
        color = colors[dilation]
        svg.text(92, y + 4, f"d = {dilation}", "label")
        pdf.text(92, y + 4, f"d = {dilation}", 8, color, True)
        svg.line(xs[0], y, xs[-1], y, COLORS["line"], 1.0)
        pdf.line(xs[0], y, xs[-1], y, COLORS["line"], 0.45)
        selected = {16 - k * dilation for k in range(4) if 16 - k * dilation >= 0}
        for i, x in enumerate(xs):
            is_selected = i in selected
            fill = color if is_selected else COLORS["paper"]
            stroke = color if is_selected else COLORS["line"]
            r = 8 if is_selected else 5.5
            svg.circle(x, y, r, fill, stroke, 1.1)
            pdf.circle(x, y, r, fill, stroke, 0.55)
        svg.text(620, y + 4, f"tap interval {dilation}", "subtitle")
        pdf.text(620, y + 4, f"tap interval {dilation}", 7, COLORS["muted"])

    label_positions = [(0, "x[n-16]"), (4, "x[n-12]"), (8, "x[n-8]"), (12, "x[n-4]"), (16, "x[n]")]
    for i, label in label_positions:
        svg.text(xs[i], 300, label, "subtitle", "middle")
        pdf.text(xs[i], 300, label, 6.5, COLORS["muted"], False, "center")

    svg.line(735, 92, 735, 268, COLORS["line"], 1.2)
    pdf.line(735, 92, 735, 268, COLORS["line"], 0.65)
    for y in row_ys.values():
        svg.arrow(682, y, 735, y, COLORS["line"])
        pdf.arrow(682, y, 735, y, COLORS["line"])
    svg.rect(765, 126, 95, 50, COLORS["paper"], COLORS["blue"], 1.2, 8)
    pdf.rect(765, 126, 95, 50, COLORS["paper"], COLORS["blue"], 0.75)
    svg.text(812, 156, "y[n]", "label", "middle")
    pdf.text(812, 156, "y[n]", 9, COLORS["ink"], True, "center")
    svg.arrow(735, 151, 765, 151, COLORS["blue"])
    pdf.arrow(735, 151, 765, 151, COLORS["blue"])

    svg.rect(725, 218, 145, 62, COLORS["panel"], COLORS["line"], 0.8, 8)
    pdf.rect(725, 218, 145, 62, COLORS["panel"], COLORS["line"], 0.5)
    svg.text(743, 241, "Causal rule", "label")
    svg.text(743, 261, "Only present and", "body")
    svg.text(743, 277, "past samples are used.", "body")
    pdf.text(743, 241, "Causal rule", 8, COLORS["ink"], True)
    pdf.text(743, 261, "Only present and", 7, COLORS["muted"])
    pdf.text(743, 277, "past samples are used.", 7, COLORS["muted"])
    svg.save()
    pdf.save()


if __name__ == "__main__":
    out = Path(__file__).resolve().parent
    pipeline(out / "pipeline")
    dilation(out / "dilated_causal_convolution")
