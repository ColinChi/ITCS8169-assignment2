"""Build the two-page report as PDF (report/assignment2_report.pdf) and Word (report/assignment2_report.docx).

Both renderers use the same text and the same Table 1 data. Table numbers are read from results/finetune.csv,
results/ssl_probes.csv and results/test/*.json (run ``python scripts/summarize.py`` first). Name and repository
URL come from report/report_metadata.json. Text uses a small inline markup: <b>, <i>, <super>, <font ...>
(monospace), <link href=...>, &nbsp;.

Usage: python scripts/build_report.py [--pdf] [--docx]     (default: both; --docx needs python-docx)
"""

from __future__ import annotations

import argparse
import csv
import json
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = ROOT / "report" / "assignment2_report.pdf"
OUT_DOCX = ROOT / "report" / "assignment2_report.docx"
TITLE = "Self-Supervised Representation Learning for Small-Data Scene Recognition"
COL_WIDTHS_IN = [3.55, 0.5, 0.55, 0.95, 0.95, 0.45]


def pm(mean, std) -> str:
    return f"{100 * float(mean):.1f} ± {100 * float(std):.1f}" if mean != "" else "–"


def content() -> dict:
    """Everything that appears in the report, independent of the output format."""
    meta = json.loads((ROOT / "report" / "report_metadata.json").read_text())
    ft = {r["group"]: r for r in csv.DictReader(open(ROOT / "results" / "finetune.csv"))}
    probes = {r["run"]: r for r in csv.DictReader(open(ROOT / "results" / "ssl_probes.csv"))}
    final = json.loads((ROOT / "results" / "test" / "final_ensemble.json").read_text())
    ens1920_test = json.loads((ROOT / "results" / "test" / "ensemble_A_1920.json").read_text())["ensemble"]
    ens1920_val = json.loads((ROOT / "results" / "val_ensemble_A.json").read_text())["ensemble"]
    n_ok = round(final["ensemble"] * 400)

    box = (f"<b>Best test accuracy: {100 * final['ensemble']:.2f}% ({n_ok}/400).</b> Ensemble of six ResNet-18 "
           "networks (three DINO-pretrained, three SimCLR-pretrained), each pretrained and fine-tuned from random "
           "initialization on the 2,400 provided training images only; no external data or weights are used. "
           "The repository contains all code, the six FP16 checkpoints, and the commands for every experiment.")

    setup = (
        "The data consist of 2,400 training images (150 per class, 16 classes) and 400 test images. Fifteen "
        "classes are grayscale; <i>Flower</i> is the only RGB class. All model selection used a fixed, "
        "class-stratified split of the training images into 1,920 training and 480 validation images (exact "
        "duplicates kept on one side). For every comparison, self-supervised pretraining used only the 1,920 "
        "training-split images; validation and test images never enter a loss. Images are decoded once to "
        "256×256 and augmented on the GPU; networks receive 224×224 inputs (128 and 256 px are studied as controls). "
        "Unless stated otherwise the network is a torchvision ResNet-18 with "
        "<font name='Courier' size='8.5'>weights=None</font>. "
        "Training budgets are fixed and the final-epoch model is reported (no early stopping), so the final "
        "configuration can be retrained on all 2,400 images without a validation set. Representation quality is "
        "measured on frozen features "
        "with a weighted cosine kNN classifier (k = 20) and a linear logistic-regression probe. The test set "
        "was evaluated only after the final configuration had been fixed on validation.")

    baseline = (
        "A randomly initialized ResNet-18 is trained with cross-entropy only for 200 epochs: SGD with "
        "Nesterov momentum 0.9, learning rate 0.1 (5 warm-up epochs, cosine decay), batch size 128, weight "
        "decay 5×10<super>-4</super>, label smoothing 0.1, and random resized crops (35–100% of the area), "
        "horizontal flips and mild brightness/contrast/saturation jitter. It reaches "
        f"{pm(ft['sup224']['val'], ft['sup224']['val_std'])}% validation and "
        f"{pm(ft['sup224']['test'], ft['sup224']['test_std'])}% test accuracy. Exactly the same recipe is "
        "used to fine-tune every pretrained encoder, so differences between rows of Table 1 isolate the "
        "effect of the initialization.")

    ssl = [
        "The main strategy is two-stage: 800 epochs of self-supervised pretraining without labels, after "
        "which the projection head is discarded and all layers are fine-tuned with the baseline recipe; joint "
        "optimization is evaluated as an alternative.",

        "<b>DINO (self-distillation).</b> A student and a teacher network of identical architecture map an "
        "image view to a probability distribution over K learned prototypes (three-layer MLP head with 2,048 "
        "hidden units, a 256-d L2-normalized bottleneck, and unit-norm prototype vectors). The teacher "
        "receives no gradients; its weights are an exponential moving average of the student's (momentum "
        "0.996 increasing to 1). For each image, two global crops (224 px, 25–100% of the area) and six local "
        "crops (96 px, 5–25%) are sampled. The teacher sees only the global crops, and the student is trained "
        "to predict the teacher's output for one global crop from every other crop by minimizing the "
        "cross-entropy between the teacher distribution (temperature 0.04, after subtracting a running mean "
        "of its outputs, the <i>center</i>) and the student distribution (temperature 0.1). Centering "
        "prevents a single prototype from dominating and the low teacher temperature prevents the uniform "
        "solution, so collapse is avoided without negative pairs. The local-to-global constraint requires a "
        "small region of a scene (e.g., part of a bed) to predict the representation of the whole image, "
        "which suits categories defined by their constituent objects and layout. Adaptations to "
        "this dataset: K = 1,024 prototypes instead of 65,536, since there are only 1,920 images; a fixed "
        "teacher temperature without warm-up; batch size 64 (30 updates per epoch); SGD with learning rate "
        "0.05 and weight decay 10<super>-4</super>. The teacher backbone is used downstream.",

        "<b>SimCLR (contrastive learning).</b> Two strongly augmented views of each image (random resized "
        "crops covering 8–100% of the area, color jitter, random grayscale, Gaussian blur) are embedded "
        "through a two-layer projection head (128-d); the NT-Xent loss (temperature 0.2) must identify the "
        "matching view among the 511 other views in a batch of 256. <b>Joint training</b> instead adds "
        "0.5 × the SimCLR loss, computed on the same mini-batch and backbone, to the cross-entropy loss "
        "during the 200-epoch supervised schedule from random initialization.",
    ]

    def row(label, group, probe=None, bold=False):
        r = ft.get(group, {})
        pr = probes.get(probe, {}) if probe else {}
        f = lambda k: f"{100 * float(pr[k]):.1f}" if pr.get(k, "") != "" else "–"  # noqa: E731
        cells = [label, f("knn"), f("linear"), pm(r.get("val", ""), r.get("val_std", "")),
                 pm(r.get("test", ""), r.get("test_std", "")), f"{100 * float(r['gap']):.1f}" if r else "–"]
        return {"kind": "data", "cells": cells, "bold": [bold, False, False, bold, bold, False]}

    sec = lambda t: {"kind": "section", "cells": [t]}  # noqa: E731
    table = [
        {"kind": "header", "cells": ["Method", "kNN", "Linear", "Val. acc.", "Test acc.", "Gap"]},
        sec("ResNet-18, 224 px"),
        row("Supervised baseline (random init.)", "sup224"),
        row("Supervised + MixUp/CutMix", "supmix224"),
        row("Joint CE + 0.5·SimCLR (random init.)", "joint224"),
        row("SimCLR → fine-tune", "ft_simclr224", "simclr224"),
        row("SimCLR → fine-tune + MixUp/CutMix", "mix_simclr224"),
        row("DINO → fine-tune", "ft_dino224", "dino224", bold=True),
        row("DINO → fine-tune + MixUp/CutMix", "mix_dino224"),
        sec("ResNet-18, resolution control (256 px is the native size of half of the images)"),
        row("128 px: supervised baseline", "sup128"),
        row("128 px: SimCLR → fine-tune", "ft_simclr128", "simclr128"),
        row("128 px: DINO → fine-tune", "ft_dino128", "dino128"),
        row("256 px: supervised baseline", "sup256"),
        row("256 px: SimCLR → fine-tune", "ft_simclr256", "simclr256"),
        row("256 px: DINO → fine-tune", "ft_dino256", "dino256"),
        sec("ResNet-50, 224 px (backbone control)"),
        row("Supervised baseline", "sup224_r50"),
        row("Supervised + MixUp/CutMix", "supmix224_r50"),
        row("DINO → fine-tune + MixUp/CutMix", "mix_dino224_r50", "dino224_r50"),
        sec("Final system, 224 px (flip test-time augmentation)"),
        {"kind": "final", "cells": ["Ensemble: 3 × DINO → FT + 3 × SimCLR → FT+MixUp, trained on 1,920 images",
                                    f"{100 * ens1920_val:.1f}", f"{100 * ens1920_test:.1f}", "–"],
         "bold": [False] * 4},
        {"kind": "final", "cells": ["Same ensemble, SSL and fine-tuning on all 2,400 images (submitted)",
                                    "–", f"{100 * final['ensemble']:.2f}", "–"],
         "bold": [True, False, True, False]},
    ]
    caption = ("<b>Table 1.</b> Accuracy (%). Except for the final-system rows: single models, mean ± std over three "
               "fine-tuning seeds, no test-time augmentation. kNN/Linear: frozen pretrained encoder, validation split. "
               "Gap: training minus validation accuracy. All rows except the last use the 1,920-image training split.")
    return {"meta": meta, "box": box, "table": table, "caption": caption, "sections": [
        ("1", "Setup and Evaluation Protocol", [setup]),
        ("2", "Supervised Baseline", [baseline]),
        ("3", "Self-Supervised Learning Strategy", ssl),
        ("4", "Results and Analysis", ["<TABLE>"] + ANALYSIS),
        ("5", "Limitations", LIMITATIONS),
        ("6", "Conclusion", [CONCLUSION]),
    ]}


# ----------------------------------------------------------------------------------------------- PDF

def render_pdf(c: dict) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    navy = colors.HexColor("#1f3a5f")
    base = dict(fontName="Times-Roman", fontSize=9.0, leading=10.7)
    body = ParagraphStyle("body", **base, alignment=TA_JUSTIFY, spaceAfter=2.5)
    small = ParagraphStyle("small", fontName="Times-Roman", fontSize=8.2, leading=9.4)
    cell = ParagraphStyle("cell", fontName="Times-Roman", fontSize=8.0, leading=8.8)
    cellb = ParagraphStyle("cellb", parent=cell, fontName="Times-Bold")
    h = ParagraphStyle("h", fontName="Times-Bold", fontSize=10.6, leading=12.5, textColor=navy,
                       spaceBefore=4, spaceAfter=1.5)
    title = ParagraphStyle("title", fontName="Times-Bold", fontSize=14.5, leading=17, alignment=TA_CENTER)
    sub = ParagraphStyle("sub", fontName="Times-Roman", fontSize=9.4, leading=11.5, alignment=TA_CENTER,
                         spaceAfter=4)
    P = Paragraph
    meta, url = c["meta"], c["meta"]["github_url"]

    story = [P(TITLE, title),
             P(f"ITCS 6169/8169 Assignment 2 &nbsp;·&nbsp; {meta['student_name']} &nbsp;·&nbsp; "
               f'<link href="{url}" color="#1155cc">{url}</link>', sub)]
    box = Table([[P(c["box"], body)]], colWidths=[7.5 * inch])
    box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, navy), ("BACKGROUND", (0, 0), (-1, -1),
                             colors.HexColor("#f2f5f9")), ("TOPPADDING", (0, 0), (-1, -1), 3),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    story += [box, Spacer(1, 2)]

    rows, style = [], [("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
                       ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                       ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black), ("TOPPADDING", (0, 0), (-1, -1), 0.4),
                       ("BOTTOMPADDING", (0, 0), (-1, -1), 0.4), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    for i, r in enumerate(c["table"]):
        if r["kind"] == "header":
            rows.append([P(f"<b>{t}</b>", cell) for t in r["cells"]])
        elif r["kind"] == "section":
            rows.append([P(f"<i>{r['cells'][0]}</i>", cell), "", "", "", "", ""])
            style += [("SPAN", (0, i), (-1, i)), ("BACKGROUND", (0, i), (-1, i), colors.HexColor("#eef1f5"))]
        elif r["kind"] == "final":
            t = [P(x, cellb if b else cell) for x, b in zip(r["cells"], r["bold"])]
            rows.append([t[0], "", "", t[1], t[2], t[3]])
            style += [("SPAN", (0, i), (2, i))]
        else:
            rows.append([P(x, cellb if b else cell) for x, b in zip(r["cells"], r["bold"])])
    table = Table(rows, colWidths=[w * inch for w in COL_WIDTHS_IN], repeatRows=1)
    table.setStyle(TableStyle(style))

    for num, name, paras in c["sections"]:
        head = P(f"{num}&nbsp;&nbsp;{name}", h)
        if paras[0] == "<TABLE>":
            story += [KeepTogether([head, table, P(c["caption"], small)]), Spacer(1, 3)]
            story += [P(t, body) for t in paras[1:]]
        else:
            story += [head] + [P(t, body) for t in paras]

    doc = SimpleDocTemplate(str(OUT_PDF), pagesize=letter, leftMargin=0.5 * inch, rightMargin=0.5 * inch,
                            topMargin=0.4 * inch, bottomMargin=0.4 * inch,
                            title="ITCS 6169/8169 Assignment 2 Report", author=meta["student_name"])
    doc.build(story)
    print(f"wrote {OUT_PDF}")


# ---------------------------------------------------------------------------------------------- DOCX

class _Runs(HTMLParser):
    """Split the inline markup into (text, bold, italic, superscript, monospace, link) runs."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.runs, self.state, self.link = [], {"b": 0, "i": 0, "super": 0, "font": 0}, None

    def handle_starttag(self, tag, attrs):
        if tag in self.state:
            self.state[tag] += 1
        if tag == "link":
            self.link = dict(attrs).get("href")

    def handle_endtag(self, tag):
        if tag in self.state:
            self.state[tag] -= 1
        if tag == "link":
            self.link = None

    def handle_data(self, data):
        s = self.state
        self.runs.append((data.replace("\xa0", " "), s["b"] > 0, s["i"] > 0, s["super"] > 0, s["font"] > 0, self.link))


def render_docx(c: dict) -> None:
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.opc.constants import RELATIONSHIP_TYPE
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor

    navy, font = RGBColor(0x1F, 0x3A, 0x5F), "Times New Roman"
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.left_margin = sec.right_margin = Inches(0.5)
    sec.top_margin = sec.bottom_margin = Inches(0.4)
    doc.core_properties.title = "ITCS 6169/8169 Assignment 2 Report"
    doc.core_properties.author = c["meta"]["student_name"]

    def set_font(style_or_run, size=None, bold=None, color=None):
        f = style_or_run.font
        f.name = font
        rpr = style_or_run.element.get_or_add_rPr()
        fonts = rpr.find(qn("w:rFonts"))
        if fonts is None:
            fonts = OxmlElement("w:rFonts")
            rpr.insert(0, fonts)
        for k in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            fonts.set(qn(k), font)
        if size:
            f.size = Pt(size)
        if bold is not None:
            f.bold = bold
        if color is not None:
            f.color.rgb = color

    normal = doc.styles["Normal"]
    set_font(normal, 9.0)
    normal.paragraph_format.space_after = Pt(2.5)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.line_spacing = Pt(10.7)
    h1 = doc.styles["Heading 1"]
    set_font(h1, 10.6, bold=True, color=navy)
    h1.font.italic = False
    h1.paragraph_format.space_before = Pt(4)
    h1.paragraph_format.space_after = Pt(1.5)
    h1.paragraph_format.line_spacing = Pt(12.5)
    h1.paragraph_format.keep_with_next = True

    def hyperlink(par, url, text, size=None):
        r_id = par.part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
        link = OxmlElement("w:hyperlink")
        link.set(qn("r:id"), r_id)
        run = OxmlElement("w:r")
        rpr = OxmlElement("w:rPr")
        for tag, val in (("w:color", "1155CC"), ("w:u", "single")):
            e = OxmlElement(tag)
            e.set(qn("w:val"), val)
            rpr.append(e)
        if size:
            sz = OxmlElement("w:sz")
            sz.set(qn("w:val"), str(int(size * 2)))
            rpr.append(sz)
        run.append(rpr)
        t = OxmlElement("w:t")
        t.text = text
        t.set(qn("xml:space"), "preserve")
        run.append(t)
        link.append(run)
        par._p.append(link)

    def add_markup(par, text, size=None, bold_all=False):
        parser = _Runs()
        parser.feed(text)
        for txt, b, i, sup, mono, link in parser.runs:
            if link:
                hyperlink(par, link, txt, size)
                continue
            run = par.add_run(txt)
            set_font(run, size, bold=b or bold_all or None)
            run.font.italic = i or None
            run.font.superscript = sup or None
            if mono:
                run.font.name = "Courier New"
                run.element.rPr.rFonts.set(qn("w:hAnsi"), "Courier New")
                run.element.rPr.rFonts.set(qn("w:ascii"), "Courier New")
                run.font.size = Pt(8.5)
        return par

    def para(text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=None, space_after=None, line=None):
        p = doc.add_paragraph()
        p.alignment = align
        if space_after is not None:
            p.paragraph_format.space_after = Pt(space_after)
        if line is not None:
            p.paragraph_format.line_spacing = Pt(line)
        return add_markup(p, text, size)

    def shade(cell, hex_fill):
        tcpr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), hex_fill)
        tcpr.append(shd)

    def borders(cell, **edges):  # edges: top/bottom/left/right -> (size in 1/8 pt, hex color) or None
        tcpr = cell._tc.get_or_add_tcPr()
        tb = OxmlElement("w:tcBorders")
        for edge in ("top", "left", "bottom", "right"):
            e = OxmlElement(f"w:{edge}")
            spec = edges.get(edge)
            if spec:
                e.set(qn("w:val"), "single")
                e.set(qn("w:sz"), str(spec[0]))
                e.set(qn("w:color"), spec[1])
            else:
                e.set(qn("w:val"), "nil")
            tb.append(e)
        shd = tcpr.find(qn("w:shd"))
        if shd is not None:  # schema order: tcBorders precedes shd
            shd.addprevious(tb)
        else:
            tcpr.append(tb)

    def _before_look(tblpr, elem):  # schema order: tblLayout, tblCellMar precede tblLook
        look = tblpr.find(qn("w:tblLook"))
        if look is not None:
            look.addprevious(elem)
        else:
            tblpr.append(elem)

    def cell_margins(table, top=10, bottom=10, left=60, right=60):
        tblpr = table._tbl.tblPr
        mar = OxmlElement("w:tblCellMar")
        for k, v in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
            e = OxmlElement(f"w:{k}")
            e.set(qn("w:w"), str(v))
            e.set(qn("w:type"), "dxa")
            mar.append(e)
        _before_look(tblpr, mar)

    def fixed_widths(table, widths_in):
        table.autofit = False  # writes <w:tblLayout w:type="fixed"/>
        for r in table.rows:
            for cell, w in zip(r.cells, widths_in):
                cell.width = Inches(w)

    meta, url = c["meta"], c["meta"]["github_url"]
    p = para(f"<b>{TITLE}</b>", WD_ALIGN_PARAGRAPH.CENTER, 14.5, 0, 17)
    p = para(f"ITCS 6169/8169 Assignment 2 · {meta['student_name']} · ", WD_ALIGN_PARAGRAPH.CENTER, 9.4, 4, 11.5)
    hyperlink(p, url, url, 9.4)

    box = doc.add_table(rows=1, cols=1)
    box.alignment = WD_TABLE_ALIGNMENT.CENTER
    fixed_widths(box, [7.5])
    cell_margins(box, 50, 30, 80, 80)
    bc = box.rows[0].cells[0]
    shade(bc, "F2F5F9")
    borders(bc, top=(5, "1F3A5F"), bottom=(5, "1F3A5F"), left=(5, "1F3A5F"), right=(5, "1F3A5F"))
    bp = bc.paragraphs[0]
    bp.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    bp.paragraph_format.space_after = Pt(0)
    add_markup(bp, c["box"])

    def add_table():
        rows = c["table"]
        t = doc.add_table(rows=len(rows), cols=6)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        fixed_widths(t, COL_WIDTHS_IN)
        cell_margins(t, 0, 0, 60, 60)
        last = len(rows) - 1
        for i, r in enumerate(rows):
            cells = t.rows[i].cells
            top = (6, "000000") if i == 0 else None
            bottom = (4, "000000") if i == 0 else ((6, "000000") if i == last else None)
            if r["kind"] == "section":
                merged = cells[0].merge(cells[5])
                shade(merged, "EEF1F5")
                borders(merged, top=top, bottom=bottom)
                texts, targets, bolds = [f"<i>{r['cells'][0]}</i>"], [merged], [False]
            elif r["kind"] == "final":
                merged = cells[0].merge(cells[2])
                targets = [merged, cells[3], cells[4], cells[5]]
                texts, bolds = r["cells"], r["bold"]
            else:
                targets, texts = cells, r["cells"]
                bolds = [True] * 6 if r["kind"] == "header" else r["bold"]
            for cell in (targets if r["kind"] != "section" else []):
                borders(cell, top=top, bottom=bottom)
            for cell, text, b in zip(targets, texts, bolds):
                cp = cell.paragraphs[0]
                cp.paragraph_format.space_after = Pt(0)
                cp.paragraph_format.line_spacing = Pt(9.4)
                cp.paragraph_format.keep_with_next = True
                add_markup(cp, text, 8.0, bold_all=b)
        cap = para(c["caption"], WD_ALIGN_PARAGRAPH.JUSTIFY, 8.2, 3, 9.4)
        cap.paragraph_format.space_before = Pt(1)

    for num, name, paras in c["sections"]:
        doc.add_heading(f"{num}  {name}", level=1)
        for t in paras:
            if t == "<TABLE>":
                add_table()
            else:
                para(t)

    zoom = doc.settings.element.find(qn("w:zoom"))  # default template omits the required percent
    if zoom is not None:
        zoom.set(qn("w:percent"), "100")
    doc.save(OUT_DOCX)
    print(f"wrote {OUT_DOCX}")


ANALYSIS = [
    "<b>SSL improves accuracy with 2,400 images.</b> With an identical fine-tuning recipe, DINO initialization "
    "improves on random initialization by 5.8 points on validation and 4.3 points on test, and SimCLR by 3.8 and "
    "1.6 points. When both are regularized with MixUp/CutMix, the SSL-initialized models are 1.5–1.9 points better "
    "on validation and 4.2–4.7 points better on test. On the test set MixUp/CutMix does not improve the randomly "
    "initialized model (83.7% in both cases) but improves both SSL-initialized models, which suggests that strong "
    "regularization and SSL pretraining are complementary rather than redundant.",

    "<b>SSL reduces overfitting.</b> All models fit the training set almost perfectly, but the train–validation "
    "gap falls from 15.0 points (random initialization) to 11.2 (SimCLR) and 9.2 (DINO), and the validation "
    "negative log-likelihood falls from 0.60 to 0.49 (DINO). The pretrained encoders already separate the classes "
    "to a large extent (83% linear-probe accuracy), so the labels are used to refine rather than to create the "
    "features.",

    "<b>Pretraining followed by fine-tuning outperforms joint training.</b> Adding 0.5 × SimCLR to the supervised "
    "loss improves validation accuracy by 1.6 points but lowers test accuracy by 1.1 points, well below SimCLR "
    "pretraining followed by fine-tuning (+3.8 / +1.6). In joint training the two objectives compete for the same "
    "features from the first step, and the SSL objective receives 200 instead of 800 epochs, so the comparison is "
    "not compute-matched. The result is nevertheless consistent with the benefit of SSL in this setting coming "
    "mainly from a better initialization rather than from SSL acting as a regularizer.",

    "<b>A more complex method helps, but less than its cost suggests.</b> DINO takes about 2.4× the wall-clock "
    "time of SimCLR (24 vs. 10 min) and has more hyperparameters. Its frozen features are only slightly better "
    "(kNN 82.9 vs. 80.6; linear 83.1 vs. 82.5), and its fine-tuning advantage (+2.0 validation, +2.8 test) "
    "shrinks to within seed variance once MixUp/CutMix is added (89.4 vs. 89.8 validation; 88.3 vs. 87.8 test). "
    "A plausible reason for SimCLR's weaker plain fine-tuning is class collision: with 16 balanced classes, about "
    "6% of the 510 negatives of each view are images of the same class, which the loss pushes apart. DINO uses no "
    "negative pairs.",

    "<b>SSL saturates early.</b> DINO's kNN accuracy reaches 82.5% after 250 of 800 epochs and then remains "
    "between 81% and 83%, while the loss keeps decreasing from 1.36 to 0.15 nats. Because the cross-entropy is "
    "an upper bound on the entropy of the teacher's output, the teacher's distribution over prototypes becomes "
    "nearly one-hot for each image, which suggests that further training mainly memorizes individual images. With 1,920 "
    "images, the amount of data rather than the training length limits the learned representation.",

    "<b>Resolution and backbone.</b> Increasing the input size from 128 to 224 px improves every method by a "
    "similar amount (3.3–4.0 validation points, 3.6–4.3 test points); the SSL gain is approximately additive to "
    "the resolution gain (DINO: +6.5 / +4.2 at 128 px, +5.8 / +4.3 at 224 px). Going further to 256 px, the native "
    "size of half of the images, changes validation accuracy inconsistently (−0.8 to +1.7 points over seven "
    "configurations, including joint training and MixUp/CutMix; see repository) but raises single-model test "
    "accuracy by 0.5–1.5 points for every configuration; since the two small evaluation sets "
    "disagree, we cannot conclude that 256 px is better than 224 px. Frozen-feature quality depends little on "
    "resolution (DINO linear probe 83.5, 83.1 and 84.0 at 128, 224 and 256 px). Trained from random initialization, ResNet-50 is worse and less stable than ResNet-18 (82.2 ± 2.0 "
    "vs. 85.0 ± 0.4 validation). DINO pretraining improves it by a similar margin as ResNet-18 and markedly reduces "
    "its sensitivity to the seed (test standard deviation 4.3 → 1.0 with MixUp/CutMix), yielding the highest "
    "single-model test accuracy among the 224 px configurations (89.1%). Its frozen features are nevertheless not better than ResNet-18's (kNN "
    "81.0, linear 82.7), again pointing to data as the limiting factor.",

    "<b>Final system.</b> The two objectives make partly different errors: an ensemble of the three DINO models "
    "reaches 91.3% validation accuracy, and an ensemble of these models with the three SimCLR + MixUp/CutMix models "
    "reaches 92.5%. This ensemble was selected on validation among three ensembles specified in advance (92.5%, "
    "92.1%, 90.6%), before the test set was used. Repeating the selected configuration, including SSL pretraining, "
    "on all 2,400 images raises its test accuracy from 88.5% to 90.75%. The 256 px study was run after this test "
    "evaluation and subjected to the same rule: its best ensemble reached 91.7% validation accuracy (vs. 92.5%), so "
    "the submitted system was not changed and the 256 px ensemble trained on all images was not evaluated on test.",
]

LIMITATIONS = [
    "<b>Statistical resolution.</b> All results use one split with 480 validation and 400 test images (one image "
    "corresponds to 0.21 and 0.25 points). Several differences discussed above are within seed variance, and "
    "validation accuracy exceeds test accuracy for every configuration (by up to 4.9 points). Each SSL configuration was pretrained "
    "once, so the reported standard deviations reflect only fine-tuning variance. "
    "<b>Tuning and controls.</b> The DINO hyperparameters (1,024 prototypes, teacher temperature 0.04) were taken "
    "from a preliminary low-resolution exploration and not re-tuned; SSL augmentations were not ablated; the joint "
    "training comparison is not compute-matched. The submitted models have no held-out validation set, so their "
    "accuracy can only be inferred from the 1,920-image counterparts. "
    "<b>Dataset artifacts.</b> <i>Flower</i> is the only RGB class, and image size is class-dependent (e.g., all "
    "<i>Coast</i> images are 256×256 and all <i>Suburb</i> images 330×220); after stretching to a square, both "
    "color and aspect distortion are potential shortcuts. Only convolutional backbones were studied; masked image "
    "modeling with Vision Transformers was not evaluated.",
]

CONCLUSION = (
    "Self-supervised pretraining on the same 2,400 images is clearly beneficial in this limited-data regime. Under "
    "an identical fine-tuning recipe it improves ResNet-18 test accuracy by 1.6–4.3 points, reduces overfitting, "
    "and stabilizes a larger backbone. The benefit is obtained by pretraining followed by fine-tuning rather than "
    "by joint optimization, is approximately independent of input resolution, and is largest for self-distillation "
    "without negative pairs. At the same time, frozen-feature quality saturates after a few hundred epochs and "
    "improves little with a larger backbone or a higher resolution, so the amount of data, rather than the choice of "
    "objective, is the main limitation. AI coding assistants (Claude Code) were used to implement the pipeline, "
    "run experiments and draft this report; see AI_USAGE.md in the repository."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--docx", action="store_true")
    a = ap.parse_args()
    c = content()
    if a.pdf or not a.docx:
        render_pdf(c)
    if a.docx or not a.pdf:
        render_docx(c)


if __name__ == "__main__":
    main()
