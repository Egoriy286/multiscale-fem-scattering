#!/usr/bin/env python3
"""Render method_define/chapters.json into a polished Russian technical manual PDF."""
from __future__ import annotations
import json, os, re, sys, subprocess, tempfile
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent
INPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'chapters.json'
OUTPUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'output/pdf/multiscale_svd_sno_manual.pdf'
EQ_DIR = ROOT / 'tmp/equations'
EQ_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
    PageBreak, Image, Table, TableStyle, KeepTogether, Preformatted, HRFlowable)

FONT_PATHS = {
    'Arial': '/System/Library/Fonts/Supplemental/Arial.ttf',
    'Arial-Bold': '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
    'Arial-Italic': '/System/Library/Fonts/Supplemental/Arial Italic.ttf',
    'Arial-BoldItalic': '/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf',
}
for name, path in FONT_PATHS.items():
    if os.path.exists(path): pdfmetrics.registerFont(TTFont(name, path))
if 'Arial' not in pdfmetrics.getRegisteredFontNames():
    raise RuntimeError('Unicode-capable Arial font unavailable; cannot safely render Cyrillic.')
pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='Arial-Bold' if 'Arial-Bold' in pdfmetrics.getRegisteredFontNames() else 'Arial', italic='Arial-Italic' if 'Arial-Italic' in pdfmetrics.getRegisteredFontNames() else 'Arial', boldItalic='Arial-BoldItalic' if 'Arial-BoldItalic' in pdfmetrics.getRegisteredFontNames() else 'Arial-Bold')

W, H = A4
M_L, M_R, M_T, M_B = 23*mm, 21*mm, 24*mm, 21*mm

class ManualDoc(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, pagesize=A4, leftMargin=M_L, rightMargin=M_R,
                         topMargin=M_T, bottomMargin=M_B, title='Многомасштабный SVD-SNO: учебное руководство',
                         author='Учебное руководство', **kwargs)
        frame = Frame(M_L, M_B, W-M_L-M_R, H-M_T-M_B, id='main', leftPadding=0, rightPadding=0,
                      topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id='manual', frames=[frame], onPage=self.decorate))
    def decorate(self, canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setStrokeColor(colors.HexColor('#D5DCE5'))
            canvas.setLineWidth(.45)
            canvas.line(M_L, H-15*mm, W-M_R, H-15*mm)
            canvas.setFont('Arial', 7.4)
            canvas.setFillColor(colors.HexColor('#64748B'))
            canvas.drawString(M_L, H-12.2*mm, 'МНОГОМАСШТАБНЫЕ НЕЙРОСЕТЕВЫЕ ОПЕРАТОРЫ')
            canvas.drawRightString(W-M_R, 11*mm, f'{doc.page}')
            canvas.line(M_L, 16*mm, W-M_R, 16*mm)
        canvas.restoreState()

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='CoverTitle', fontName='Arial-Bold', fontSize=25, leading=32, textColor=colors.HexColor('#102A43'), alignment=TA_LEFT, spaceAfter=9*mm))
styles.add(ParagraphStyle(name='CoverSub', fontName='Arial', fontSize=13, leading=19, textColor=colors.HexColor('#486581'), spaceAfter=7*mm))
styles.add(ParagraphStyle(name='H1x', fontName='Arial-Bold', fontSize=18, leading=23, textColor=colors.HexColor('#102A43'), spaceBefore=2*mm, spaceAfter=5*mm, keepWithNext=True))
styles.add(ParagraphStyle(name='H2x', fontName='Arial-Bold', fontSize=12.6, leading=16, textColor=colors.HexColor('#176B87'), spaceBefore=4*mm, spaceAfter=2*mm, keepWithNext=True))
styles.add(ParagraphStyle(name='H3x', fontName='Arial-Bold', fontSize=10.2, leading=13, textColor=colors.HexColor('#334E68'), spaceBefore=2.5*mm, spaceAfter=1.5*mm, keepWithNext=True))
styles.add(ParagraphStyle(name='Bodyx', fontName='Arial', fontSize=10.5, leading=14, textColor=colors.HexColor('#243B53'), alignment=TA_JUSTIFY, spaceAfter=2.6*mm, allowWidows=0, allowOrphans=0))
styles.add(ParagraphStyle(name='Smallx', parent=styles['Bodyx'], fontSize=8.6, leading=11.2, alignment=TA_LEFT, textColor=colors.HexColor('#486581'), spaceAfter=1.5*mm))
styles.add(ParagraphStyle(name='Captionx', fontName='Arial-Italic' if 'Arial-Italic' in pdfmetrics.getRegisteredFontNames() else 'Arial', fontSize=8.1, leading=11, textColor=colors.HexColor('#486581'), alignment=TA_CENTER, spaceBefore=1.2*mm, spaceAfter=3*mm))
styles.add(ParagraphStyle(name='EqCaptionx', parent=styles['Captionx'], fontSize=7.7, spaceBefore=1*mm, spaceAfter=2.5*mm))
styles.add(ParagraphStyle(name='TOCx', fontName='Arial', fontSize=10, leading=15, textColor=colors.HexColor('#243B53'), leftIndent=3*mm, spaceAfter=1.5*mm))
styles.add(ParagraphStyle(name='Codex', fontName='Courier', fontSize=8.0, leading=10.2, textColor=colors.HexColor('#243B53'), backColor=colors.HexColor('#F3F6F9'), borderColor=colors.HexColor('#D5DCE5'), borderWidth=.5, borderPadding=7, spaceBefore=1*mm, spaceAfter=3*mm))


def safe_markup(s):
    s = str(s).translate(str.maketrans({'‐':'-', '‑':'-', '‒':'-', '–':'-', '—':'-', '―':'-'}))
    # Permit a small, deliberate subset of ReportLab markup while escaping raw text.
    if re.search(r'<(b|i|u|sub|sup|br|font|a)(?:\s|>)', str(s), re.I):
        return str(s).replace('&', '&amp;') if '&' in str(s) and not re.search(r'&(amp|lt|gt|quot|apos);', str(s)) else str(s)
    return escape(str(s)).replace('\n', '<br/>')

def p(text, style='Bodyx'):
    return Paragraph(safe_markup(text), styles[style])

def equation_image(latex, index):
    dest = EQ_DIR / f'eq_{index:03d}.png'
    # Use the repository venv's Matplotlib for mathtext rasterization; no LaTeX install required.
    # Matplotlib mathtext is intentionally used (no external TeX dependency).
    # Normalize two common LaTeX shortcuts that mathtext does not accept.
    tex = str(latex).replace(r'\frac1N', r'\frac{1}{N}')
    tex = re.sub(r'\\mathcal\s+([A-Za-z])', r'\\mathcal{\1}', tex)
    script = """import sys, matplotlib; matplotlib.use('Agg')\nimport matplotlib.pyplot as plt\ntex,out=sys.argv[1],sys.argv[2]\nfig=plt.figure(figsize=(12,0.8),dpi=220); fig.patch.set_alpha(0)\nfig.text(.5,.5,'$'+tex+'$',ha='center',va='center',fontsize=16,color='#102A43')\nfig.savefig(out,transparent=True,bbox_inches='tight',pad_inches=.08); plt.close(fig)\n"""
    vpy = ROOT.parent / '.venv/bin/python'
    try:
        subprocess.run([str(vpy), '-c', script, tex, str(dest)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return dest
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode('utf-8', errors='replace')[-700:] if exc.stderr else str(exc)
        raise RuntimeError(f'Mathtext could not render equation {index}: {latex}\\n{detail}') from exc

def image_block(path, maxw, maxh):
    path = Path(path)
    if not path.is_absolute(): path = ROOT / path
    if not path.exists(): return p(f'[Иллюстрация не найдена: {path.name}]', 'Smallx')
    im = Image(str(path))
    scale = min(maxw/im.imageWidth, maxh/im.imageHeight, 1.0)
    im.drawWidth, im.drawHeight = im.imageWidth*scale, im.imageHeight*scale
    im.hAlign = 'CENTER'
    return im

def render_block(b, eqcount):
    if isinstance(b, str): return [p(b)]
    typ = str(b.get('type', 'p')).lower()
    text = b.get('text', b.get('content', ''))
    if typ in ('p','paragraph','text','callout','note','definition','example','warning'):
        out = [p(text)]
        if typ in ('callout','note','definition','example','warning'):
            tb = Table([[out[0]]], colWidths=[W-M_L-M_R-8*mm])
            bg = {'warning':'#FFF7E6','definition':'#EEF6FA','example':'#F1F8F3'}.get(typ,'#F3F6F9')
            tb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor(bg)),('BOX',(0,0),(-1,-1),.5,colors.HexColor('#CBD5E1')),('LEFTPADDING',(0,0),(-1,-1),3*mm),('RIGHTPADDING',(0,0),(-1,-1),3*mm),('TOPPADDING',(0,0),(-1,-1),2*mm),('BOTTOMPADDING',(0,0),(-1,-1),1*mm)]))
            out = [tb, Spacer(1,2*mm)]
        return out
    if typ in ('h1','h2','h3','heading','subheading'):
        return [p(text, {'h1':'H1x','heading':'H2x','h2':'H2x','h3':'H3x','subheading':'H3x'}[typ])]
    if typ in ('equation','eq','formula'):
        latex = b.get('latex', text)
        pic = equation_image(latex, eqcount)
        if pic:
            content = image_block(pic, W-M_L-M_R-12*mm, 24*mm)
        else:
            content = p(latex, 'Smallx')
        out = [Spacer(1,1.2*mm), content]
        caption = b.get('caption')
        if caption: out.append(p(caption, 'EqCaptionx'))
        return out + [Spacer(1,1*mm)]
    if typ in ('figure','image','plot','diagram'):
        path = b.get('path', b.get('src', ''))
        out = [image_block(path, W-M_L-M_R-6*mm, float(b.get('max_height_mm', 105))*mm)]
        if b.get('caption'): out.append(p(b['caption'], 'Captionx'))
        return out
    if typ == 'table':
        headers = b.get('headers', [])
        rows = b.get('rows', [])
        raw = ([headers] if headers else []) + rows
        cells = [[p(str(c), 'Smallx') for c in row] for row in raw]
        if not cells: return []
        ncol = max(map(len,cells))
        cw = (W-M_L-M_R)/ncol
        tab = Table(cells, colWidths=[cw]*ncol, repeatRows=1 if headers else 0, hAlign='CENTER')
        cmds = [('GRID',(0,0),(-1,-1),.4,colors.HexColor('#CBD5E1')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('ROWBACKGROUNDS',(0,1 if headers else 0),(-1,-1),[colors.white,colors.HexColor('#F7F9FB')])]
        if headers: cmds += [('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7EEF4')),('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#102A43'))]
        tab.setStyle(TableStyle(cmds)); return [tab, Spacer(1,3*mm)]
    if typ in ('code','listing','pseudocode'):
        return [Preformatted(str(text), styles['Codex'])]
    if typ in ('pagebreak','page_break','break'):
        return [PageBreak()]
    if typ == 'spacer': return [Spacer(1, float(b.get('height_mm',3))*mm)]
    return [p(text)]

def build(data):
    title = data.get('title', 'Многомасштабный SVD-SNO')
    subtitle = data.get('subtitle', 'Подробное учебное руководство по нейросетевым операторам и многомасштабным методам')
    pages = data.get('pages', data.get('chapters', []))
    story = []
    # Cover page
    story += [Spacer(1,40*mm), HRFlowable(width='100%', thickness=2, color=colors.HexColor('#176B87')), Spacer(1,12*mm), p(title, 'CoverTitle'), p(subtitle, 'CoverSub')]
    if data.get('description'): story.append(p(data['description'], 'Bodyx'))
    story += [Spacer(1,17*mm), p('Учебное руководство • математические основы • архитектуры • алгоритмы • научные примеры', 'Smallx'), PageBreak()]
    # Contents generated from chapter headings
    story += [p('Содержание', 'H1x')]
    for i, chapter in enumerate(pages,1):
        ctitle = chapter.get('title', f'Раздел {i}') if isinstance(chapter,dict) else f'Раздел {i}'
        story.append(p(f'{i:02d}  {ctitle}', 'TOCx'))
    story.append(PageBreak())
    eqcount=0
    for i, chapter in enumerate(pages,1):
        if not isinstance(chapter,dict): chapter={'title':str(chapter),'blocks':[]}
        # Let short sections share pages; isolate the required final comparison note.
        if i == len(pages): story.append(PageBreak())
        story.append(p(chapter.get('title', f'Раздел {i}'), 'H1x'))
        blocks = chapter.get('blocks', [])
        for block in blocks:
            if isinstance(block,dict) and str(block.get('type','')).lower() in ('equation','eq','formula'):
                eqcount += 1
            story.extend(render_block(block,eqcount))
    doc = ManualDoc(str(OUTPUT))
    doc.build(story)

if __name__ == '__main__':
    if not INPUT.exists():
        raise SystemExit(f'Input JSON not found: {INPUT}')
    data = json.loads(INPUT.read_text(encoding='utf-8'))
    build(data)
    print(OUTPUT)
