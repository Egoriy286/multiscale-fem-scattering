"""Structural/text QA for the final manual; visual review is still required."""
from pathlib import Path
import hashlib
import json
import re

import pdfplumber
from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
PDF=HERE/'output/pdf/svd_sno_student_manual.pdf'


def main():
    reader=PdfReader(PDF)
    pages=[]
    outside=[]
    replacements=[]
    with pdfplumber.open(PDF) as doc:
        for i,page in enumerate(doc.pages,1):
            text=page.extract_text(x_tolerance=1) or ''
            if '\ufffd' in text:replacements.append(i)
            for char in page.chars:
                if (char['x0'] < -.5 or char['x1'] > page.width+.5
                        or char['top'] < -.5 or char['bottom'] > page.height+.5):
                    outside.append(dict(page=i,text=char['text'],x=char['x0'],top=char['top']))
            body=[c for c in page.chars if 64<c['top']<790]
            pages.append(dict(pdf_page=i,characters=len(text),
                              body_bottom=max((c['bottom'] for c in body),default=0),
                              body_top=min((c['top'] for c in body),default=0)))
    log=(HERE/'tmp/latex/main.log').read_text(errors='replace')
    flags=[line for line in log.splitlines() if re.search(
        r'Overfull|Missing character|LaTeX Error|undefined|multiply defined|Rerun to',line)]
    assert not outside,outside[:5]
    assert not replacements,replacements
    assert not flags,flags
    assert all(p['characters']>80 for p in pages),pages
    assert len(reader.outline)>0
    links=sum(len(p.get('/Annots',[]).get_object()) if hasattr(p.get('/Annots',[]),'get_object')
              else len(p.get('/Annots',[])) for p in reader.pages)
    result=dict(pdf=str(PDF.name),sha256=hashlib.sha256(PDF.read_bytes()).hexdigest(),
                page_count=len(reader.pages),a4=all(abs(float(p.mediabox.width)-595.28)<1
                    and abs(float(p.mediabox.height)-841.89)<1 for p in reader.pages),
                annotation_count=links,
                outline_top_level_entries=sum(isinstance(item,dict) for item in reader.outline),
                tex_critical_messages=flags,off_page_characters=outside,
                replacement_character_pages=replacements,pages=pages)
    (HERE/'assets/pdf_qa.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='pages'},indent=2,ensure_ascii=False))


if __name__=='__main__':main()
