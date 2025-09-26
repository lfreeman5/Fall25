import os
from reportlab.pdfgen import canvas
from PIL import Image

folder = "./"
pngs = sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])

for f in pngs:
    img = Image.open(os.path.join(folder, f))
    w, h = img.size
    pdf_path = os.path.join(folder, f.rsplit(".",1)[0] + ".pdf")
    c = canvas.Canvas(pdf_path, pagesize=(w, h))
    c.drawInlineImage(os.path.join(folder, f), 0, 0, width=w, height=h)
    c.showPage()
    c.save()
