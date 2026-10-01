import pymupdf

pdf = pymupdf.open("pandas.pdf")

all_text = ""

for page in pdf:
    text = page.get_text()
    all_text += text + "\n"

pdf.close()

with open("extracted_text.txt", "w", encoding="utf-8") as file:
    file.write(all_text)

print("PDF text extracted successfully!")