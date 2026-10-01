def create_chunks(text, chunk_size=50, overlap=10):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


text = """
Python is a high-level programming language.
Python is widely used for data analysis and artificial intelligence.
Python lists are ordered and mutable collections.
Python dictionaries store data in key-value pairs.
Python functions are reusable blocks of code.
"""

chunks = create_chunks(text, chunk_size=10, overlap=3)

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print("\nChunk", i + 1)
    print(chunk)