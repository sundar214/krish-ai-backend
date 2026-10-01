from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Pandas is an open-source Python library for data analysis.
It provides data structures and tools for working with data.

A DataFrame is a two-dimensional data structure.
It contains rows and columns.

A Series is a one-dimensional data structure.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)

print("\nNUMBER OF CHUNKS:", len(chunks))

for i, chunk in enumerate(chunks):
    print("\n-------------------------")
    print("CHUNK", i + 1)
    print("-------------------------")
    print(chunk)