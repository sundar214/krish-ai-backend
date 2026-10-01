from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Load PDF
loader = PyPDFLoader("pandas.pdf")

documents = loader.load()

print("Number of pages:", len(documents))


# 2. Create text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


# 3. Split documents
chunks = splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# 4. Display first 3 chunks
for i, chunk in enumerate(chunks[:3]):

    print("\n==============================")
    print("CHUNK", i + 1)
    print("==============================")

    print("Content:")
    print(chunk.page_content)

    print("\nMetadata:")
    print(chunk.metadata)