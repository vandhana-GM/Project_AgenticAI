from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_path):
    loader = PyPDFLoader("C:\\Users\\HP\\Downloads\\Project_Agentic_AI\\pdf_rag_system\\data\\Machine_Learning.pdf")

    documents = loader.load()

    return documents