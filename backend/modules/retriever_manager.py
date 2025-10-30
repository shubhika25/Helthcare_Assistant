from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from typing import List
import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

class TrustedWebRetriever:
    def __init__(self, max_results=5):
        self.search = DuckDuckGoSearchResults(max_results=max_results)
        self.trusted_domains = [
            "pubmed.ncbi.nlm.nih.gov",
            "www.who.int",
            "www.cdc.gov",
            "www.nih.gov",
            "jamanetwork.com",
            "www.thelancet.com",
            "www.mayoclinic.org",
            "medlineplus.gov"
        ]

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve only from trusted domains"""
        results_text = self.search.run(query)
        docs = []
        for domain in self.trusted_domains:
            if domain in results_text:
                docs.append(Document(
                    page_content=results_text,
                    metadata={"source": f"Web ({domain})", "trust_score": 10}
                ))
        return docs


class PineconeRetriever:
    def __init__(self):
        PINECONE_API_KEY = "pcsk_2SZmuv_CgZ7WHxy576vkw5LBGGMAPtH6sep3zF2zYCwdoe1jr2BdcKuuVWi4RiXB1PsD92"
        self.index_name = "medicalassistantindex"
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(self.index_name)
        self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def retrieve(self, query: str, top_k=3) -> List[Document]:
        """Retrieve from uploaded Pinecone documents"""
        embedded_query = self.embed_model.embed_query(query)
        res = self.index.query(vector=embedded_query, top_k=top_k, include_metadata=True)
        docs = []
        for match in res.get("matches", []):
            text = match["metadata"].get("text", "")
            docs.append(Document(
                page_content=text,
                metadata={"source": "Pinecone", "trust_score": 8, **match["metadata"]}
            ))
        return docs


class HybridRetriever:
    def __init__(self):
        self.web_retriever = TrustedWebRetriever()
        self.vector_retriever = PineconeRetriever()

    def retrieve(self, query: str) -> List[Document]:
        """Combine both web and Pinecone sources"""
        web_docs = self.web_retriever.retrieve(query)
        local_docs = self.vector_retriever.retrieve(query)

        all_docs = web_docs + local_docs
        # Optional: sort by trust_score descending
        all_docs.sort(key=lambda d: d.metadata.get("trust_score", 0), reverse=True)
        return all_docs
