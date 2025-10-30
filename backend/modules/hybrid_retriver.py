import os
import re
import requests
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchResults

# ✅ Load .env file
load_dotenv()
print(">>> [DEBUG] hybrid_retriever module loaded")

# ====================================
# 1️⃣ PubMed Retriever
# ====================================
class PubMedRetriever:
    def __init__(self, max_results=5):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.max_results = max_results

    def retrieve(self, query: str) -> List[Document]:
        """Fetch abstracts from PubMed for medical context."""
        try:
            search_url = f"{self.base_url}esearch.fcgi?db=pubmed&term={query}&retmax={self.max_results}&retmode=json"
            search_res = requests.get(search_url).json()
            ids = search_res.get("esearchresult", {}).get("idlist", [])

            docs = []
            for pmid in ids:
                fetch_url = f"{self.base_url}efetch.fcgi?db=pubmed&id={pmid}&retmode=text&rettype=abstract"
                abstract_text = requests.get(fetch_url).text.strip()
                if abstract_text:
                    docs.append(
                        Document(
                            page_content=abstract_text,
                            metadata={"source": f"PubMed PMID:{pmid}", "weight": 1.0}
                        )
                    )
            return docs
        except Exception as e:
            print(f"❌ [PubMedRetriever] Error: {e}")
            return []

# ====================================
# 2️⃣ Trusted Web Retriever (DuckDuckGo)
# ====================================
class TrustedWebRetriever:
    def __init__(self, max_results=5):
        self.search = DuckDuckGoSearchResults(max_results=max_results)
        self.trusted_domains = [
            "pubmed.ncbi.nlm.nih.gov", "www.who.int", "www.cdc.gov", "www.nih.gov",
            "jamanetwork.com", "www.thelancet.com", "www.mayoclinic.org", "medlineplus.gov"
        ]

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve results only from trusted health domains."""
        try:
            results_text = self.search.run(query)
            docs = []
            for domain in self.trusted_domains:
                if domain in results_text:
                    docs.append(
                        Document(
                            page_content=results_text,
                            metadata={"source": f"Web ({domain})", "weight": 0.6}
                        )
                    )
            return docs
        except Exception as e:
            print(f"❌ [TrustedWebRetriever] Error: {e}")
            return []

# ====================================
# 3️⃣ PDF Retriever (Pinecone v3 Compatible)
# ====================================
class PDFPineconeRetriever:
    def __init__(self, index_name: str, embedder):
        from pinecone import Pinecone

        # ✅ Hard-code API key for debugging (temporary!)
        api_key = "pcsk_2SZmuv_CgZ7WHxy576vkw5LBGGMAPtH6sep3zF2zYCwdoe1jr2BdcKuuVWi4RiXB1PsD92"

        # Initialize Pinecone
        self.pc = Pinecone(api_key=api_key)
        print("🔑 [DEBUG] Pinecone client initialized successfully.")

        # ✅ Ensure index exists
        indexes = [i["name"] for i in self.pc.list_indexes()]
        if index_name not in indexes:
            raise ValueError(f"❌ Pinecone index '{index_name}' not found. Available: {indexes}")

        self.index = self.pc.Index(index_name)
        self.embedder = embedder

    def retrieve(self, query: str) -> List[Document]:
        """Query Pinecone for uploaded PDF document chunks."""
        try:
            query_vector = self.embedder.embed_query(query)
            response = self.index.query(
                vector=query_vector,
                top_k=5,
                include_metadata=True
            )

            # ✅ Handle both dict/object formats
            matches = []
            if isinstance(response, dict):
                matches = response.get("matches", [])
            elif hasattr(response, "matches"):
                matches = response.matches

            docs = []
            for match in matches:
                metadata = getattr(match, "metadata", None) or match.get("metadata", {})
                content = metadata.get("text", "")
                if content:
                    docs.append(
                        Document(
                            page_content=content,
                            metadata={
                                **metadata,
                                "score": getattr(match, "score", 0.0),
                                "source": "Uploaded PDF",
                                "weight": 0.9
                            }
                        )
                    )

            print(f">>> [DEBUG] Pinecone returned {len(docs)} matching chunks.")
            return docs
        except Exception as e:
            print(f"❌ [PDFPineconeRetriever] Error: {e}")
            return []

# ====================================
# 4️⃣ Hybrid Retriever (Intelligent Orchestrator)
# ====================================
class HybridRetriever:
    def __init__(self, index_name: str, embedder):
        self.pubmed = PubMedRetriever()
        self.web = TrustedWebRetriever()
        self.pdf = PDFPineconeRetriever(index_name=index_name, embedder=embedder)

        # 🩺 Keywords that indicate patient-specific/lab queries
        self.lab_keywords = [
            r"\bmy\b", r"\breport\b", r"\blab\b", r"\bblood\b",
            r"\bhemoglobin\b", r"\bglucose\b", r"\bcholesterol\b",
            r"\bhdl\b", r"\bldl\b", r"\bcreatinine\b", r"\btest\b",
            r"\bresults\b", r"\bvalue\b", r"\blevel\b", r"\btests\b"
        ]

    def _is_patient_query(self, query: str) -> bool:
        """Detect if user query relates to personal/lab data."""
        pattern = "|".join(self.lab_keywords)
        return bool(re.search(pattern, query.lower()))

    def retrieve(self, query: str) -> List[Document]:
        """Hybrid retrieval: intelligently decide source based on query type."""
        try:
            if self._is_patient_query(query):
                print(">>> [DEBUG] Detected patient-specific query → prioritizing PDF (lab reports).")
                pdf_docs = self.pdf.retrieve(query)
                pubmed_docs = self.pubmed.retrieve(query)
                all_docs = pdf_docs + pubmed_docs
            else:
                print(">>> [DEBUG] General medical query → using PubMed + Web + PDF.")
                pubmed_docs = self.pubmed.retrieve(query)
                web_docs = self.web.retrieve(query)
                pdf_docs = self.pdf.retrieve(query)
                all_docs = pubmed_docs + pdf_docs + web_docs

            # ✅ Weighted sorting
            all_docs.sort(key=lambda d: d.metadata.get("weight", 0), reverse=True)

            print(f">>> [DEBUG] Total retrieved documents: {len(all_docs)}")
            return all_docs
        except Exception as e:
            print(f"❌ [HybridRetriever] Error: {e}")
            return []
