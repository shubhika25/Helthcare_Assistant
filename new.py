from pinecone import Pinecone

pc = Pinecone(api_key="pcsk_2SZmuv_CgZ7WHxy576vkw5LBGGMAPtH6sep3zF2zYCwdoe1jr2BdcKuuVWi4RiXB1PsD92")  # your key here
print(pc.list_indexes())
