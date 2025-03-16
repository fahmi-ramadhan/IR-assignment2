from bsbi import BSBIIndex
from compression import VBEPostings, Simple8bPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_path = 'arxiv_collections', \
                          postings_encoding = VBEPostings, \
                          output_path = 'index_vb')

BSBI_instance_simple8b = BSBIIndex(data_path = 'arxiv_collections', \
                          postings_encoding = Simple8bPostings, \
                          output_path = 'index_simple8b')

queries = ["(cosmological AND (quantum OR continuum)) AND geodesics"]
for query in queries:
    print("Query  : ", query)
    print("Results:")
    for doc in BSBI_instance.boolean_retrieve(query):
        print(doc)
    print()


for query in queries:
    print("Query  : ", query)
    print("Results:")
    for doc in BSBI_instance_simple8b.boolean_retrieve(query):
        print(doc)
    print()
