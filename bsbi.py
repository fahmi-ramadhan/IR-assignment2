import os
import pickle
import contextlib
import heapq
import time
import re
from nltk.corpus import stopwords
from porter2stemmer import Porter2Stemmer

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, QueryParser, sort_diff_list, sort_intersect_list, sort_union_list
from compression import VBEPostings, Simple8bPostings

""" 
Ingat untuk install tqdm terlebih dahulu
pip intall tqdm
"""
from tqdm import tqdm

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_path(str): Path ke data
    output_path(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_path, output_path, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_path = data_path
        self.output_path = output_path
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

        # Untuk menyimpan statistik waktu indexing
        self.timing_stats = {
            'parsing_blocks': 0.0,
            'writing_indices': 0.0,
            'merging_indices': 0.0,
            'total': 0.0
        }

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_path, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_path, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_path, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_path, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def start_indexing(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        start_time = time.time()
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_path in tqdm(sorted(next(os.walk(self.data_path))[1])):
            block_start_time = time.time()
            td_pairs = self.parsing_block(block_path)
            block_parsing_time = time.time() - block_start_time
            self.timing_stats['parsing_blocks'] += block_parsing_time

            index_id = 'intermediate_index_'+block_path
            self.intermediate_indices.append(index_id)

            write_start_time = time.time()
            with InvertedIndexWriter(index_id, self.postings_encoding, path = self.output_path) as index:
                self.write_to_index(td_pairs, index)
                td_pairs = None
            write_time = time.time() - write_start_time
            self.timing_stats['writing_indices'] += write_time
    
        self.save()

        print("Starting index merging...")
        merge_start_time = time.time()
        with InvertedIndexWriter(self.index_name, self.postings_encoding, path = self.output_path) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, path=self.output_path))
                               for index_id in self.intermediate_indices]
                self.merge_index(indices, merged_index)
        merge_time = time.time() - merge_start_time
        self.timing_stats['merging_indices'] = merge_time
        print(f"Index merging completed in {merge_time:.2f}s")

        total_time = time.time() - start_time
        self.timing_stats['total'] = total_time

        # Update timing stats file
        self.save()
        
        # Print summary of timing statistics
        print("\nIndexing completed!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Parsing blocks: {self.timing_stats['parsing_blocks']:.2f}s ({self.timing_stats['parsing_blocks']/total_time*100:.1f}%)")
        print(f"Writing indices: {self.timing_stats['writing_indices']:.2f}s ({self.timing_stats['writing_indices']/total_time*100:.1f}%)")
        print(f"Merging indices: {self.timing_stats['merging_indices']:.2f}s ({self.timing_stats['merging_indices']/total_time*100:.1f}%)")

    def parsing_block(self, block_path):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Anda bisa menggunakan stemmer bahasa Inggris yang tersedia, seperti Porter Stemmer
        https://github.com/evandempsey/porter2-stemmer

        Untuk membuang stopwords, Anda dapat menggunakan library seperti NLTK.

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_path : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus persis untuk semua pemanggilan
        parse_block(...).
        """
        td_pairs = []
        full_block_path = os.path.join(self.data_path, block_path)
        stopword_set = set(stopwords.words('english'))
        stemmer = Porter2Stemmer()
        for doc_name in tqdm(sorted(next(os.walk(full_block_path))[2])):
            file_path = os.path.join(full_block_path, doc_name)
            doc_id = self.doc_id_map[doc_name]
            with open(file_path, 'r') as f:
                content = f.read()
                tokens = re.findall(r'\w+', content.lower())
                for token in tokens:
                    if token not in stopword_set:
                        stemmed = stemmer.stem(token)
                        term_id = self.term_id_map[stemmed]
                        td_pairs.append((term_id, doc_id))
        return td_pairs

    def write_to_index(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
            term_dict[term_id].add(doc_id)
        for term_id in sorted(term_dict.keys()):
            index.append(term_id, sorted(list(term_dict[term_id])))

    def merge_index(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        min_heap = []
        # Inisialisasi min heap dengan satu entry (term id, index id, postings list) pada setiap index
        for i, index in enumerate(indices):
            try:
                term_id, postings_list = next(index)
                heapq.heappush(min_heap, (term_id, i, postings_list))
            except StopIteration: # Skip index yang kosong
                pass
        
        current_term_id, current_postings_list = -1, []
        # Proses merging
        while min_heap:
            # Ambil entry dengan term_id terkecil
            term_id, i, postings_list = heapq.heappop(min_heap)
            # Kalau sudah selesai menggabungkan postings list dari term_id sebelumnya, tulis ke merged index
            if current_term_id != -1 and term_id != current_term_id:
                merged_index.append(current_term_id, current_postings_list)
                current_postings_list = []
            current_term_id = term_id
            # Gabungkan postings list dari semua index yang memiliki term_id yang sama
            if not current_postings_list:
                current_postings_list = postings_list
            else:
                current_postings_list = sort_union_list(current_postings_list, postings_list)
            # Ambil entry selanjutnya dari index yang sama
            try:
                next_term_id, next_postings_list = next(indices[i])
                heapq.heappush(min_heap, (next_term_id, i, next_postings_list))
            except StopIteration: # Skip index yang kosong
                pass
        
        # Tulis sisa postings list yang belum ditulis
        if current_postings_list:
            merged_index.append(current_term_id, current_postings_list)

    def boolean_retrieve(self, query):
        """
        Melakukan boolean retrieval untuk mengambil semua dokumen yang
        mengandung semua kata pada query. Lakukan pre-processing seperti
        yang telah dilakukan pada tahap indexing, kecuali *penghapusan stopwords*.

        Jika terdapat stopwords dalam query, return list kosong dan berikan pesan bahwa
        terdapat stopwords di dalam query.

        Parse query dengan class QueryParser. Ambil representasi postfix dari ekspresi
        untuk kemudian dievaluasi di method ini. Silakan baca pada URL di bawah untuk lebih lanjut.
        https://www.geeksforgeeks.org/evaluation-of-postfix-expression/

        Anda tidak wajib mengimplementasikan conjunctive queries optimization.

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi. Ini dapat mengandung operator
            himpunan AND, NOT, dan DIFF, serta tanda kurung untuk presedensi. 

            contoh: (universitas AND indonesia OR depok) DIFF ilmu AND komputer

        Returns
        ------
        List[str]
            Daftar dokumen terurut yang mengandung sebuah query tokens.
            Harus mengembalikan EMPTY LIST [] jika tidak ada yang match.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.
        """
        query_parser = QueryParser(query, Porter2Stemmer(), set(stopwords.words('english')))
        if not query_parser.is_valid():
            return []
        postfix_query = query_parser.infix_to_postfix()
        
        stack = []
        for token in postfix_query:
            if token not in query_parser.special_tokens:
                term_id = self.term_id_map[token] if token in self.term_id_map else -1
                if term_id == -1: # Term tidak ada di collection
                    stack.append([])
                else:
                    try:
                        with InvertedIndexReader(self.index_name, self.postings_encoding, path=self.output_path) as index:
                            postings_list = index.get_postings_list(term_id)
                            stack.append(postings_list)
                    except: # Term ada tapi tidak ada postings list-nya
                        stack.append([])
            else:
                if token == 'AND':
                    stack.append(sort_intersect_list(stack.pop(), stack.pop()))
                elif token == 'OR':
                    stack.append(sort_union_list(stack.pop(), stack.pop()))
                elif token == 'DIFF':
                    stack.append(sort_diff_list(stack.pop(), stack.pop()))

        if stack:
            result_doc_ids = stack.pop()
            return [self.doc_id_map[doc_id] for doc_id in result_doc_ids]
        else:
            return []


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_path = 'arxiv_collections', \
                              postings_encoding = VBEPostings, \
                              output_path = 'index_vb')
    BSBI_instance.start_indexing() # memulai indexing!


    BSBI_instance_simple8b = BSBIIndex(data_path = 'arxiv_collections', \
                              postings_encoding = Simple8bPostings, \
                              output_path = 'index_simple8b')
    BSBI_instance_simple8b.start_indexing() # memulai indexing!

