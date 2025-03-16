class IdMap:
    """
    Ingat kembali di kuliah, bahwa secara praktis, sebuah dokumen dan
    sebuah term akan direpresentasikan sebagai sebuah integer. Oleh
    karena itu, kita perlu maintain mapping antara string term (atau
    dokumen) ke integer yang bersesuaian, dan sebaliknya. Kelas IdMap ini
    akan melakukan hal tersebut.
    """

    def __init__(self):
        """
        Mapping dari string (term atau nama dokumen) ke id disimpan dalam
        python's dictionary; cukup efisien. Mapping sebaliknya disimpan dalam
        python's list.

        contoh:
            str_to_id["halo"] ---> 8
            str_to_id["/collection/dir0/gamma.txt"] ---> 54

            id_to_str[8] ---> "halo"
            id_to_str[54] ---> "/collection/dir0/gamma.txt"
        """
        self.str_to_id = {}
        self.id_to_str = []

    def __len__(self):
        """Mengembalikan banyaknya term (atau dokumen) yang disimpan di IdMap."""
        # TODO
        return 0

    def __get_id(self, s):
        """
        Mengembalikan integer id i yang berkorespondensi dengan sebuah string s.
        Jika s tidak ada pada IdMap, lalu assign sebuah integer id baru dan kembalikan
        integer id baru tersebut.
        """
        # TODO
        return 0
    
    def __get_str(self, i):
        """Mengembalikan string yang terasosiasi dengan index i."""
        # TODO
        return ""

    def __getitem__(self, key):
        """
        __getitem__(...) adalah special method di Python, yang mengizinkan sebuah
        collection class (seperti IdMap ini) mempunyai mekanisme akses atau
        modifikasi elemen dengan syntax [..] seperti pada list dan dictionary di Python.

        Silakan search informasi ini di Web search engine favorit Anda. Saya mendapatkan
        link berikut:

        https://stackoverflow.com/questions/43627405/understanding-getitem-method

        """
        # TODO
        return None

class QueryParser:
    """
    Class untuk melakukan parsing query untuk boolean search
    
    Parameters
    ----------
    query: str
        Query string yang akan di-parse. Input dijamin valid, tidak ada imbalanced parentheses.
        Tanda kurung buka dijamin "menempel" di awal kata yang mengikuti (atau tanda kurung buka lainnya) dan
        tanda kurung tutup dijamin "menempel" di akhir kata yang diikuti (atau tanda kurung tutup lainnya).
        Sebagai contoh, bisa lihat pada contoh method query_string_to_list() atau pada test case.
    stemmer
        Objek stemmer untuk stemming token
    stopwords: set
        Set yang berisi stopwords
    """
    def __init__(self, query: str, stemmer, stopwords: set):
        self.query = query
        self.stemmer = stemmer
        self.stopwords = stopwords
        self.token_list = self.__query_string_to_list()
        self.token_preprocessed = self.__preprocess_tokens()
    
    def is_valid(self):
        """
        Gunakan method ini untuk validasi query saat melakukan boolean retrieval,
        untuk menentukan apakah suatu query valid (tidak mengandung stopwords) atau tidak.
        """
        for token in self.token_list:
            if token in self.stopwords:
                return False
        return True

    def __query_string_to_list(self):
        """
        Melakukan parsing query dari yang berbentuk string menjadi list of tokens.
        Contoh: "term1 AND term2 OR (term3 DIFF term4)" --> ["term1", "AND", "term2", "OR", "(",
                                                             "term3", "DIFF", "term4", ")"]

        Returns
        -------
        List[str]
            query yang sudah di-parse
        """   
        # TODO
        return []

    def __preprocess_tokens(self):
        """
        Melakukan pre-processing pada query input, cukup lakukan stemming saja.
        Asumsikan bahwa tidak ada stopwords yang diberikan pada query input.
        Jangan lakukan pre-processing pada token spesial ('AND', 'OR', 'DIFF', '(', ')')
        
        Returns
        -------
        List[str]
            Daftar token yang telah di-preprocess
        """
        # TODO
        return []

    def infix_to_postfix(self):
        """
        Fungsi ini mengubah ekspresi infix menjadi postfix dengan menggunakan Algoritma Shunting-Yard. 
        Evaluasi akan dilakukan secara postfix juga. Gunakan tokens yang sudah di-pre-processed.
        Contoh: "A AND B" (infix) --> ["A", "B", "AND"] (postfix)
        Untuk selengkapnya, silakan lihat algoritma berikut: 
        https://www.geeksforgeeks.org/convert-infix-expression-to-postfix-expression/

        Returns
        -------
        list[str]
            list yang berisi token dalam ekspresi postfix
        """
        # TODO
        return []

def sort_intersect_list(list_A, list_B):
    """
    Intersects two (ascending) sorted lists and returns the sorted result
    Melakukan Intersection dua (ascending) sorted lists dan mengembalikan hasilnya
    yang juga terurut.

    Parameters
    ----------
    list_A: List[Comparable]
    list_B: List[Comparable]
        Dua buah sorted list yang akan di-intersect.

    Returns
    -------
    List[Comparable]
        intersection yang sudah terurut
    """
    # TODO
    return []

def sort_union_list(list_A, list_B):
    """
    Melakukan union dua (ascending) sorted lists dan mengembalikan hasilnya
    yang juga terurut.

    Parameters
    ----------
    list_A: List[Comparable]
    list_B: List[Comparable]
        Dua buah sorted list yang akan di-union.

    Returns
    -------
    List[Comparable]
        union yang sudah terurut
    """
    # TODO
    return []

def sort_diff_list(list_A, list_B):
    """
    Melakukan difference dua (ascending) sorted lists dan mengembalikan hasilnya
    yang juga terurut.

    Parameters
    ----------
    list_A: List[Comparable]
    list_B: List[Comparable]
        Dua buah sorted list yang akan di-difference.

    Returns
    -------
    List[Comparable]
        difference yang sudah terurut
    """
    # TODO
    return []

if __name__ == '__main__':

    doc = ["halo", "semua", "selamat", "pagi", "semua"]
    term_id_map = IdMap()
    assert [term_id_map[term] for term in doc] == [0, 1, 2, 3, 1], "term_id salah"
    assert term_id_map[1] == "semua", "term_id salah"
    assert term_id_map[0] == "halo", "term_id salah"
    assert term_id_map["selamat"] == 2, "term_id salah"
    assert term_id_map["pagi"] == 3, "term_id salah"

    docs = ["/collection/0/data0.txt",
            "/collection/0/data10.txt",
            "/collection/1/data53.txt"]
    doc_id_map = IdMap()
    assert [doc_id_map[docname] for docname in docs] == [0, 1, 2], "docs_id salah"
    
    assert sort_intersect_list([1, 2, 3], [2, 3]) == [2, 3], "sorted_intersect salah"
    assert sort_intersect_list([4, 5], [1, 4, 7]) == [4], "sorted_intersect salah"
    assert sort_intersect_list([], []) == [], "sorted_intersect salah"

    assert sort_union_list([1, 2, 3], [2, 3]) == [1, 2, 3], "sorted_union salah"
    assert sort_union_list([4, 5], [1, 4, 7]) == [1, 4, 5, 7], "sorted_union salah"
    assert sort_union_list([], []) == [], "sorted_union salah"

    assert sort_diff_list([1, 2, 3], [2, 3]) == [1], "sorted_diff salah"
    assert sort_diff_list([4, 5], [1, 4, 7]) == [5], "sorted_diff salah"
    assert sort_diff_list([], []) == [], "sorted_diff salah"

    from porter2stemmer import Porter2Stemmer
    qp = QueryParser("((term1 AND term2) OR term3) DIFF (term6 AND (term4 OR term5) DIFF (term7 OR term8))", 
                     Porter2Stemmer(), set())
    assert qp.token_list == ['(', '(', 'term1', 'AND', 'term2', ')', 'OR', 'term3', ')', 
                                     'DIFF', '(', 'term6', 'AND', '(', 'term4', 'OR', 'term5', 
                                     ')', 'DIFF', '(', 'term7', 'OR', 'term8', ')', ')'], "parsing to list salah"
    assert qp.infix_to_postfix() == ['term1', 'term2', 'AND', 'term3', 'OR', 'term6', 'term4', 
                                     'term5', 'OR', 'AND', 'term7', 'term8', 'OR', 'DIFF', 'DIFF'], "postfix salah"
    
    qp1 = QueryParser("term1 OR ((term2 AND term3) DIFF (term4 OR term5))", Porter2Stemmer(), set())
    assert qp1.token_list == ['term1', 'OR', '(', '(', 'term2', 'AND', 'term3', ')', 'DIFF', 
                              '(', 'term4', 'OR', 'term5', ')', ')'], "parsing to list salah"
    assert qp1.infix_to_postfix() == ['term1', 'term2', 'term3', 'AND', 'term4', 'term5', 'OR', 
                                      'DIFF', 'OR'], "postfix salah"