import array

class StandardPostings:
    """ 
    Class dengan static methods, untuk mengubah representasi postings list
    yang awalnya adalah List of integer, berubah menjadi sequence of bytes.
    Kita menggunakan Library array di Python.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    Silakan pelajari:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        # Untuk yang standard, gunakan L untuk unsigned long, karena docID
        # tidak akan negatif. Dan kita asumsikan docID yang paling besar
        # cukup ditampung di representasi 4 byte unsigned.
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()


class VBEPostings:
    """ 
    Berbeda dengan StandardPostings, dimana untuk suatu postings list,
    yang disimpan di disk adalah sequence of integers asli dari postings
    list tersebut apa adanya.

    Pada VBEPostings, kali ini, yang disimpan adalah gap-nya, kecuali
    posting yang pertama. Barulah setelah itu di-encode dengan Variable-Byte
    Enconding algorithm ke bytestream.

    Contoh:
    postings list [34, 67, 89, 454] akan diubah dulu menjadi gap-based,
    yaitu [34, 33, 22, 365]. Barulah setelah itu di-encode dengan algoritma
    compression Variable-Byte Encoding, dan kemudian diubah ke bytesream.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes (dengan Variable-Byte
        Encoding). JANGAN LUPA diubah dulu ke gap-based list, sebelum
        di-encode dan diubah ke bytearray.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        gap_based_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_based_list.append(postings_list[i] - postings_list[i-1])
        return VBEPostings.vb_encode(gap_based_list)
    
    @staticmethod
    def vb_encode(list_of_numbers):
        """ 
        Melakukan encoding (tentunya dengan compression) terhadap
        list of numbers, dengan Variable-Byte Encoding
        """
        bytestream = bytearray()
        for number in list_of_numbers:
            bytestream.extend(VBEPostings.vb_encode_number(number))
        return bytestream

    @staticmethod
    def vb_encode_number(number):
        """
        Encodes a number using Variable-Byte Encoding
        Lihat buku teks kita!
        """
        bytes = []
        while True:
            bytes.insert(0, number & 0x7F) # number % 128
            if number < 128:
                break
            number = number >> 7 # number // 128
        bytes[-1] |= 0x80 # set MSB of the last byte
        return bytes

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes. JANGAN LUPA
        bytestream yang di-decode dari encoded_postings_list masih berupa
        gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        gap_based_list = VBEPostings.vb_decode(encoded_postings_list)
        postings_list = [gap_based_list[0]]
        for i in range(1, len(gap_based_list)):
            postings_list.append(postings_list[i-1] + gap_based_list[i])
        return postings_list

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decoding sebuah bytestream yang sebelumnya di-encode dengan
        variable-byte encoding.
        """
        numbers = []
        n = 0
        for byte in encoded_bytestream:
            if (byte < 128):
                n = 128 * n + byte
            else:
                n = 128 * n + byte - 128
                numbers.append(n)
                n = 0
        return numbers

class Simple8bPostings:
    """
    Menerapkan kompresi Simple-8b untuk encoding dan decoding posting list dalam bentuk gap list.
    Tidak seperti VBE, Simple-8b menggunakan bit-level encoding dan meng-encode integers ke dalam 64-bit.

    """
    SELECTOR_TABLE = [  # Format: (bits_per_integer, integers_coded)
        (0,240),(0,120),(1,60),(2,30),(3,20),(4,15),(5,12),(6,10),
        (7,8),(8,7),(10,6),(12,5),(15,4),(20,3),(30,2),(60,1)
    ]
    
    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes menggunakan Simple-8b

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        gap_based_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_based_list.append(postings_list[i] - postings_list[i-1])
        return Simple8bPostings.simple8b_encode(gap_based_list)
    
    @staticmethod
    def simple8b_encode(list_of_numbers):
        """
        Encode list of numbers menggunakan algoritma Simple-8b.
        """
        bytestream = bytearray()
        i = 0
        while i < len(list_of_numbers):
            selector = Simple8bPostings.find_selector(list_of_numbers[i:])
            bits_per_integer, integers_coded = Simple8bPostings.SELECTOR_TABLE[selector]
            if selector == 0:
                bytestream.extend((selector).to_bytes(8, byteorder='big'))
                i += 240
            elif selector == 1:
                bytestream.extend((selector).to_bytes(8, byteorder='big'))
                i += 120
            else:
                encoded = selector
                for j in range(integers_coded):
                    encoded |= (list_of_numbers[i + j] << (4 + bits_per_integer * j))
                bytestream.extend(encoded.to_bytes(8, byteorder='big'))
                i += integers_coded
        return bytestream

    @staticmethod
    def find_selector(numbers):
        """
        Mencari selector yang paling sesuai untuk mengompresi list of numbers.
        """
        # Gunakan selector 0 atau 1 untuk handle runs of 1's. (sumber: paper Simple-8b hal. 137)
        if len(numbers) >= 240 and all(num == 1 for num in numbers[:240]):
            return 0
        if len(numbers) >= 120 and all(num == 1 for num in numbers[:120]):
            return 1
        # Cek selector lainnya
        for selector in range(2, 16):
            bits_per_integer, integers_coded = Simple8bPostings.SELECTOR_TABLE[selector]
            if len(numbers) >= integers_coded and max(numbers[:integers_coded]) < (1 << bits_per_integer):
                return selector
        # Jika nilai terlalu besar, throw error
        raise ValueError("No suitable selector found for Simple-8b encoding, value too large")

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes. 

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        gap_based_list = Simple8bPostings.simple8b_decode(encoded_postings_list)
        postings_list = [gap_based_list[0]]
        for i in range(1, len(gap_based_list)):
            postings_list.append(postings_list[i-1] + gap_based_list[i])
        return postings_list
    
    @staticmethod
    def simple8b_decode(encoded_bytestream):
        """
        Decode bytestream yang sebelumnya di-encode dengan Simple-8b.
        """
        result = []
        # Proses bytestream per 8 bytes (64 bits / 1 block pada Simple-8b)
        for i in range(0, len(encoded_bytestream), 8):
            block = int.from_bytes(encoded_bytestream[i:i+8], byteorder='big')
            selector = block & 0xF
            if selector == 0:
                result.extend([1] * 240)
            elif selector == 1:
                result.extend([1] * 120)
            else:
                bits_per_integer, integers_coded = Simple8bPostings.SELECTOR_TABLE[selector]
                for j in range(integers_coded):
                    result.append((block >> (4 + bits_per_integer * j)) & ((1 << bits_per_integer) - 1))
        return result

if __name__ == '__main__':
    
    postings_list = [34, 67, 89, 454, 2345738]
    for Postings in [StandardPostings, VBEPostings, Simple8bPostings]:
        # Silakan sesuaikan jika ada perbedaan parameter pada metode encode dan decode Simple8bPostings
        print(Postings.__name__)
        encoded_postings_list = Postings.encode(postings_list)
        print("byte hasil encode: ", encoded_postings_list)
        print("ukuran encoded postings: ", len(encoded_postings_list), "bytes")
        decoded_posting_list = Postings.decode(encoded_postings_list)
        print("hasil decoding: ", decoded_posting_list)
        assert decoded_posting_list == postings_list, "hasil decoding tidak sama dengan postings original"
        print()
