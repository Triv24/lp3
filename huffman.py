import heapq

class Tree:
    class Node:
        def __init__(self, char, freq):
            self.char = char
            self.freq = freq
            self.left = None
            self.right = None
    def __init__(self):
        self.root = None
        self.codes = {}

    def BuildTree(self, string):
        if not string:
            self.root = None
            self.codes = {}
            return
        freq = {}
        for char in string:
            freq[char] = freq.get(char, 0) + 1
        heap = []
        count = 0
        for char, frequency in freq.items():
            node = self.Node(char, frequency)
            heapq.heappush(heap, (frequency, count, node))
            count += 1
        while len(heap) > 1:
            freq1, _, left = heapq.heappop(heap)
            freq2, _, right = heapq.heappop(heap)
            new_node = self.Node(None, freq1 + freq2)
            new_node.left = left
            new_node.right = right
            heapq.heappush(
                heap,
                (new_node.freq, count, new_node)
            )
            count += 1
        self.root = heap[0][2]
        self.codes = {}
        self.generate_codes(self.root, "")

    def generate_codes(self, node, code):
        if node is None:
            return
        if node.char is not None:
            if code == "":
                code = "0"
            self.codes[node.char] = code
            return
        self.generate_codes(node.left, code + "0")
        self.generate_codes(node.right, code + "1")

    def encode_str(self, string):
        encoded = ""
        for char in string:
            encoded += self.codes[char]
        return encoded

    def decodestring(self, string):
        if self.root is None:
            return ""
        decoded = ""
        current = self.root
        if current.char is not None:
            return current.char * len(string)
        for bit in string:
            if bit == "0":
                current = current.left
            else:
                current = current.right
            if current.char is not None:
                decoded += current.char
                current = self.root
        return decoded

text = input("Enter a string: ")
huffman = Tree()

huffman.BuildTree(text)

print("\nHuffman Codes:")
for char, code in huffman.codes.items():
    print(repr(char), ":", code)
encoded_string = huffman.encode_str(text)
print("\nOriginal String :", text)
print("Encoded String  :", encoded_string)
decoded_string = huffman.decodestring(encoded_string)
print("Decoded String :", decoded_string)
