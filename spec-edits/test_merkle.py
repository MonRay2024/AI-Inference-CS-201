import unittest
from merkle_analysis import MerkleCodeAnalyzer, CodeBlock

class TestMerkleAnalysis(unittest.TestCase):
    def setUp(self):
        self.analyzer = MerkleCodeAnalyzer()
        
    def test_code_block_hashing(self):
        code1 = """
def test_function():
    x = 1
    return x
"""
        code2 = """
def test_function():
    x = 1
    return x  # Same code with comment
"""
        # Test that semantically identical code produces same hash
        tree1 = self.analyzer.build_merkle_tree(code1)
        tree2 = self.analyzer.build_merkle_tree(code2)
        self.assertEqual(tree1.hash, tree2.hash)
        
    def test_change_detection(self):
        old_code = """
def func1():
    return 1

def func2():
    return 2
"""
        new_code = """
def func1():
    return 1

def func2():
    x = 2
    return x
"""
        old_tree = self.analyzer.build_merkle_tree(old_code)
        new_tree = self.analyzer.build_merkle_tree(new_code)
        changes = self.analyzer.detect_changes(old_tree, new_tree)
        self.assertTrue(any(change[2] == "modified" for change in changes))

if __name__ == '__main__':
    unittest.main() 