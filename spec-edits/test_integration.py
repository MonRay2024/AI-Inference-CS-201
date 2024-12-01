import unittest
from perfect_edit import SmartEditor
from bug_detection import SmartBugDetector
from text_generation import SpeculativeGenerator

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.model = load_test_model()
        self.tokenizer = load_test_tokenizer()
        self.editor = SmartEditor(self.model, self.tokenizer)
        self.bug_detector = SmartBugDetector(self.model, self.tokenizer)
        self.generator = SpeculativeGenerator(self.model, self.tokenizer)
        
    def test_end_to_end_workflow(self):
        code = "def test(): pass"
        instruction = "Add error handling"
        
        # Test edit generation
        edited_code, operations, scores = self.editor.perfect_edit(code, instruction)
        self.assertIsNotNone(edited_code)
        
        # Test bug detection
        bugs, analysis = self.bug_detector.detect_bugs(edited_code)
        self.assertIsInstance(bugs, list)
        
        # Test generation
        completion, stats = self.generator.generate(edited_code, max_tokens=100)
        self.assertIsNotNone(completion) 