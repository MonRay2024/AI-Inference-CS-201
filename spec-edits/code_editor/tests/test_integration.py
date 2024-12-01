def test_full_pipeline():
    """Test complete editing pipeline"""
    code = "def example(): pass"
    instruction = "Add error handling"
    
    # Test edit generation
    editor = CodeEditor(model, tokenizer)
    edited_code, operations, impact = editor.perfect_edit(code, instruction)
    assert edited_code != code
    
    # Test bug detection
    bug_detector = SmartBugDetector(model, tokenizer)
    bugs = bug_detector.detect_bugs(edited_code)
    assert isinstance(bugs, list)
    
    # Test semantic analysis
    analyzer = SemanticAnalyzer(model, tokenizer)
    analysis = analyzer.analyze_semantics(edited_code)
    assert 'semantic_understanding' in analysis