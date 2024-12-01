from code_editor.core.speculative_generator import SpeculativeGenerator
from code_editor.core.code_editor import CodeEditor
from code_editor.core.bug_analyzer import BugAnalyzer
from code_editor.utils.model_loader import load_model 

def main():
    # Load model and tokenizer
    model, tokenizer = load_model()
    editor = CodeEditor(model, tokenizer)
    bug_detector = SmartBugDetector(model, tokenizer)
    mcts_detector = MCTSBugDetector()
    semantic_analyzer = SemanticAnalyzer(model, tokenizer)
    metrics_collector = MetricsCollector()

    # Process edit
    edited_code, operations, impact = editor.perfect_edit(code, instruction)
    
    # Additional analysis
    bugs = bug_detector.detect_bugs(edited_code)
    semantic_analysis = semantic_analyzer.analyze_semantics(edited_code)
    metrics = metrics_collector.collect_metrics()