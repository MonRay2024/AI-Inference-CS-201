from typing import Dict, List, Tuple, Set
import hashlib
from dataclasses import dataclass
import ast
from collections import defaultdict
import random

@dataclass
class CodeBlock:
    content: str
    start_line: int
    end_line: int
    hash: str
    children: List['CodeBlock'] = None

class ProbabilisticSkipList:
    def __init__(self, max_level=16, p=0.5):
        self.max_level = max_level
        self.p = p
        self.head = self._create_node(self.max_level, None, None)
        self.level = 0
    
    def _random_level(self):
        level = 0
        while random.random() < self.p and level < self.max_level - 1:
            level += 1
        return level
    
    def _create_node(self, level, key, value):
        return {
            'key': key,
            'value': value,
            'forward': [None] * (level + 1)
        }
    
    def insert(self, key, value):
        update = [None] * self.max_level
        current = self.head
        
        for i in range(self.level, -1, -1):
            while (current['forward'][i] and 
                   current['forward'][i]['key'] < key):
                current = current['forward'][i]
            update[i] = current
        
        level = self._random_level()
        if level > self.level:
            for i in range(self.level + 1, level + 1):
                update[i] = self.head
            self.level = level
        
        node = self._create_node(level, key, value)
        for i in range(level + 1):
            node['forward'][i] = update[i]['forward'][i]
            update[i]['forward'][i] = node

class MerkleCodeAnalyzer:
    def __init__(self):
        self.block_cache = {}  # Cache for computed hashes
        self.change_history = defaultdict(list)  # Track changes per block
        
    def _compute_hash(self, content: str) -> str:
        """Compute semantic-aware hash of code block"""
        # Normalize code to ignore non-semantic differences
        normalized = self._normalize_code(content)
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def _normalize_code(self, code: str) -> str:
        """Normalize code by removing whitespace, comments, etc."""
        tree = ast.parse(code)
        return ast.unparse(tree)  # Returns normalized code
    
    def build_merkle_tree(self, code: str) -> CodeBlock:
        """Build Merkle tree from code"""
        tree = ast.parse(code)
        
        def process_node(node: ast.AST, start_line: int) -> CodeBlock:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                # Process function/class definitions
                content = ast.unparse(node)
                children = []
                
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.ClassDef)):
                        children.append(process_node(child, child.lineno))
                
                block = CodeBlock(
                    content=content,
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    hash=self._compute_hash(content),
                    children=children
                )
                self.block_cache[block.hash] = block
                return block
                
            return None
        
        root_blocks = []
        for node in ast.iter_child_nodes(tree):
            block = process_node(node, node.lineno)
            if block:
                root_blocks.append(block)
        
        # Create root block
        root_content = code
        root_hash = self._compute_hash(root_content)
        root = CodeBlock(
            content=root_content,
            start_line=1,
            end_line=len(code.splitlines()),
            hash=root_hash,
            children=root_blocks
        )
        
        return root

    def detect_changes(self, old_tree: CodeBlock, new_tree: CodeBlock) -> List[Tuple[int, int, str]]:
        """Detect changes between two Merkle trees"""
        changes = []
        
        def compare_blocks(old: CodeBlock, new: CodeBlock):
            if old.hash != new.hash:
                # Hash mismatch indicates change
                if not old.children or not new.children:
                    # Leaf node or structure change
                    changes.append((new.start_line, new.end_line, "modified"))
                else:
                    # Compare children
                    old_children = {child.hash: child for child in old.children}
                    new_children = {child.hash: child for child in new.children}
                    
                    # Find modified blocks
                    for hash_, new_child in new_children.items():
                        if hash_ in old_children:
                            compare_blocks(old_children[hash_], new_child)
                        else:
                            changes.append((new_child.start_line, new_child.end_line, "added"))
                    
                    # Find removed blocks
                    for hash_, old_child in old_children.items():
                        if hash_ not in new_children:
                            changes.append((old_child.start_line, old_child.end_line, "removed"))
        
        compare_blocks(old_tree, new_tree)
        return changes

    def track_block_history(self, block_hash: str, edit: str):
        """Track edit history for code blocks"""
        self.change_history[block_hash].append(edit)
    
    def predict_impact(self, changed_block: CodeBlock, full_tree: CodeBlock) -> Dict[str, float]:
        """Predict impact of changes using Merkle tree structure"""
        impact_scores = {}
        
        def calculate_block_impact(block: CodeBlock, distance: int):
            # Calculate impact score based on:
            # 1. Distance from changed block
            # 2. Historical changes
            # 3. Dependency relationships
            history_factor = len(self.change_history[block.hash]) / 10  # Normalize
            impact = 1.0 / (1 + distance) * (1 + history_factor)
            impact_scores[block.hash] = min(impact, 1.0)
            
            if block.children:
                for child in block.children:
                    calculate_block_impact(child, distance + 1)
        
        calculate_block_impact(changed_block, 0)
        return impact_scores 