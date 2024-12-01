import math

class MCTSNode:
    def __init__(self, code: str, parent=None):
        self.code = code
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        
    def select_child(self):
        """Select child using UCT formula"""
        return max(self.children, 
                  key=lambda c: c.value/c.visits + 
                  math.sqrt(2*math.log(self.visits)/c.visits))
    
    def expand(self):
        """Generate possible modifications"""
        possible_changes = self._generate_changes()
        for change in possible_changes:
            self.children.append(MCTSNode(change, self))
            
    def simulate(self) -> float:
        """Run simulation from current state"""
        return self._evaluate_code(self.code)
        
    def backpropagate(self, result: float):
        """Update node statistics"""
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(result)