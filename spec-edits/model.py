from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class SpeculativeEditor:
    def __init__(self, model_name="gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
    def edit_code(self, prompt, original_code):
        # Tokenize the full context (prompt + original code)
        full_context = prompt + "\n```ts\n" + original_code + "\n```"
        inputs = self.tokenizer(full_context, return_tensors="pt").to(self.device)
        
        # Use the original code as speculation
        speculation_tokens = self.tokenizer(original_code, return_tensors="pt")["input_ids"]
        
        # Generate with speculation
        output_ids = []
        current_pos = 0
        
        with torch.no_grad():
            while current_pos < len(speculation_tokens[0]):
                # Get model prediction for next token
                outputs = self.model(**inputs)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)
                
                # Compare with speculation
                if next_token == speculation_tokens[0][current_pos]:
                    # Speculation matches, accept token
                    output_ids.append(next_token.item())
                else:
                    # Speculation differs, generate new tokens
                    break
                
                current_pos += 1
                inputs = self.tokenizer(self.tokenizer.decode(output_ids), return_tensors="pt").to(self.device)
        
        # Complete the generation
        remaining_output = self.model.generate(
            inputs["input_ids"],
            max_length=len(original_code) * 2,  # Reasonable max length
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=0.0  # Greedy sampling
        )
        
        return self.tokenizer.decode(remaining_output[0], skip_special_tokens=True) 