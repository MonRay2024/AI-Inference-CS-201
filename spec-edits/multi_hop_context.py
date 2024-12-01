from typing import List

def retrieve_multi_hop_context(model, tokenizer, code: str, related_files: List[str]) -> str:
    """Retrieves relevant context from related files"""
    context_prompt = f"""Given this code:
{code}

And these related files: {', '.join(related_files)}

Relevant context and relationships:"""
    
    inputs = tokenizer(context_prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        inputs["input_ids"],
        max_length=500,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id
    )
    
    context = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return context.split("Relevant context and relationships:")[-1].strip()