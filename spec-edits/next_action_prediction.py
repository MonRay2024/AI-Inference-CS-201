def predict_next_action(model, tokenizer, prompt: str) -> str:
    """Predicts the next likely action based on the current code state"""
    prediction_prompt = prompt + "\nNext likely action:"
    inputs = tokenizer(prediction_prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        inputs["input_ids"],
        max_length=100,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id
    )
    
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction.split("Next likely action:")[-1].strip()