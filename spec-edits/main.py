import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from text_generation import vanilla_edit, speculative_edit, compare_methods
from next_action_prediction import predict_next_action
from perfect_edit import perfect_edit
from bug_detection import detect_bugs
from multi_hop_context import retrieve_multi_hop_context
from torch.utils.tensorboard import SummaryWriter
from typing import List, Tuple, Dict
from model import SpeculativeEditor
from perfect_edit import SmartEditor

def setup_tensorboard():
    return SummaryWriter()

def compare_methods(model, tokenizer, prompt: str, max_tokens: int, writer) -> Tuple[str, str, float, float, float]:
    import time
    from difflib import SequenceMatcher

    start_time = time.time()
    vanilla_output = vanilla_edit(model, tokenizer, prompt, max_tokens)
    vanilla_time = time.time() - start_time

    start_time = time.time()
    speculative_output, correct_speculations = speculative_edit(model, tokenizer, prompt, max_tokens)
    speculative_time = time.time() - start_time

    # Calculate similarity ratio
    similarity = SequenceMatcher(None, vanilla_output, speculative_output).ratio()

    writer.add_scalar('Time_Difference', vanilla_time - speculative_time, max_tokens)
    writer.add_scalar('Output_Similarity', similarity, max_tokens)
    writer.add_scalar('Correct_Speculations', correct_speculations, max_tokens)

    return vanilla_output, speculative_output, vanilla_time, speculative_time, similarity

def main():
    # Load model and tokenizer
    model_path = "/workspace/llama3finetune/model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()
    model.eval()

    # Setup editor with Merkle analysis
    editor = SmartEditor(model, tokenizer)
    
    prompt = "Please add a single comment"
    original_code = """export default function Visualization() {
      const [instanceIdInputs, setInstanceIdInputs] = createSignal<
        InstanceId[] | null
      >(null);
      const [storedInput, setStoredInput] = createSignal<string>("");
      const [datapointOptions, setDatapointOptions] = createSignal<PropsInstance[]>(
        []
      );
      const [shouldRefreshGold, setShouldRefreshGold] =
        createSignal<boolean>(false);
      const [showGold, setShowGold] = createSignal<boolean>(false);
      const [selectedGoldRequestId, setSelectedGoldRequestId] = createSignal<
        string | undefined
      >(undefined);
      const [goldInstances, setGoldInstances] = createSignal<
        {
          sessionId: string;
          email: string | undefined;
          requestId: string | undefined;
          dateAdded: Date;
          type: $Enums.CppGoldExampleType;
        }[]
      >([]);
    }"""
    
    edited_code, operations, impact_scores = editor.perfect_edit(prompt, original_code)
    print("Edited code:", edited_code)
    print("\nOperations:", operations)
    print("\nImpact analysis:")
    for block_hash, score in impact_scores.items():
        print(f"Block {block_hash[:8]}: Impact score {score:.2f}")

    # Test next action prediction
    next_action = predict_next_action(model, tokenizer, test_prompt)
    print(f"\nPredicted next action: {next_action}")

    # Test perfect edit
    perfect_edit_result = perfect_edit(model, tokenizer, test_prompt, "Add a comment explaining the purpose of the Visualization function")
    print(f"\nPerfect Edit result:\n{perfect_edit_result}")

    # Test bug detection
    bugs = detect_bugs(model, tokenizer, test_prompt)
    print(f"\nDetected bugs:\n{bugs}")

    # Test multi-hop context retrieval
    context = retrieve_multi_hop_context(model, tokenizer, test_prompt, ["file1.ts", "file2.ts"])
    print(f"\nRetrieved multi-hop context:\n{context}")

    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()