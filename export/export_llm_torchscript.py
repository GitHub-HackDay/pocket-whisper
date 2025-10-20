#!/usr/bin/env python3
"""
Export Qwen2-0.5B to TorchScript for PyTorch Mobile
Simplified for hackathon - working solution over perfect optimization
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import time

def export_llm_for_mobile():
    """Export Qwen2-0.5B to TorchScript (.pt) for PyTorch Mobile."""
    
    print("="*60)
    print("🚀 Exporting Qwen2-0.5B for PyTorch Mobile")
    print("="*60)
    
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    
    # 1. Load model
    print(f"\n📥 Loading {model_name}...")
    print("   (This will download ~1GB, may take 5-10 minutes)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    model.eval()
    
    print(f"✅ Model loaded")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Test next-word prediction
    print("\n🧪 Testing next-word prediction...")
    test_contexts = [
        "Could you please",
        "I think we should",
        "Thank you for"
    ]
    
    for context in test_contexts:
        # Simple prompt for next word
        prompt = f"Complete with one word: '{context}' →"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        latency = (time.time() - start) * 1000
        
        next_word = tokenizer.decode(outputs[0][-1:], skip_special_tokens=True)
        print(f"   '{context}' → '{next_word}' ({latency:.0f}ms)")
    
    # 3. Create wrapper for cleaner export
    class NextWordModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, input_ids, attention_mask):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.logits
    
    wrapped_model = NextWordModel(model)
    wrapped_model.eval()
    
    # 4. Trace model
    print("\n📝 Tracing model for mobile...")
    sample_input = tokenizer("Sample text", return_tensors="pt")
    
    traced_model = torch.jit.trace(
        wrapped_model, 
        (sample_input.input_ids, sample_input.attention_mask)
    )
    
    # 5. Optimize for mobile (basic optimization only)
    print("⚡ Optimizing for mobile...")
    # Note: Full mobile optimization may break some transformer models
    # Using basic optimization for safety
    traced_model = torch.jit.optimize_for_mobile(traced_model, backend="cpu")
    
    # 6. Save model
    output_path = Path("../app/src/main/assets/llm_qwen2_0.5b.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced_model._save_for_lite_interpreter(str(output_path))
    
    # 7. Save tokenizer
    tokenizer_path = Path("../app/src/main/assets/llm_tokenizer")
    tokenizer_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(tokenizer_path))
    
    size_mb = output_path.stat().st_size / (1024*1024)
    print(f"\n✅ SUCCESS!")
    print(f"   Model: {output_path}")
    print(f"   Size: {size_mb:.1f} MB")
    print(f"   Tokenizer: {tokenizer_path}")
    print(f"   Format: TorchScript (PyTorch Mobile)")
    
    # 8. Usage example
    print("\n📱 Android Usage:")
    print("""
// Load in Kotlin
val module = Module.load(assetFilePath("llm_qwen2_0.5b.pt"))

// Tokenize input (need to port tokenizer logic)
val inputIds = tokenize(context)  
val attentionMask = createAttentionMask(inputIds)

// Get logits
val logits = module.forward(
    IValue.from(inputIds),
    IValue.from(attentionMask)
).toTensor()

// Get next word from logits
val nextTokenId = argmax(logits[-1])
val nextWord = decode(nextTokenId)
""")
    
    print("\n⚠️ Note: For production, consider quantization to reduce size:")
    print("   - Dynamic quantization can reduce to ~250MB")
    print("   - But may increase latency slightly")
    
    return output_path

if __name__ == "__main__":
    export_llm_for_mobile()
