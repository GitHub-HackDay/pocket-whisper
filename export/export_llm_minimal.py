#!/usr/bin/env python3
"""
Minimal LLM export - just get it working!
Using GPT-2 small (124M params)
"""

import torch
import sys
from pathlib import Path

def export_llm_minimal():
    print("="*60)
    print("🚀 Simple LLM Export - GPT-2 Small")
    print("="*60)
    
    try:
        # Import only what we need
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "gpt2"  # Original GPT-2, 124M params
        
        print(f"\n📥 Loading {model_name}...")
        print("   Small model (124M params), ~500MB")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        
        print(f"✅ Model loaded!")
        
        # Quick test
        print("\n🧪 Testing generation...")
        text = "Hello world"
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   '{text}' → '{generated}'")
        
        # Create wrapper for clean export
        class SimpleGPT2(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, input_ids):
                # Just return logits, no attention mask
                outputs = self.model(input_ids=input_ids, use_cache=False)
                return outputs.logits
        
        # Wrap and trace
        print("\n📝 Tracing model...")
        wrapped = SimpleGPT2(model)
        wrapped.eval()
        
        dummy_input = torch.randint(0, 50000, (1, 10))  # Random token IDs
        
        with torch.no_grad():
            traced = torch.jit.trace(wrapped, dummy_input, strict=False)
        
        # Skip mobile optimization to avoid issues
        print("💾 Saving model...")
        
        output_dir = Path("../app/src/main/assets")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as regular TorchScript
        output_path = output_dir / "llm_gpt2_small.pt"
        traced.save(str(output_path))
        
        # Save tokenizer
        tokenizer_path = output_dir / "llm_tokenizer"
        tokenizer_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(tokenizer_path))
        
        # Report
        size_mb = output_path.stat().st_size / (1024*1024)
        print(f"\n✅ SUCCESS!")
        print(f"   Model: {output_path}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   Tokenizer: {tokenizer_path}")
        print(f"   Type: Standard GPT-2 (124M params)")
        
        print("\n📱 For Android:")
        print("   - Use Module.load() instead of lite interpreter")
        print("   - Model is ready for next-word prediction")
        print("   - Expected latency: 50-150ms on S25 Ultra")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    export_llm_minimal()
