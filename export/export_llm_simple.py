#!/usr/bin/env python3
"""
Simple LLM export script for Qwen2-0.5B
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path
import time
import sys

def export_llm_simple():
    print("="*60)
    print("🚀 Exporting DistilGPT2 LLM Model")
    print("="*60)
    
    # Using DistilGPT2 - small (82M params), fast, and well-supported
    model_name = "distilgpt2"
    
    try:
        # Load model and tokenizer
        print(f"\n📥 Loading {model_name}...")
        print("   This is a small model (~350MB), should download quickly")
        
        tokenizer = GPT2Tokenizer.from_pretrained(
            model_name,
            force_download=False
        )
        
        model = GPT2LMHeadModel.from_pretrained(
            model_name,
            force_download=False
        )
        
        model.eval()
        print(f"✅ Model loaded successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        
        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Test the model
        print("\n🧪 Testing next-word prediction...")
        test_contexts = [
            "Could you please",
            "I think we should",
            "Thank you for"
        ]
        
        for context in test_contexts[:1]:  # Test just one for speed
            inputs = tokenizer(context, return_tensors="pt")
            
            start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            latency = (time.time() - start) * 1000
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            next_word = generated[len(context):].strip()
            print(f"   '{context}' → '{next_word}' ({latency:.0f}ms)")
        
        # Create a wrapper for cleaner export
        print("\n📦 Creating simplified model wrapper...")
        class SimpleLLM(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, input_ids, attention_mask=None):
                # For mobile, we'll just return logits
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False  # Disable KV cache for tracing
                )
                return outputs.logits
        
        # Wrap the model
        simple_model = SimpleLLM(model)
        simple_model.eval()
        
        # Create sample input for tracing
        print("\n📝 Tracing model for mobile...")
        sample_text = "Hello world"
        sample_input = tokenizer(sample_text, return_tensors="pt", padding=True)
        input_ids = sample_input.input_ids
        attention_mask = sample_input.attention_mask
        
        # Trace the model
        with torch.no_grad():
            traced_model = torch.jit.trace(
                simple_model,
                (input_ids, attention_mask),
                strict=False  # Allow some flexibility in tracing
            )
        
        # Optimize for mobile
        print("⚡ Optimizing for mobile...")
        print("   Note: This may take several minutes for a 500M parameter model...")
        from torch.utils.mobile_optimizer import optimize_for_mobile
        optimized = optimize_for_mobile(traced_model)
        
        # Save model
        output_dir = Path("../app/src/main/assets")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "llm_distilgpt2.pt"
        print(f"\n💾 Saving to {output_path}...")
        print("   This will take a few minutes...")
        optimized._save_for_lite_interpreter(str(output_path))
        
        # Save tokenizer
        tokenizer_path = output_dir / "llm_tokenizer"
        tokenizer_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(tokenizer_path))
        
        # Report success
        size_mb = output_path.stat().st_size / (1024*1024)
        print(f"\n✅ SUCCESS!")
        print(f"   Model saved: {output_path}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   Tokenizer saved: {tokenizer_path}")
        
        print("\n📊 Model Stats:")
        print(f"   - Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        print(f"   - Vocab size: {tokenizer.vocab_size}")
        print(f"   - Model size: {size_mb:.1f} MB")
        print(f"   - Expected mobile inference: 100-200ms with optimization")
        
        return str(output_path)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Ensure you have ~2GB free disk space")
        print("3. Try setting force_download=True to re-download")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    export_llm_simple()
