#!/usr/bin/env python3
"""
Export Qwen2-0.5B-Instruct model for next-word prediction to ExecuTorch format (.pte)
Model: Qwen/Qwen2-0.5B-Instruct
Size: ~1GB (fp32) -> ~500MB (int8 quantized)
Latency target: 100-150ms for single token generation with QNN

Why Qwen2-0.5B:
- Smallest quality LLM that actually works well
- Good instruction following capabilities
- Supports QNN delegate for NPU acceleration
- Fast enough for real-time next-word prediction
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys
import time
import json
from torch.ao.quantization import quantize_dynamic
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def load_qwen2_model():
    """Load Qwen2-0.5B-Instruct model and tokenizer."""
    
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    print(f"📥 Loading {model_name}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Start with fp32, will quantize
        device_map="cpu",
        trust_remote_code=True
    )
    
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"   Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    print(f"   Vocab size: {model.config.vocab_size}")
    print(f"   Hidden size: {model.config.hidden_size}")
    
    return model, tokenizer


def create_next_word_prompt(context: str) -> str:
    """Create a prompt optimized for next-word prediction."""
    
    prompt = f"""You are a helpful assistant that predicts the next word.
Context: "{context}"
Task: Predict only the NEXT SINGLE WORD that should follow.
Next word:"""
    
    return prompt


def export_qwen2_to_executorch():
    """Export Qwen2-0.5B to ExecuTorch format with int8 quantization."""
    
    # 1. Load model and tokenizer
    model, tokenizer = load_qwen2_model()
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Prepare sample inputs for tracing
    test_contexts = [
        "Could you please",
        "I think we should",
        "Let me know if",
        "Thank you for"
    ]
    
    print("\n🧪 Testing next-word prediction...")
    for context in test_contexts:
        prompt = create_next_word_prompt(context)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids
        
        print(f"\n   Context: '{context}'")
        print(f"   Input shape: {input_ids.shape}")
        
        # Generate next word (1 token)
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=1,  # Only 1 token
                do_sample=False,    # Deterministic for tracing
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generation_time = (time.time() - start_time) * 1000
        
        # Decode the generated token
        generated_token = outputs[0][-1:]
        next_word = tokenizer.decode(generated_token, skip_special_tokens=True)
        
        print(f"   Next word: '{next_word}'")
        print(f"   Generation time (fp32): {generation_time:.2f}ms")
    
    # 3. Quantize model to int8
    print("\n⚡ Quantizing model to int8...")
    quantized_model = quantize_dynamic(
        model,
        {torch.nn.Linear},  # Quantize Linear layers
        dtype=torch.qint8
    )
    
    # Test quantized inference
    sample_prompt = create_next_word_prompt("Could you please")
    sample_inputs = tokenizer(sample_prompt, return_tensors="pt", padding=True)
    sample_input_ids = sample_inputs.input_ids
    
    start_time = time.time()
    with torch.no_grad():
        q_outputs = quantized_model(sample_input_ids)
    q_inference_time = (time.time() - start_time) * 1000
    print(f"   Quantized forward pass: {q_inference_time:.2f}ms")
    
    # 4. Export just the forward pass (we'll handle sampling in Kotlin)
    print("\n📝 Tracing model forward pass...")
    
    # Create a wrapper for cleaner export
    class ForwardOnlyWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, input_ids):
            # Just return logits, no generation
            outputs = self.model(input_ids)
            return outputs.logits
    
    wrapped_model = ForwardOnlyWrapper(quantized_model)
    wrapped_model.eval()
    
    # Trace the forward pass
    traced_model = torch.jit.trace(wrapped_model, sample_input_ids)
    
    # 5. Save TorchScript model
    torchscript_path = Path("../app/src/main/assets/llm_qwen2_0.5b_int8.pt")
    torchscript_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced_model, str(torchscript_path))
    print(f"   TorchScript saved to: {torchscript_path}")
    
    # 6. Export to ExecuTorch with QNN backend support
    print("\n🚀 Converting to ExecuTorch format...")
    try:
        from executorch.exir import to_edge, EdgeCompileConfig
        import torch.export as torch_export
        
        # Check if QNN backend is available
        try:
            from executorch.backends.qualcomm import QnnBackend
            has_qnn = True
            print("   ✅ QNN backend available for NPU acceleration")
        except ImportError:
            has_qnn = False
            print("   ⚠️  QNN backend not available, using CPU backend")
        
        # Export to Edge format
        edge_program = to_edge(
            torch_export.export(
                wrapped_model,
                (sample_input_ids,),
                dynamic_shapes=None
            )
        )
        
        # Compile to ExecuTorch with appropriate backend
        if has_qnn:
            # Use QNN backend for NPU acceleration on Snapdragon
            et_program = edge_program.to_executorch(
                config=EdgeCompileConfig(
                    _check_ir_validity=False,
                    _skip_dim_order=True
                ),
                backend_config={"qnn": QnnBackend()}
            )
            output_filename = "llm_qwen2_0.5b_int8_qnn.pte"
        else:
            # Fallback to CPU backend
            et_program = edge_program.to_executorch(
                config=EdgeCompileConfig(
                    _check_ir_validity=False,
                    _skip_dim_order=True
                )
            )
            output_filename = "llm_qwen2_0.5b_int8.pte"
        
        # Save .pte file
        output_path = Path(f"../app/src/main/assets/{output_filename}")
        with open(output_path, "wb") as f:
            f.write(et_program.buffer)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n✅ Successfully exported to ExecuTorch!")
        print(f"   File: {output_path}")
        print(f"   Size: {file_size_mb:.2f} MB")
        print(f"   Backend: {'QNN (NPU)' if has_qnn else 'CPU'}")
        
    except ImportError as e:
        print(f"⚠️  ExecuTorch export failed, keeping TorchScript version")
        print(f"   Error: {e}")
        output_path = torchscript_path
    
    # 7. Save tokenizer
    print("\n💾 Saving tokenizer...")
    tokenizer_dir = Path("../app/src/main/assets/llm_tokenizer")
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(tokenizer_dir))
    
    # Save simplified vocab for Android
    vocab = tokenizer.get_vocab()
    vocab_path = tokenizer_dir / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"   Tokenizer saved to: {tokenizer_dir}")
    print(f"   Vocab size: {len(vocab)} tokens")
    
    # 8. Create next-word prediction helper
    helper_code = '''
"""
Next-word prediction helper for Qwen2-0.5B
Handles prompt construction and token sampling
"""

import torch
import torch.nn.functional as F
import json
from typing import Tuple, List, Optional

class NextWordPredictor:
    def __init__(self, model_path: str, tokenizer):
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.tokenizer = tokenizer
        
        # Cache for better latency
        self.cache = {}
        self.max_cache_size = 100
        
    def predict_next_word(
        self, 
        context: str, 
        temperature: float = 0.8,
        top_k: int = 10
    ) -> Tuple[str, float]:
        """
        Predict the next word given context.
        Returns: (word, confidence_score)
        """
        
        # Check cache first
        cache_key = f"{context}_{temperature}_{top_k}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Create prompt
        prompt = f"""You are a helpful assistant that predicts the next word.
Context: "{context}"
Task: Predict only the NEXT SINGLE WORD that should follow.
Next word:"""
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        
        # Get logits
        with torch.no_grad():
            logits = self.model(input_ids)
            
        # Get last token logits
        next_token_logits = logits[0, -1, :]
        
        # Apply temperature
        next_token_logits = next_token_logits / temperature
        
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = -float('Inf')
        
        # Softmax to get probabilities
        probs = F.softmax(next_token_logits, dim=-1)
        
        # Get top prediction
        top_prob, top_id = torch.topk(probs, 1)
        
        # Decode
        next_word = self.tokenizer.decode([top_id.item()], skip_special_tokens=True).strip()
        confidence = top_prob.item()
        
        # Cache result
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry
            self.cache.pop(next(iter(self.cache)))
        self.cache[cache_key] = (next_word, confidence)
        
        return next_word, confidence
    
    def get_semantic_coherence(self, context: str, next_word: str) -> float:
        """
        Check if the next word is semantically coherent with context.
        Returns score 0-1.
        """
        
        # Simple heuristic: check if the combined phrase makes sense
        combined = f"{context} {next_word}"
        
        # You could use a small classifier here or simple rules
        # For now, use a basic check
        
        # Check for basic grammar patterns
        last_word = context.split()[-1] if context else ""
        
        # Common patterns that make sense
        good_patterns = {
            ("could", "you"): ["help", "please", "tell", "show"],
            ("thank", "you"): ["for", "very", "so"],
            ("I", "think"): ["we", "that", "it", "about"],
            ("let", "me"): ["know", "help", "see", "think"],
        }
        
        context_words = context.lower().split()[-2:] if len(context.split()) >= 2 else []
        if len(context_words) == 2:
            pattern = tuple(context_words)
            if pattern in good_patterns:
                if next_word.lower() in good_patterns[pattern]:
                    return 0.95
        
        # Default: moderate confidence
        return 0.75

# Usage:
# predictor = NextWordPredictor("model.pt", tokenizer)
# word, confidence = predictor.predict_next_word("Could you please")
# coherence = predictor.get_semantic_coherence("Could you please", word)
'''
    
    helper_path = Path("next_word_helper.py")
    helper_path.write_text(helper_code)
    print(f"\n📝 Next-word helper saved to: {helper_path}")
    
    return output_path


if __name__ == "__main__":
    print("=" * 60)
    print("Qwen2-0.5B-Instruct Export to ExecuTorch")
    print("=" * 60)
    
    try:
        output_file = export_qwen2_to_executorch()
        print("\n🎉 LLM export completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
