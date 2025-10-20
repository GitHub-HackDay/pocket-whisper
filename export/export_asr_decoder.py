#!/usr/bin/env python3
"""
Export Distil-Whisper decoder for text generation.

The decoder takes encoder embeddings and generates text tokens autoregressively.
This is needed for the full ASR pipeline: audio → encoder → decoder → text
"""

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def export_whisper_decoder():
    """Export Distil-Whisper decoder to ExecuTorch."""

    print("=" * 70)
    print("EXPORTING DISTIL-WHISPER DECODER")
    print("=" * 70)

    model_id = "distil-whisper/distil-small.en"

    # Load model
    print(f"\n[1/7] Loading {model_id}...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    print("  ✓ Model loaded")

    # Extract decoder
    print("\n[2/7] Extracting decoder...")
    decoder_base = model.get_decoder()
    decoder_base.eval()
    
    # Wrap decoder to handle inputs properly
    class DecoderWrapper(torch.nn.Module):
        def __init__(self, decoder, embed_tokens):
            super().__init__()
            self.decoder = decoder
            self.embed_tokens = embed_tokens
            
        def forward(self, input_ids, encoder_hidden_states):
            """
            Args:
                input_ids: (batch, seq_len) - decoder input token IDs
                encoder_hidden_states: (batch, enc_seq_len, hidden_size) - from encoder
            Returns:
                logits: (batch, seq_len, vocab_size)
            """
            # Get input embeddings
            inputs_embeds = self.embed_tokens(input_ids)
            
            # Run decoder
            decoder_outputs = self.decoder(
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                use_cache=False,
                return_dict=True
            )
            
            # Project to vocabulary
            # Note: Whisper uses shared embeddings for output projection
            logits = torch.matmul(
                decoder_outputs.last_hidden_state,
                self.embed_tokens.weight.t()
            )
            
            return logits
    
    # Get embedding layer
    embed_tokens = model.get_decoder().embed_tokens
    decoder = DecoderWrapper(decoder_base, embed_tokens)
    decoder.eval()
    print("  ✓ Decoder wrapped")

    # Create sample inputs
    print("\n[3/7] Creating sample inputs...")
    batch_size = 1
    dec_seq_len = 1  # Start with single token (autoregressive)
    enc_seq_len = 1500  # From encoder output
    hidden_size = 768  # Distil-Whisper Small
    
    sample_input_ids = torch.tensor([[50258]], dtype=torch.long)  # SOT token
    sample_encoder_states = torch.randn(batch_size, enc_seq_len, hidden_size)
    
    print(f"  ✓ Input IDs shape: {sample_input_ids.shape}")
    print(f"  ✓ Encoder states shape: {sample_encoder_states.shape}")

    # Export to torch.export format
    print("\n[4/7] Exporting decoder...")
    with torch.no_grad():
        exported = torch.export.export(
            decoder,
            (sample_input_ids, sample_encoder_states),
            strict=False
        )
    print("  ✓ Decoder exported")

    # Convert to ExecuTorch
    print("\n[5/7] Converting to ExecuTorch...")
    from executorch.exir import to_edge, EdgeCompileConfig

    edge_program = to_edge(
        exported,
        compile_config=EdgeCompileConfig(_check_ir_validity=False)
    )
    et_program = edge_program.to_executorch()
    print("  ✓ Converted to ExecuTorch")

    # Save
    print("\n[6/7] Saving .pte file...")
    output_path = Path("../app/src/main/assets/asr_decoder.pte")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved to: {output_path}")
    print(f"  ✓ File size: {size_mb:.1f} MB")

    # Validate
    print("\n[7/7] Validating export...")
    validate_decoder(output_path, sample_input_ids, sample_encoder_states)

    print("\n" + "=" * 70)
    print("✅ DECODER EXPORT COMPLETE!")
    print("=" * 70)
    print(f"\nDecoder output: {output_path.absolute()}")
    print(f"Size: {size_mb:.1f} MB")
    print("\nUsage in Android:")
    print("  1. Run encoder on mel spectrogram → encoder_states")
    print("  2. Start with SOT token (50258)")
    print("  3. Run decoder(token, encoder_states) → logits")
    print("  4. Sample next token from logits")
    print("  5. Repeat until EOT token (50257)")


def validate_decoder(pte_path, sample_ids, sample_encoder):
    """Test exported decoder."""
    from executorch.extension.pybindings import portable_lib

    try:
        runtime = portable_lib._load_for_executorch(str(pte_path))

        # Test forward pass
        output = runtime.forward([sample_ids, sample_encoder])[0]
        
        # Output should be (batch, seq_len, vocab_size)
        vocab_size = 51864  # Distil-Whisper vocab size
        print(f"  ✓ Decoder output shape: {output.shape}")
        print(f"  ✓ Expected: (batch=1, seq_len=1, vocab={vocab_size})")
        
        # Get next token prediction
        logits = output[0, -1, :]  # Last token logits
        next_token = torch.argmax(torch.tensor(logits)).item()
        print(f"  ✓ Predicted next token: {next_token}")
        print("  ✓ Validation successful!")

    except Exception as e:
        print(f"  ✗ Validation failed: {e}")
        raise


if __name__ == "__main__":
    export_whisper_decoder()

