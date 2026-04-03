"""Monkey patches for training-free SOTA evaluation (FastV, LVPruning) on Qwen3.5-VL.

This module intercepts `Qwen2VLTextModel.forward` to dynamically prune visual tokens
in the LLM after a few layers, simulating FastV and LVPruning.
"""

import types
import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast


def _find_visual_tokens(inputs_embeds, model_config):
    """
    Heuristically find visual tokens by inspecting inputs_embeds or config.
    In Qwen2-VL, vision tokens are typically inserted in large contiguous blocks.
    To be safe and training-free, we identify PRUNABLE tokens as any tokens except the first N
    and the last M (where the user question in VQA usually lies).
    """
    seq_len = inputs_embeds.shape[1]
    can_prune_mask = torch.zeros(seq_len, dtype=torch.bool, device=inputs_embeds.device)
    # Protect the first 16 tokens (system prompt) and the last 64 tokens (user query)
    # In VQA, visual tokens are in the middle.
    protect_start = min(16, seq_len // 4)
    protect_end = min(96, seq_len // 4)
    if seq_len > protect_start + protect_end:
        can_prune_mask[protect_start:-protect_end] = True
    return can_prune_mask


def patch_qwen2vl_text_model_forward(model: nn.Module, prune_layer_indices: list[int], keep_ratios: list[float], method: str = "fastv"):
    """
    Applies Monkey Patching to the `Qwen2VLTextModel`.
    Allows progressive pruning of visual tokens at specified layers.
    """
    assert hasattr(model, "model"), "Expected language model inside the VLM."
    if hasattr(model.model, "language_model"):
        llm = model.model.language_model
    elif hasattr(model, "language_model"):
        llm = model.language_model
    else:
        llm = model.model
        
    print(f"PATCHING LLM of type: {llm.__class__.__name__}")
    assert hasattr(llm, "layers"), f"Failed to find layers in llm. Found attributes: {dir(llm)}"
    if hasattr(model, "_qacr_sota_patched") and model._qacr_sota_patched:
        # Update config and return
        model._qacr_sota_config = {"prune_layers": prune_layer_indices, "keep_ratios": keep_ratios, "method": method}
        return

    model._qacr_sota_patched = True
    model._qacr_sota_config = {"prune_layers": prune_layer_indices, "keep_ratios": keep_ratios, "method": method}
    model._sota_tracked_original_length = 0
    model._sota_tracked_pruned_length = 0

    orig_forward = llm.forward

    def sota_patched_forward(self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs):
        
        # This wrapper calls the original forward, BUT we must intercept the hidden states MIDWAY.
        # Since we cannot easily inject into the middle of the original `forward` without completely copying
        # its 100+ lines of complex RoPE and position ID preparation, we use a hook based approach on the layers!
        # Wait, if we use a forward hook on the DecoderLayer, we can modify `hidden_states`, but we can't easily modify `past_key_values` and `attention_mask` coming in the next layers kwargs because they are passed from `Qwen2VLTextModel.forward` local variables.
        pass
        
    # An alternative is replacing `llm.layers` with a custom ModuleList that handles pruning internally!
    class PruningModuleList(nn.ModuleList):
        def __init__(self, original_layers):
            super().__init__()
            self.extend(original_layers)
            
        def __iter__(self):
            # This is called by `for i, decoder_layer in enumerate(self.layers)` in transformers.
            # Unfortunately, transformers does `for idx, decoder_layer in enumerate(self.layers):`
            # which uses `__iter__` or `__getitem__`.
            return super().__iter__()
            
    # The only foolproof way without copying `forward` is wrapping the original layers.
    # We replace each layer with a Wrapper that prunes its OWN inputs if it's the layer AFTER a prune layer!
    
    for i in range(len(llm.layers)):
        orig_layer = llm.layers[i]
        
        class PruningWrapper(nn.Module):
            def __init__(self, layer, layer_idx):
                super().__init__()
                self.layer = layer
                self.layer_idx = layer_idx
                
            def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None, **kwargs):
                # Check if we need to prune BEFORE running this layer.
                # If layer_idx matches a target prune layer, it means pruning happened at output of layer_idx - 1.
                # Actually, FastV computes attention at layer K, and prunes input to layer K+1.
                config = model._qacr_sota_config
                
                seq_len = hidden_states.shape[1]
                if self.layer_idx < 5:
                     print(f"Layer {self.layer_idx} forward: seq_len={seq_len}")
                
                # Check if this layer is a pruning trigger (i.e. we are at layer K+1, and need to prune based on layer K)
                # But we need attention scores from layer K. If we are at layer K, we should compute them and save them.
                is_eval_prefill = (seq_len > 1) and not self.training
                
                if is_eval_prefill and self.layer_idx in config["prune_layers"]:
                    # We are at the layer where pruning should happen. We compute attention scores, prune hidden_states,
                    # past_key_values, attention_mask, and position_ids.
                    idx = config["prune_layers"].index(self.layer_idx)
                    keep_r = config["keep_ratios"][idx]
                    
                    # 1. Compute pruning scores (using a proxy: average attention to last token)
                    # We use the previous layer's output (current hidden_states) to compute query-key dot products
                    q_proj = self.layer.self_attn.q_proj
                    k_proj = self.layer.self_attn.k_proj
                    
                    q = q_proj(hidden_states[:, -2:-1]) # Use the second to last token (assuming last is padding or similar, or just use last)
                    k = k_proj(hidden_states)
                    
                    bsz = q.shape[0]
                    num_heads = self.layer.self_attn.num_heads
                    num_kv_heads = self.layer.self_attn.num_key_value_heads
                    head_dim = self.layer.self_attn.head_dim
                    
                    q = q.view(bsz, 1, num_heads, head_dim).mean(dim=2)
                    k = k.view(bsz, seq_len, num_kv_heads, head_dim).mean(dim=2)
                    
                    attn_scores = torch.einsum('bxd,byd->bxy', q, k).squeeze(1) # [B, S]
                    
                    can_prune_mask = _find_visual_tokens(hidden_states, None)
                    num_prune_candidates = can_prune_mask.sum().item()
                    num_keep = int(num_prune_candidates * keep_r)
                    num_drop = num_prune_candidates - num_keep
                    
                    if num_drop > 0:
                        prune_scores = attn_scores[:, can_prune_mask].clone()
                        _, drop_indices_in_prune = torch.topk(prune_scores, num_drop, dim=1, largest=False)
                        
                        keep_indices_list = []
                        for b in range(bsz):
                            d_idx = drop_indices_in_prune[b]
                            prune_original_idx = torch.where(can_prune_mask)[0]
                            drop_orig = prune_original_idx[d_idx]
                            
                            keep_mask = torch.ones(seq_len, dtype=torch.bool, device=hidden_states.device)
                            keep_mask[drop_orig] = False
                            keep_indices_list.append(torch.where(keep_mask)[0])
                            
                        keep_indices = torch.stack(keep_indices_list, dim=0) # [B, new_seq_len]
                        new_seq_len = keep_indices.shape[1]
                        
                        model._sota_tracked_original_length = seq_len
                        model._sota_tracked_pruned_length = new_seq_len
                        print(f"FASTV PRUNE TRIGGERED: {seq_len} -> {new_seq_len}")
                        
                        # Apply Pruning to hidden_states
                        hidden_states = torch.gather(hidden_states, 1, keep_indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1]))
                        
                        # Apply to position_ids if present. position_ids shape depends on implementation (usually 3D for Qwen2VL)
                        if position_ids is not None:
                            if position_ids.ndim == 3: # [3, B, S]
                                # gather along S (dim 2)
                                pos_keep = keep_indices.unsqueeze(0).expand(3, -1, -1)
                                position_ids = torch.gather(position_ids, 2, pos_keep)
                            elif position_ids.ndim == 2: # [B, S]
                                position_ids = torch.gather(position_ids, 1, keep_indices)
                                
                        # Apply to position_embeddings
                        if "position_embeddings" in kwargs and kwargs["position_embeddings"] is not None:
                            cos, sin = kwargs["position_embeddings"]
                            # cos, sin shape: [B, S, head_dim]
                            cos = torch.gather(cos, 1, keep_indices.unsqueeze(-1).expand(-1, -1, cos.shape[-1]))
                            sin = torch.gather(sin, 1, keep_indices.unsqueeze(-1).expand(-1, -1, sin.shape[-1]))
                            kwargs["position_embeddings"] = (cos, sin)
                                
                        # Apply to attention_mask if present
                        if getattr(attention_mask, "shape", None) is not None:
                            # Causal mask is [B, 1, S_q, S_k]
                            if attention_mask.ndim == 4:
                                attention_mask = attention_mask[:, :, :new_seq_len, :new_seq_len] # Just slice to new len, assume mask is causal and valid
                        elif isinstance(attention_mask, dict): # Qwen2VL custom mask mapping
                            for k_mask, v_mask in list(attention_mask.items()):
                                if v_mask is not None and v_mask.ndim == 4:
                                    v_mask = v_mask[:, :, :new_seq_len, :new_seq_len]
                                    attention_mask[k_mask] = v_mask
                        
                        # Apply to past_key_values
                        if past_key_values is not None:
                            for l_idx in range(self.layer_idx):
                                if l_idx < len(past_key_values.key_cache):
                                    k_v = past_key_values.key_cache[l_idx] # [B, num_heads, S, head_dim]
                                    v_v = past_key_values.value_cache[l_idx]
                                    k_v = torch.gather(k_v, 2, keep_indices.unsqueeze(1).unsqueeze(-1).expand(-1, k_v.shape[1], -1, k_v.shape[-1]))
                                    v_v = torch.gather(v_v, 2, keep_indices.unsqueeze(1).unsqueeze(-1).expand(-1, v_v.shape[1], -1, v_v.shape[-1]))
                                    past_key_values.key_cache[l_idx] = k_v
                                    past_key_values.value_cache[l_idx] = v_v
                                    
                        # Save the keep_indices on the model so we can prune cache again during decode phase if necessary? 
                        # No, during decode phase seq_len=1, the cache is already pruned! 
                        # When decoding, past_key_values has length new_seq_len.
                        # Wait! During decode, Qwen2VL update `position_ids` using cache length! 
                        # Qwen2VL natively uses `past_key_values.get_seq_length()` to determine `position_ids`.
                        # Since we truncated `past_key_values`, the `past_seen_tokens` will drop!
                        # This ruins relative position IDs for generation.
                        # To fix: we must preserve `get_seq_length()` or spoof it.
                        model._sota_cache_length_offset = num_drop
                        
                # Call original layer
                out = self.layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    **kwargs
                )
                
                # If out is a tuple, we return it but we override hidden_states just in case
                if isinstance(out, tuple):
                    return out
                return out

        llm.layers[i] = PruningWrapper(orig_layer, i)
        
    # We must patch past_key_values.get_seq_length() during decode so it doesn't rewind text pos!
    orig_text_forward = llm.forward
    def offset_text_forward(self, *args, **kwargs):
        if "past_key_values" in kwargs and kwargs["past_key_values"] is not None:
            if hasattr(model, "_sota_cache_length_offset") and model._sota_cache_length_offset > 0:
                pkv = kwargs["past_key_values"]
                orig_get = pkv.get_seq_length
                # We monkey patch the cache object instance
                if not hasattr(pkv, "_qacr_offset_patched"):
                    def spoof_len(self):
                        actual_len = len(self.key_cache[0][0][0]) if len(self.key_cache) > 0 else 0
                        return actual_len + model._sota_cache_length_offset
                    pkv.get_seq_length = types.MethodType(spoof_len, pkv)
                    pkv._qacr_offset_patched = True
        return orig_text_forward(*args, **kwargs)
        
    llm.forward = types.MethodType(offset_text_forward, llm)


def apply_fastv(model, prune_layer_idx=3, keep_ratio=0.45):
    patch_qwen2vl_text_model_forward(model, [prune_layer_idx], [keep_ratio], "fastv")
    
def apply_lvpruning(model, prune_layer_indices=[3, 7, 11], keep_ratios=[0.8, 0.6, 0.45]):
    patch_qwen2vl_text_model_forward(model, prune_layer_indices, keep_ratios, "lvpruning")


