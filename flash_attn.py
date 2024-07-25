import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def flash_attention_forward_kernel(
    Query, Key, Value,
    softmax_scale,
    TempBuffer,
    Lengths, MaxLengths,  # Renamed from L, M for clarity
    Output,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    BatchSize, NumHeads, SequenceLength,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Compute the forward pass of flash attention.
    
    This kernel processes the input in blocks for efficient computation.
    
    Flash Attention Tiling per Thread Block

    Sequence Length
    │
    ▼
    ┌─────────────────────────────────────────────┐
    │ Q1│ Q2│ Q3│ Q4│ Q5│ Q6│ Q7│ Q8│ Q9│Q10│Q11│Q12│ ◄─ Query (Q)
    ├─────────────────────────────────────────────┤
    │ K1│ K2│ K3│ K4│ K5│ K6│ K7│ K8│ K9│K10│K11│K12│ ◄─ Key (K)
    ├─────────────────────────────────────────────┤
    │ V1│ V2│ V3│ V4│ V5│ V6│ V7│ V8│ V9│V10│V11│V12│ ◄─ Value (V)
    └─────────────────────────────────────────────┘
    │   │   │   │
    │   │   │   └───────────────────┐
    │   │   └─────────────┐         │
    │   └───────┐         │         │
    │           │         │         │
    ▼           ▼         ▼         ▼
    ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
    │Block│   │Block│   │Block│   │Block│ ◄─ Thread Blocks
    │  1  │   │  2  │   │  3  │   │  4  │
    └─────┘   └─────┘   └─────┘   └─────┘
    │         │         │         │
    │         │         │         │
    ▼         ▼         ▼         ▼
    ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
    │ Out │   │ Out │   │ Out │   │ Out │ ◄─ Output
    │  1  │   │  2  │   │  3  │   │  4  │
    └─────┘   └─────┘   └─────┘   └─────┘

    Legend:
    Qi, Ki, Vi: Query, Key, and Value elements for position i
    Block: Thread block processing a tile of Q, K, and V
    Out: Output for each thread block

    Note: Each thread block processes a tile of size BLOCK_M x BLOCK_N
    """
    # Program ID in dimension 0 (sequence length dimension)
    start_m = tl.program_id(0)
    # Program ID in dimension 1 (batch * num_heads dimension)
    batch_head_index = tl.program_id(1)

    # Initialize offsets for accessing different parts of the input tensors
    seq_offsets = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    key_seq_offsets = tl.arange(0, BLOCK_N)
    model_dim_offsets = tl.arange(0, BLOCK_DMODEL)

    # Compute offsets for Query, Key, and Value tensors
    query_offset = batch_head_index * stride_qh + seq_offsets[:, None] * stride_qm + model_dim_offsets[None, :] * stride_qk
    key_offset = batch_head_index * stride_qh + key_seq_offsets[:, None] * stride_kn + model_dim_offsets[None, :] * stride_kk
    value_offset = batch_head_index * stride_qh + key_seq_offsets[:, None] * stride_qm + model_dim_offsets[None, :] * stride_qk

    # Initialize pointers to input tensors
    query_ptrs = Query + query_offset
    key_ptrs = Key + key_offset
    value_ptrs = Value + value_offset

    # Initialize pointers to temporary buffer
    temp_ptrs = TempBuffer + batch_head_index * SequenceLength + seq_offsets

    # Initialize accumulators and temporary variables
    max_scores = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    lengths = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Load query block: it will stay in SRAM throughout the computation
    query_block = tl.load(query_ptrs)

    # Main loop: process key-value pairs in blocks
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Load key block and compute query-key products
        key_block = tl.load(key_ptrs + start_n * stride_kn)
        query_key_products = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        query_key_products += tl.dot(query_block, key_block, trans_b=True)
        query_key_products *= softmax_scale

        # Apply causal mask: set scores to -inf where seq_offsets >= (start_n + key_seq_offsets)
        query_key_products += tl.where(seq_offsets[:, None] >= (start_n + key_seq_offsets[None, :]), 0, float("-inf"))

        # Compute local max scores and exponentials
        local_max_scores = tl.max(query_key_products, 1)
        local_exp_values = tl.exp(query_key_products - local_max_scores[:, None])
        local_exp_sum = tl.sum(local_exp_values, 1)

        # Update global max scores and lengths
        new_max_scores = tl.maximum(max_scores, local_max_scores)
        scale_old = tl.exp(max_scores - new_max_scores)
        scale_new = tl.exp(local_max_scores - new_max_scores)
        new_lengths = scale_old * lengths + scale_new * local_exp_sum

        # Update output accumulator
        output_scale = scale_new / new_lengths
        local_exp_values = local_exp_values * output_scale[:, None]

        acc_scale = lengths / new_lengths * scale_old
        tl.store(temp_ptrs, acc_scale)  # Workaround for compiler bug
        acc_scale = tl.load(temp_ptrs)
        acc = acc * acc_scale[:, None]

        # Load value block and update accumulator
        value_block = tl.load(value_ptrs + start_n * stride_vk)
        local_exp_values = local_exp_values.to(value_block.dtype)
        acc += tl.dot(local_exp_values, value_block)

        # Update max scores and lengths
        lengths = new_lengths
        max_scores = new_max_scores

    # Write out the final results
    length_ptrs = Lengths + batch_head_index * SequenceLength + seq_offsets
    max_score_ptrs = MaxLengths + batch_head_index * SequenceLength + seq_offsets
    tl.store(length_ptrs, lengths)
    tl.store(max_score_ptrs, max_scores)

    # Compute output pointers and store the result
    output_offsets = tl.arange(0, BLOCK_DMODEL)
    output_ptrs = Output + batch_head_index * stride_oh + seq_offsets[:, None] * stride_om + output_offsets[None, :] * stride_on
    tl.store(output_ptrs, acc)


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, softmax_scale):
        """
        Compute the forward pass of Flash Attention.
        
        Args:
            query, key, value: Input tensors
            softmax_scale: Scaling factor for softmax
        
        Returns:
            output: Result of attention mechanism
        """
        BLOCK_SIZE = 128
        MODEL_DIM = query.shape[-1]
        assert MODEL_DIM in {16, 32, 64, 128}, "Model dimension must be 16, 32, 64, or 128"
        assert query.shape[-1] == key.shape[-1] == value.shape[-1], "Q, K, V must have the same last dimension"

        # Initialize output and temporary tensors
        output = torch.empty_like(query)
        batch_size, num_heads, seq_length = query.shape[:3]
        grid = (triton.cdiv(seq_length, BLOCK_SIZE), batch_size * num_heads)
        
        temp_buffer = torch.empty((batch_size * num_heads, seq_length), device=query.device, dtype=torch.float32)
        lengths = torch.empty((batch_size * num_heads, seq_length), device=query.device, dtype=torch.float32)
        max_lengths = torch.empty((batch_size * num_heads, seq_length), device=query.device, dtype=torch.float32)

        num_warps = 4 if MODEL_DIM <= 64 else 8

        # Launch the CUDA kernel
        flash_attention_forward_kernel[grid](
            query, key, value, softmax_scale, 
            temp_buffer, lengths, max_lengths, output,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            batch_size, num_heads, seq_length,
            BLOCK_M=BLOCK_SIZE, BLOCK_N=BLOCK_SIZE, BLOCK_DMODEL=MODEL_DIM,
            num_warps=num_warps, num_stages=1,
        )

        # Save tensors for backward pass
        ctx.save_for_backward(query, key, value, output)
        ctx.softmax_scale = softmax_scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute the backward pass of Flash Attention.
        
        This implementation falls back to PyTorch's native attention mechanism
        for the backward pass.
        """
        query, key, value, output = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale
        
        # Reshape tensors for compatibility with PyTorch's attention mechanism
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Compute attention scores and softmax
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * softmax_scale
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Compute gradients
        grad_query = torch.matmul(grad_output.transpose(1, 2), key)
        grad_key = torch.matmul(query.transpose(-2, -1), grad_output.transpose(1, 2))
        grad_value = torch.matmul(attention_probs.transpose(-2, -1), grad_output.transpose(1, 2))
        
        # Restore original tensor shapes
        return grad_query.transpose(1, 2), grad_key.transpose(1, 2), grad_value.transpose(1, 2), None