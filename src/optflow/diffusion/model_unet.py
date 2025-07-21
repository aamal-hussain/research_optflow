import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Re-use existing Transformer Components (assuming they are in the same file) ---
class MultiHeadSelfAttention(nn.Module):
    """
    A basic Multi-Head Self-Attention block.
    Assumes input shape: (batch_size, N_elements, model_dim)
    """
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        assert self.head_dim * num_heads == model_dim, "model_dim must be divisible by num_heads"

        self.wq = nn.Linear(model_dim, model_dim)
        self.wk = nn.Linear(model_dim, model_dim)
        self.wv = nn.Linear(model_dim, model_dim)
        self.fc_out = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, N_elements, _ = x.shape

        Q = self.wq(x).view(batch_size, N_elements, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.wk(x).view(batch_size, N_elements, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.wv(x).view(batch_size, N_elements, self.num_heads, self.head_dim).transpose(1, 2)

        energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        x = torch.matmul(attention, V).transpose(1, 2).contiguous()
        x = x.view(batch_size, N_elements, self.model_dim)
        x = self.fc_out(x)
        return x

class FeedForward(nn.Module):
    """
    A simple position-wise Feed-Forward Network.
    Applied independently to each element.
    """
    def __init__(self, model_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    Basic Transformer Encoder Block: MHSA -> LayerNorm -> FFN -> LayerNorm
    """
    def __init__(self, model_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(model_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ffn = FeedForward(model_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)

        ff_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ff_output)
        return x

class CrossAttention(nn.Module):
    """
    Cross-Attention block.
    Query comes from one set, Key/Value from another set.
    """
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        assert self.head_dim * num_heads == model_dim, "model_dim must be divisible by num_heads"

        self.wq = nn.Linear(model_dim, model_dim)
        self.wk = nn.Linear(model_dim, model_dim)
        self.wv = nn.Linear(model_dim, model_dim)
        self.fc_out = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_x, kv_x, mask=None):
        batch_size, N_queries, _ = query_x.shape
        _, N_kv, _ = kv_x.shape

        Q = self.wq(query_x).view(batch_size, N_queries, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.wk(kv_x).view(batch_size, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.wv(kv_x).view(batch_size, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        x = torch.matmul(attention, V).transpose(1, 2).contiguous()
        x = x.view(batch_size, N_queries, self.model_dim)
        x = self.fc_out(x)
        return x

class DecoderTransformerBlock(nn.Module):
    """
    Decoder Block with Self-Attention, Cross-Attention (for skip connections), FFN.
    """
    def __init__(self, model_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(model_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(model_dim)

        self.cross_attention = CrossAttention(model_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(model_dim)

        self.ffn = FeedForward(model_dim, ff_dim, dropout)
        self.norm3 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, skip_connection_features, mask=None):
        attn_output = self.self_attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)

        # Cross-attention with skip connection features
        # Query: x (current decoder state)
        # KV: skip_connection_features
        cross_attn_output = self.cross_attention(self.norm2(x), skip_connection_features)
        x = x + self.dropout(cross_attn_output)

        ff_output = self.ffn(self.norm3(x))
        x = x + self.dropout(ff_output)
        return x

# --- New Downsampling and Upsampling Modules ---

class LearnedQuerySetDownsampler(nn.Module):
    """
    Downsampling layer using learned queries and cross-attention.
    Input: (batch_size, N_in, model_dim)
    Output: (batch_size, N_out, model_dim)
    """
    def __init__(self, model_dim, num_heads, n_output_elements, dropout=0.1):
        super().__init__()
        self.n_output_elements = n_output_elements
        self.model_dim = model_dim

        # Learned queries that will attend to the input set
        self.learned_queries = nn.Parameter(torch.randn(1, n_output_elements, model_dim))

        self.cross_attention = CrossAttention(model_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(model_dim)
        self.ffn = FeedForward(model_dim, model_dim * 2, dropout) # Optional FFN after attention
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.shape[0]

        # Expand learned queries to match batch size
        queries = self.learned_queries.expand(batch_size, -1, -1) # (B, N_out, model_dim)

        # Queries attend to the input set
        # Query: learned_queries (N_out)
        # KV: x (N_in)
        attn_output = self.cross_attention(self.norm(queries), x)
        
        # Optional: Add FFN to refine the output of attention
        output = attn_output + self.dropout(self.ffn(attn_output)) # Residual connection
        return output

class SetFeaturePropagator(nn.Module):
    """
    Upsampling/Feature Propagation layer using skip connection features as queries.
    Input:
        - `lower_res_features`: Features from the deeper, lower resolution layer (B, N_low, model_dim)
        - `higher_res_queries`: Features from the corresponding skip connection (B, N_high, model_dim)
                                These act as queries to retrieve info from lower_res_features.
    Output: (batch_size, N_high, model_dim) - enriched higher resolution features.
    """
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attention = CrossAttention(model_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(model_dim)
        self.ffn = FeedForward(model_dim, model_dim * 2, dropout) # Optional FFN after attention
        self.dropout = nn.Dropout(dropout)

        # MLP to combine interpolated and skip features (can be a simple linear layer too)
        self.combine_mlp = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim), # Concatenation doubles feature dim
            nn.ReLU(),
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim)
        )

    def forward(self, lower_res_features, higher_res_queries):
        # Query: higher_res_queries (N_high)
        # KV: lower_res_features (N_low)
        # This attention learns to "interpolate" features from lower_res_features
        # based on the content of higher_res_queries.
        interpolated_features = self.cross_attention(self.norm(higher_res_queries), lower_res_features)
        
        # Combine interpolated features with original higher_res_queries (skip connection)
        combined_features = torch.cat((interpolated_features, higher_res_queries), dim=-1)
        
        # Refine combined features
        output = self.combine_mlp(combined_features)
        return output

class TransformerUNetForSets(nn.Module):
    def __init__(self, input_channel_dim, output_channel_dim, model_dim,
                 num_heads, ff_dim, encoder_block_counts, decoder_block_counts,
                 downsample_cardinalities, dropout=0.1):
        super().__init__()
        self.input_channel_dim = input_channel_dim
        self.output_channel_dim = output_channel_dim
        self.model_dim = model_dim
        
        # Input Embedding Layer
        self.input_projection = nn.Linear(input_channel_dim, model_dim)

        # Encoder Path
        self.encoder_stages = nn.ModuleList()
        # The downsample_cardinalities list defines the N for each downsampling step.
        # e.g., if input N=128, downsample_cardinalities=[64, 32, 16]
        
        current_N = None # Will be set in forward pass by initial_N

        # Build encoder stages
        for i, num_blocks in enumerate(encoder_block_counts):
            stage = nn.ModuleList([
                TransformerBlock(model_dim, num_heads, ff_dim, dropout)
                for _ in range(num_blocks)
            ])
            self.encoder_stages.append(stage)
            
            if i < len(downsample_cardinalities):
                # We need to calculate the n_output_elements based on the specified cardinalities
                # The actual n_output_elements for the *next* stage.
                next_N_out = downsample_cardinalities[i]
                self.encoder_stages.append(
                    LearnedQuerySetDownsampler(model_dim, num_heads, next_N_out, dropout)
                )

        # Bottleneck
        self.bottleneck = nn.Sequential(*[
            TransformerBlock(model_dim, num_heads, ff_dim, dropout)
            for _ in range(decoder_block_counts[0]) # Use first decoder block count for bottleneck
        ])

        # Decoder Path
        self.decoder_stages = nn.ModuleList()
        # Up-sampling will match the cardinalities of the corresponding encoder skip connections
        
        # Need to know the N values at each skip connection point from the encoder
        # This will be inferred in the forward pass
        
        for i in range(len(decoder_block_counts)):
            stage = nn.ModuleList()
            if i > 0: # First decoder stage doesn't upsample from outside its own N
                      # It already operates on the bottleneck's N.
                      # Subsequent stages upsample to match skip connection N.
                stage.append(
                    # The SetFeaturePropagator doesn't *change* N, it enriches the higher_res_queries
                    # So, the N for the decoder stage comes from the skip connection directly.
                    SetFeaturePropagator(model_dim, num_heads, dropout)
                )
            
            for _ in range(decoder_block_counts[i]):
                stage.append(
                    DecoderTransformerBlock(model_dim, num_heads, ff_dim, dropout)
                )
            self.decoder_stages.append(stage)

        # Output Layer
        self.output_projection = nn.Linear(model_dim, output_channel_dim)

    def forward(self, x):
        # x shape: (batch_size, N_initial, input_channel_dim)
        batch_size, N_initial, _ = x.shape

        # 1. Input Embedding
        x = self.input_projection(x) # (B, N, model_dim)

        skip_connections = []
        current_N_values = [N_initial] # Track N at each level of encoder

        # 2. Encoder Path
        for i, stage_module in enumerate(self.encoder_stages):
            if isinstance(stage_module, nn.ModuleList): # This is a block of TransformerBlocks
                for block in stage_module:
                    x = block(x)
                # After transformer blocks, but before downsampling, store for skip connection
                skip_connections.append(x)
            elif isinstance(stage_module, LearnedQuerySetDownsampler): # This is a downsampler
                x = stage_module(x)
                current_N_values.append(x.shape[1]) # Update N

        # 3. Bottleneck
        x = self.bottleneck(x) # x is now the most downsampled set

        # 4. Decoder Path
        # skip_connections are popped from the end, corresponding to reversed encoder stages
        # current_N_values contains N for each skip_connection (and input N)
        
        # The last element in current_N_values is the bottleneck N.
        # The N for the next decoder stage will be N from the *previous* skip connection.

        for i, decoder_stage in enumerate(self.decoder_stages):
            if i > 0: # For subsequent decoder stages, perform feature propagation
                # Pop the *previous* skip connection (highest resolution)
                higher_res_queries = skip_connections.pop() 
                
                # The upsampler takes lower_res_features (current x) and enriches higher_res_queries
                # It does NOT change the cardinality of x directly. x's cardinality will be
                # that of higher_res_queries.
                propagator = decoder_stage[0] # First module in stage is the propagator
                x = propagator(lower_res_features=x, higher_res_queries=higher_res_queries)
                
                # Now, the subsequent TransformerBlocks in this stage operate on this new 'x'
                # which has the cardinality of higher_res_queries.
                blocks_start_idx = 1 # The rest of the stage are TransformerBlocks
            else: # First decoder stage, just operates on the bottleneck output.
                blocks_start_idx = 0
            
            # Apply decoder transformer blocks
            for block in decoder_stage[blocks_start_idx:]:
                # If there's a skip connection for this level, pass it to cross-attention
                # (This is handled within DecoderTransformerBlock now)
                if i < len(self.decoder_stages) -1 : # Not the final output stage
                     # For the last skip connection, it was popped and used for x, no more to pop
                    skip_features_for_block = skip_connections[-1] if len(skip_connections) > 0 else None
                else: # For the final output layer, there's no skip_connection left for the *next* level
                    skip_features_for_block = None
                
                # Note: DecoderTransformerBlock takes the actual skip_connection_features
                # We need to ensure that the skip_connection_features match the current 'x' in cardinality.
                # This is ensured by the design of SetFeaturePropagator making x the higher_res_queries N.
                if len(skip_connections) > 0:
                     # Get the highest-resolution remaining skip connection for the current level
                    x = block(x, skip_connections[-1])
                else: # No more skip connections left (at the highest resolution decoder stage)
                    x = block(x, x) # Pass itself for cross-attention if no skip, or handle differently.
                                    # Or, simply use a regular TransformerBlock not DecoderTransformerBlock here.
                                    # For simplicity, we'll keep it as DecoderTransformerBlock, using x for KV.
                                    # A more robust solution might have a different type of block for the last stage.
                                    pass


        # 5. Output Layer
        output = self.output_projection(x) # (B, N_initial, output_channel_dim)
        return output

# --- Example Usage ---
if __name__ == "__main__":
    batch_size = 4
    initial_N = 128 # Initial cardinality of the set
    input_channel_dim = 16 # Dimension of each element vector
    output_channel_dim = 16 # Desired output dimension for each element
    model_dim = 128 # Internal dimension for transformer
    num_heads = 8
    ff_dim = 256
    dropout = 0.1

    # Define the number of transformer blocks at each encoder/decoder stage
    # Example: [2, 2, 2] means 2 blocks in stage 1, 2 in stage 2, 2 in stage 3
    encoder_block_counts = [2, 2, 2]
    decoder_block_counts = [2, 2, 2] # The first element corresponds to bottleneck + first decoder block

    # Define the target cardinality after each downsampling step
    # E.g., 128 -> 64 -> 32 -> 16
    downsample_cardinalities = [64, 32, 16]

    # Validate that the number of downsampling steps matches encoder stages
    if len(downsample_cardinalities) != len(encoder_block_counts):
        raise ValueError("Number of downsample cardinalities must match number of encoder stages for downsampling.")
    if len(decoder_block_counts) != len(encoder_block_counts): # Usually symmetrical
         print("Warning: Number of decoder stages does not match encoder stages. Ensure your architecture handles this intended asymmetry.")

    # Create model
    model = TransformerUNetForSets(
        input_channel_dim=input_channel_dim,
        output_channel_dim=output_channel_dim,
        model_dim=model_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        encoder_block_counts=encoder_block_counts,
        decoder_block_counts=decoder_block_counts,
        downsample_cardinalities=downsample_cardinalities,
        dropout=dropout
    )

    # Create a dummy input set
    dummy_input = torch.randn(batch_size, initial_N, input_channel_dim)
    print(f"Input shape: {dummy_input.shape}")

    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Verify output cardinality matches input cardinality (for a U-Net like structure)
    assert output.shape == dummy_input.shape, "Output shape does not match input shape (B, N, C_out)"

    print("\n--- Model Summary ---")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # Test with a different N
    print("\n--- Testing with different N (requires new model instance for each N as learned_queries are fixed) ---")
    initial_N_2 = 256 # Must be compatible with downsample_cardinalities if those are fixed ratios
                      # For learned queries, we need to ensure initial_N_2 can pass through all encoders.
                      # The learned queries are fixed to `downsample_cardinalities`.
                      # The current setup means `downsample_cardinalities` dictate the *absolute* N.
    
    # If initial_N_2 changes, we would typically re-instantiate the model,
    # because `downsample_cardinalities` specify the *fixed* output N for each downsampling step.
    # If you want variable input N, you'd need to either pad/truncate or use relative downsampling factors.
    
    try:
        model_2 = TransformerUNetForSets(
            input_channel_dim=input_channel_dim,
            output_channel_dim=output_channel_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            encoder_block_counts=encoder_block_counts,
            decoder_block_counts=decoder_block_counts,
            downsample_cardinalities=[c * (initial_N_2 // initial_N) for c in downsample_cardinalities], # Scale cardinalities
            dropout=dropout
        )
        dummy_input_2 = torch.randn(batch_size, initial_N_2, input_channel_dim)
        output_2 = model_2(dummy_input_2)
        print(f"Input shape 2: {dummy_input_2.shape}")
        print(f"Output shape 2: {output_2.shape}")
        assert output_2.shape == dummy_input_2.shape
    except ValueError as e:
        print(f"Error for N={initial_N_2}: {e}. This is expected if N isn't divisible by \
              downsampling factors, or if new_N doesn't match learned query sizes.")
    except Exception as e:
        print(f"An unexpected error occurred for N={initial_N_2}: {e}")