import torch

def generate_conversion_script(file_path="conversion_script.py"):
    """
    Generates a Python script to convert original Stable Diffusion weights
    to the naming convention of the custom model provided.
    """
    
    script_content = """
import torch

def load_from_standard_weights(input_file, device):
    \"\"\"
    Loads weights from an original Stable Diffusion checkpoint and converts them
    to the naming convention of the custom model.
    \"\"\"
    original_model = torch.load(input_file, map_location=device, weights_only=False)["state_dict"]
    
    converted = {
        'diffusion': {},
        'encoder': {},
        'decoder': {},
        'clip': {}
    }

    # Time Embedding Mapping
    converted['diffusion']['time_embedding.up.weight'] = original_model['model.diffusion_model.time_embed.0.weight']
    converted['diffusion']['time_embedding.up.bias'] = original_model['model.diffusion_model.time_embed.0.bias']
    converted['diffusion']['time_embedding.out.weight'] = original_model['model.diffusion_model.time_embed.2.weight']
    converted['diffusion']['time_embedding.out.bias'] = original_model['model.diffusion_model.time_embed.2.bias']

    # UNET Input Blocks (Encoders) Mapping
    converted['diffusion']['unet.encoders.0.0.weight'] = original_model['model.diffusion_model.input_blocks.0.0.weight']
    converted['diffusion']['unet.encoders.0.0.bias'] = original_model['model.diffusion_model.input_blocks.0.0.bias']

    input_block_map = {1: 1, 2: 2, 4: 4, 5: 5, 7: 7, 8: 8, 10: 10, 11: 11}
    for i, b in input_block_map.items():
        # Residual Block
        res_block_prefix = f'diffusion.unet.encoders.{i}.0'
        orig_res_prefix = f'model.diffusion_model.input_blocks.{b}.0'
        converted[f'{res_block_prefix}.group_norm_first.weight'] = original_model[f'{orig_res_prefix}.in_layers.0.weight']
        converted[f'{res_block_prefix}.group_norm_first.bias'] = original_model[f'{orig_res_prefix}.in_layers.0.bias']
        converted[f'{res_block_prefix}.conv_first.0.weight'] = original_model[f'{orig_res_prefix}.in_layers.2.weight']
        converted[f'{res_block_prefix}.conv_first.0.bias'] = original_model[f'{orig_res_prefix}.in_layers.2.bias']
        converted[f'{res_block_prefix}.linear_time.weight'] = original_model[f'{orig_res_prefix}.emb_layers.1.weight']
        converted[f'{res_block_prefix}.linear_time.bias'] = original_model[f'{orig_res_prefix}.emb_layers.1.bias']
        converted[f'{res_block_prefix}.group_norm_merged.weight'] = original_model[f'{orig_res_prefix}.out_layers.0.weight']
        converted[f'{res_block_prefix}.group_norm_merged.bias'] = original_model[f'{orig_res_prefix}.out_layers.0.bias']
        converted[f'{res_block_prefix}.conv_merged.weight'] = original_model[f'{orig_res_prefix}.out_layers.3.weight']
        converted[f'{res_block_prefix}.conv_merged.bias'] = original_model[f'{orig_res_prefix}.out_layers.3.bias']
        if f'{orig_res_prefix}.skip_connection.weight' in original_model:
            converted[f'{res_block_prefix}.res_layer.weight'] = original_model[f'{orig_res_prefix}.skip_connection.weight']
            converted[f'{res_block_prefix}.res_layer.bias'] = original_model[f'{orig_res_prefix}.skip_connection.bias']

        # Attention Block
        attn_block_prefix = f'diffusion.unet.encoders.{i}.1'
        orig_attn_prefix = f'model.diffusion_model.input_blocks.{b}.1'
        converted[f'{attn_block_prefix}.group_norm.weight'] = original_model[f'{orig_attn_prefix}.norm.weight']
        converted[f'{attn_block_prefix}.group_norm.bias'] = original_model[f'{orig_attn_prefix}.norm.bias']
        converted[f'{attn_block_prefix}.conv_in.weight'] = original_model[f'{orig_attn_prefix}.proj_in.weight']
        converted[f'{attn_block_prefix}.conv_in.bias'] = original_model[f'{orig_attn_prefix}.proj_in.bias']
        converted[f'{attn_block_prefix}.layer_norm_1.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm1.weight']
        converted[f'{attn_block_prefix}.layer_norm_1.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm1.bias']
        converted[f'{attn_block_prefix}.attention_1.wo.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn1.to_out.0.weight']
        converted[f'{attn_block_prefix}.attention_1.wo.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn1.to_out.0.bias']
        converted[f'{attn_block_prefix}.layer_norm_2.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm2.weight']
        converted[f'{attn_block_prefix}.layer_norm_2.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm2.bias']
        converted[f'{attn_block_prefix}.attention_2.q.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn2.to_q.weight']
        converted[f'{attn_block_prefix}.attention_2.kv.weight'] = torch.cat([original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn2.to_k.weight'], original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn2.to_v.weight']], dim=0)
        converted[f'{attn_block_prefix}.attention_2.wo.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn2.to_out.0.weight']
        converted[f'{attn_block_prefix}.attention_2.wo.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn2.to_out.0.bias']
        converted[f'{attn_block_prefix}.layer_norm_3.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm3.weight']
        converted[f'{attn_block_prefix}.layer_norm_3.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm3.bias']
        converted[f'{attn_block_prefix}.linear_geglu_1.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.ff.net.0.proj.weight']
        converted[f'{attn_block_prefix}.linear_geglu_1.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.ff.net.0.proj.bias']
        converted[f'{attn_block_prefix}.linear_geglu_2.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.ff.net.2.weight']
        converted[f'{attn_block_prefix}.linear_geglu_2.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.ff.net.2.bias']
        converted[f'{attn_block_prefix}.conv_out.weight'] = original_model[f'{orig_attn_prefix}.proj_out.weight']
        converted[f'{attn_block_prefix}.conv_out.bias'] = original_model[f'{orig_attn_prefix}.proj_out.bias']

    # Downsampling layers
    downsample_map = {3: 3, 6: 6, 9: 9}
    for i, b in downsample_map.items():
        converted[f'diffusion.unet.encoders.{i}.0.weight'] = original_model[f'model.diffusion_model.input_blocks.{b}.0.op.weight']
        converted[f'diffusion.unet.encoders.{i}.0.bias'] = original_model[f'model.diffusion_model.input_blocks.{b}.0.op.bias']

    # UNET Middle Block Mapping
    # ResBlock 1
    converted['diffusion.unet.bottom.0.group_norm_first.weight'] = original_model['model.diffusion_model.middle_block.0.in_layers.0.weight']
    converted['diffusion.unet.bottom.0.group_norm_first.bias'] = original_model['model.diffusion_model.middle_block.0.in_layers.0.bias']
    converted['diffusion.unet.bottom.0.conv_first.0.weight'] = original_model['model.diffusion_model.middle_block.0.in_layers.2.weight']
    converted['diffusion.unet.bottom.0.conv_first.0.bias'] = original_model['model.diffusion_model.middle_block.0.in_layers.2.bias']
    converted['diffusion.unet.bottom.0.linear_time.weight'] = original_model['model.diffusion_model.middle_block.0.emb_layers.1.weight']
    converted['diffusion.unet.bottom.0.linear_time.bias'] = original_model['model.diffusion_model.middle_block.0.emb_layers.1.bias']
    converted['diffusion.unet.bottom.0.group_norm_merged.weight'] = original_model['model.diffusion_model.middle_block.0.out_layers.0.weight']
    converted['diffusion.unet.bottom.0.group_norm_merged.bias'] = original_model['model.diffusion_model.middle_block.0.out_layers.0.bias']
    converted['diffusion.unet.bottom.0.conv_merged.weight'] = original_model['model.diffusion_model.middle_block.0.out_layers.3.weight']
    converted['diffusion.unet.bottom.0.conv_merged.bias'] = original_model['model.diffusion_model.middle_block.0.out_layers.3.bias']
    # Attention Block
    converted['diffusion.unet.bottom.1.group_norm.weight'] = original_model['model.diffusion_model.middle_block.1.norm.weight']
    converted['diffusion.unet.bottom.1.group_norm.bias'] = original_model['model.diffusion_model.middle_block.1.norm.bias']
    converted['diffusion.unet.bottom.1.conv_in.weight'] = original_model['model.diffusion_model.middle_block.1.proj_in.weight']
    converted['diffusion.unet.bottom.1.conv_in.bias'] = original_model['model.diffusion_model.middle_block.1.proj_in.bias']
    converted['diffusion.unet.bottom.1.layer_norm_1.weight'] = original_model['model.diffusion_model.middle_block.1.transformer_blocks.0.norm1.weight']
    converted['diffusion.unet.bottom.1.layer_norm_1.bias'] = original_model['model.diffusion_model.middle_block.1.transformer_blocks.0.norm1.bias']
    converted['diffusion.unet.bottom.1.attention_1.wo.weight'] = original_model['model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight']
    converted['diffusion.unet.bottom.1.attention_1.wo.bias'] = original_model['model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.bias']
    converted['diffusion.unet.bottom.1.layer_norm_2.weight'] = original_model['model.diffusion_model.middle_block.1.transformer_blocks.0.norm2.weight']
    converted['diffusion.unet.bottom.1.layer_norm_2.bias'] = original_model['model.diffusion_model.middle_block.1.transformer_blocks.0.norm2.bias']
    converted['diffusion.unet.bottom.1.attention_2.q.weight'] = original_model['model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_q.weight']
    converted['diffusion.unet.bottom.1.attention_2.kv.weight'] = torch.cat([original_model['model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight'], original_model['model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_v.weight']], dim=0)
    converted['diffusion.unet.bottom.1.attention_2.wo.weight'] = original_model['model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_out.0.weight']
    converted['diffusion.unet.bottom.1.attention_2.wo.bias'] = original_model['model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_out.0.bias']
    converted['diffusion.unet.bottom.1.layer_norm_3.weight'] = original_model['model.diffusion_model.middle_block.1.transformer_blocks.0.norm3.weight']
    converted['diffusion.unet.bottom.1.layer_norm_3.bias'] = original_model['model.diffusion_model.middle_block.1.transformer_blocks.0.norm3.bias']
    converted['diffusion.unet.bottom.1.linear_geglu_1.weight'] = original_model['model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.0.proj.weight']
    converted['diffusion.unet.bottom.1.linear_geglu_1.bias'] = original_model['model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.0.proj.bias']
    converted['diffusion.unet.bottom.1.linear_geglu_2.weight'] = original_model['model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.2.weight']
    converted['diffusion.unet.bottom.1.linear_geglu_2.bias'] = original_model['model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.2.bias']
    converted['diffusion.unet.bottom.1.conv_out.weight'] = original_model['model.diffusion_model.middle_block.1.proj_out.weight']
    converted['diffusion.unet.bottom.1.conv_out.bias'] = original_model['model.diffusion_model.middle_block.1.proj_out.bias']
    # ResBlock 2
    converted['diffusion.unet.bottom.2.group_norm_first.weight'] = original_model['model.diffusion_model.middle_block.2.in_layers.0.weight']
    converted['diffusion.unet.bottom.2.group_norm_first.bias'] = original_model['model.diffusion_model.middle_block.2.in_layers.0.bias']
    converted['diffusion.unet.bottom.2.conv_first.0.weight'] = original_model['model.diffusion_model.middle_block.2.in_layers.2.weight']
    converted['diffusion.unet.bottom.2.conv_first.0.bias'] = original_model['model.diffusion_model.middle_block.2.in_layers.2.bias']
    converted['diffusion.unet.bottom.2.linear_time.weight'] = original_model['model.diffusion_model.middle_block.2.emb_layers.1.weight']
    converted['diffusion.unet.bottom.2.linear_time.bias'] = original_model['model.diffusion_model.middle_block.2.emb_layers.1.bias']
    converted['diffusion.unet.bottom.2.group_norm_merged.weight'] = original_model['model.diffusion_model.middle_block.2.out_layers.0.weight']
    converted['diffusion.unet.bottom.2.group_norm_merged.bias'] = original_model['model.diffusion_model.middle_block.2.out_layers.0.bias']
    converted['diffusion.unet.bottom.2.conv_merged.weight'] = original_model['model.diffusion_model.middle_block.2.out_layers.3.weight']
    converted['diffusion.unet.bottom.2.conv_merged.bias'] = original_model['model.diffusion_model.middle_block.2.out_layers.3.bias']

    # UNET Output Blocks (Decoders) Mapping
    output_block_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11}
    for i, b in output_block_map.items():
        res_block_prefix = f'diffusion.unet.decoders.{i}.0'
        orig_res_prefix = f'model.diffusion_model.output_blocks.{b}.0'
        converted[f'{res_block_prefix}.group_norm_first.weight'] = original_model[f'{orig_res_prefix}.in_layers.0.weight']
        converted[f'{res_block_prefix}.group_norm_first.bias'] = original_model[f'{orig_res_prefix}.in_layers.0.bias']
        converted[f'{res_block_prefix}.conv_first.0.weight'] = original_model[f'{orig_res_prefix}.in_layers.2.weight']
        converted[f'{res_block_prefix}.conv_first.0.bias'] = original_model[f'{orig_res_prefix}.in_layers.2.bias']
        converted[f'{res_block_prefix}.linear_time.weight'] = original_model[f'{orig_res_prefix}.emb_layers.1.weight']
        converted[f'{res_block_prefix}.linear_time.bias'] = original_model[f'{orig_res_prefix}.emb_layers.1.bias']
        converted[f'{res_block_prefix}.group_norm_merged.weight'] = original_model[f'{orig_res_prefix}.out_layers.0.weight']
        converted[f'{res_block_prefix}.group_norm_merged.bias'] = original_model[f'{orig_res_prefix}.out_layers.0.bias']
        converted[f'{res_block_prefix}.conv_merged.weight'] = original_model[f'{orig_res_prefix}.out_layers.3.weight']
        converted[f'{res_block_prefix}.conv_merged.bias'] = original_model[f'{orig_res_prefix}.out_layers.3.bias']
        if f'{orig_res_prefix}.skip_connection.weight' in original_model:
            converted[f'{res_block_prefix}.res_layer.weight'] = original_model[f'{orig_res_prefix}.skip_connection.weight']
            converted[f'{res_block_prefix}.res_layer.bias'] = original_model[f'{orig_res_prefix}.skip_connection.bias']

        if len(original_model[f'model.diffusion_model.output_blocks.{b}']) > 1 and isinstance(original_model[f'model.diffusion_model.output_blocks.{b}'][1], torch.nn.Module): # Check if attention block exists
            attn_block_prefix = f'diffusion.unet.decoders.{i}.1'
            orig_attn_prefix = f'model.diffusion_model.output_blocks.{b}.1'
            converted[f'{attn_block_prefix}.group_norm.weight'] = original_model[f'{orig_attn_prefix}.norm.weight']
            converted[f'{attn_block_prefix}.group_norm.bias'] = original_model[f'{orig_attn_prefix}.norm.bias']
            converted[f'{attn_block_prefix}.conv_in.weight'] = original_model[f'{orig_attn_prefix}.proj_in.weight']
            converted[f'{attn_block_prefix}.conv_in.bias'] = original_model[f'{orig_attn_prefix}.proj_in.bias']
            converted[f'{attn_block_prefix}.layer_norm_1.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm1.weight']
            converted[f'{attn_block_prefix}.layer_norm_1.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm1.bias']
            converted[f'{attn_block_prefix}.attention_1.wo.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn1.to_out.0.weight']
            converted[f'{attn_block_prefix}.attention_1.wo.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn1.to_out.0.bias']
            converted[f'{attn_block_prefix}.layer_norm_2.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm2.weight']
            converted[f'{attn_block_prefix}.layer_norm_2.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm2.bias']
            converted[f'{attn_block_prefix}.attention_2.q.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn2.to_q.weight']
            converted[f'{attn_block_prefix}.attention_2.kv.weight'] = torch.cat([original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn2.to_k.weight'], original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn2.to_v.weight']], dim=0)
            converted[f'{attn_block_prefix}.attention_2.wo.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn2.to_out.0.weight']
            converted[f'{attn_block_prefix}.attention_2.wo.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn2.to_out.0.bias']
            converted[f'{attn_block_prefix}.layer_norm_3.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm3.weight']
            converted[f'{attn_block_prefix}.layer_norm_3.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm3.bias']
            converted[f'{attn_block_prefix}.linear_geglu_1.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.ff.net.0.proj.weight']
            converted[f'{attn_block_prefix}.linear_geglu_1.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.ff.net.0.proj.bias']
            converted[f'{attn_block_prefix}.linear_geglu_2.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.ff.net.2.weight']
            converted[f'{attn_block_prefix}.linear_geglu_2.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.ff.net.2.bias']
            converted[f'{attn_block_prefix}.conv_out.weight'] = original_model[f'{orig_attn_prefix}.proj_out.weight']
            converted[f'{attn_block_prefix}.conv_out.bias'] = original_model[f'{orig_attn_prefix}.proj_out.bias']

    # UNET Final Layer Mapping
    converted['diffusion.final.norm.weight'] = original_model['model.diffusion_model.out.0.weight']
    converted['diffusion.final.norm.bias'] = original_model['model.diffusion_model.out.0.bias']
    converted['diffusion.final.conv.weight'] = original_model['model.diffusion_model.out.2.weight']
    converted['diffusion.final.conv.bias'] = original_model['model.diffusion_model.out.2.bias']

    # VAE Encoder Mapping
    converted['encoder.layers.0.weight'] = original_model['first_stage_model.encoder.conv_in.weight']
    converted['encoder.layers.0.bias'] = original_model['first_stage_model.encoder.conv_in.bias']
    # ... Continue mapping for all encoder layers based on their sequential order
    # This part requires manually matching your flat structure to the original hierarchical one.
    # Example for first residual block:
    converted['encoder.layers.1.main.0.weight'] = original_model['first_stage_model.encoder.down.0.block.0.norm1.weight']
    # ... etc for the entire VAE encoder

    # VAE Decoder Mapping
    converted['decoder.layers.0.weight'] = original_model['first_stage_model.post_quant_conv.weight']
    converted['decoder.layers.0.bias'] = original_model['first_stage_model.post_quant_conv.bias']
    # ... Continue mapping for all decoder layers
    # Example for a mid block:
    converted['decoder.layers.2.main.0.weight'] = original_model['first_stage_model.decoder.mid.block_1.norm1.weight']
    # ... etc for the entire VAE decoder

    # CLIP Model Mapping
    converted['clip.embedding.weight'] = original_model['cond_stage_model.transformer.text_model.embeddings.token_embedding.weight']
    converted['clip.pos_embedding'] = original_model['cond_stage_model.transformer.text_model.embeddings.position_embedding.weight']
    
    for i in range(12):
        clip_layer_prefix = f'clip.layers.0.{i}'
        orig_clip_prefix = f'cond_stage_model.transformer.text_model.encoder.layers.{i}'
        
        converted[f'{clip_layer_prefix}.norm_1.weight'] = original_model[f'{orig_clip_prefix}.layer_norm1.weight']
        converted[f'{clip_layer_prefix}.norm_1.bias'] = original_model[f'{orig_clip_prefix}.layer_norm1.bias']

        q_w = original_model[f'{orig_clip_prefix}.self_attn.q_proj.weight']
        k_w = original_model[f'{orig_clip_prefix}.self_attn.k_proj.weight']
        v_w = original_model[f'{orig_clip_prefix}.self_attn.v_proj.weight']
        converted[f'{clip_layer_prefix}.attention.qkv.weight'] = torch.cat([q_w, k_w, v_w], dim=0)
        
        converted[f'{clip_layer_prefix}.attention.wo.weight'] = original_model[f'{orig_clip_prefix}.self_attn.out_proj.weight']
        converted[f'{clip_layer_prefix}.attention.wo.bias'] = original_model[f'{orig_clip_prefix}.self_attn.out_proj.bias']

        converted[f'{clip_layer_prefix}.norm_2.weight'] = original_model[f'{orig_clip_prefix}.layer_norm2.weight']
        converted[f'{clip_layer_prefix}.norm_2.bias'] = original_model[f'{orig_clip_prefix}.layer_norm2.bias']
        converted[f'{clip_layer_prefix}.up.weight'] = original_model[f'{orig_clip_prefix}.mlp.fc1.weight']
        converted[f'{clip_layer_prefix}.up.bias'] = original_model[f'{orig_clip_prefix}.mlp.fc1.bias']
        converted[f'{clip_layer_prefix}.down.weight'] = original_model[f'{orig_clip_prefix}.mlp.fc2.weight']
        converted[f'{clip_layer_prefix}.down.bias'] = original_model[f'{orig_clip_prefix}.mlp.fc2.bias']

    converted['clip.layernorm.weight'] = original_model['cond_stage_model.transformer.text_model.final_layer_norm.weight']
    converted['clip.layernorm.bias'] = original_model['cond_stage_model.transformer.text_model.final_layer_norm.bias']

    return converted
    """

    try:
        with open(file_path, "w") as f:
            f.write(script_content)
        print(f"Successfully generated conversion script at: {file_path}")
    except IOError as e:
        print(f"Error writing to file {file_path}: {e}")

if __name__ == '__main__':
    generate_conversion_script()