
import torch

def map_vae_residual_block(converted, custom_prefix, orig_prefix, original_model):

    converted[f'{custom_prefix}.main.0.weight'] = original_model[f'{orig_prefix}.norm1.weight']
    converted[f'{custom_prefix}.main.0.bias'] = original_model[f'{orig_prefix}.norm1.bias']
    converted[f'{custom_prefix}.main.2.weight'] = original_model[f'{orig_prefix}.conv1.weight']
    converted[f'{custom_prefix}.main.2.bias'] = original_model[f'{orig_prefix}.conv1.bias']
    converted[f'{custom_prefix}.main.3.weight'] = original_model[f'{orig_prefix}.norm2.weight']
    converted[f'{custom_prefix}.main.3.bias'] = original_model[f'{orig_prefix}.norm2.bias']
    converted[f'{custom_prefix}.main.5.weight'] = original_model[f'{orig_prefix}.conv2.weight']
    converted[f'{custom_prefix}.main.5.bias'] = original_model[f'{orig_prefix}.conv2.bias']
    
    if f'{orig_prefix}.nin_shortcut.weight' in original_model:
        converted[f'{custom_prefix}.res_layer.weight'] = original_model[f'{orig_prefix}.nin_shortcut.weight']
        converted[f'{custom_prefix}.res_layer.bias'] = original_model[f'{orig_prefix}.nin_shortcut.bias']

def load_from_standard_weights(input_file, device):
    original_model = torch.load(input_file, map_location=device, weights_only=False)["state_dict"]
    
    converted = {'diffusion': {}, 'encoder': {}, 'decoder': {}, 'clip': {}}

#================================================================================#
    #                              CLIP Model Mapping                                #
    #================================================================================#
    clip_converted = converted['clip']
    clip_converted['embedding.weight'] = original_model['cond_stage_model.transformer.text_model.embeddings.token_embedding.weight']
    clip_converted['pos_embedding'] = original_model['cond_stage_model.transformer.text_model.embeddings.position_embedding.weight']
    
    for i in range(12):
        # Correct prefix, relative to the CLIP model itself
        layer_prefix = f'layers.{i}'
        orig_prefix = f'cond_stage_model.transformer.text_model.encoder.layers.{i}'
        
        clip_converted[f'{layer_prefix}.norm_1.weight'] = original_model[f'{orig_prefix}.layer_norm1.weight']
        clip_converted[f'{layer_prefix}.norm_1.bias'] = original_model[f'{orig_prefix}.layer_norm1.bias']

        q_w = original_model[f'{orig_prefix}.self_attn.q_proj.weight']
        k_w = original_model[f'{orig_prefix}.self_attn.k_proj.weight']
        v_w = original_model[f'{orig_prefix}.self_attn.v_proj.weight']
        clip_converted[f'{layer_prefix}.attention.qkv.weight'] = torch.cat([q_w, k_w, v_w])
        
        q_b = original_model[f'{orig_prefix}.self_attn.q_proj.bias']
        k_b = original_model[f'{orig_prefix}.self_attn.k_proj.bias']
        v_b = original_model[f'{orig_prefix}.self_attn.v_proj.bias']
        clip_converted[f'{layer_prefix}.attention.qkv.bias'] = torch.cat([q_b, k_b, v_b])
        
        clip_converted[f'{layer_prefix}.attention.wo.weight'] = original_model[f'{orig_prefix}.self_attn.out_proj.weight']

        clip_converted[f'{layer_prefix}.attention.wo.bias'] = original_model[f'{orig_prefix}.self_attn.out_proj.bias']
        
        clip_converted[f'{layer_prefix}.attention.wo.weight'] = original_model[f'{orig_prefix}.self_attn.out_proj.weight']

        clip_converted[f'{layer_prefix}.norm_2.weight'] = original_model[f'{orig_prefix}.layer_norm2.weight']
        clip_converted[f'{layer_prefix}.norm_2.bias'] = original_model[f'{orig_prefix}.layer_norm2.bias']
        clip_converted[f'{layer_prefix}.up.weight'] = original_model[f'{orig_prefix}.mlp.fc1.weight']
        clip_converted[f'{layer_prefix}.up.bias'] = original_model[f'{orig_prefix}.mlp.fc1.bias']
        clip_converted[f'{layer_prefix}.down.weight'] = original_model[f'{orig_prefix}.mlp.fc2.weight']
        clip_converted[f'{layer_prefix}.down.bias'] = original_model[f'{orig_prefix}.mlp.fc2.bias']

    clip_converted['layernorm.weight'] = original_model['cond_stage_model.transformer.text_model.final_layer_norm.weight']
    clip_converted['layernorm.bias'] = original_model['cond_stage_model.transformer.text_model.final_layer_norm.bias']

    #================================================================================#
    #                             VAE Encoder Mapping                                #
    #================================================================================#
    enc = converted['encoder']
    o_enc = 'first_stage_model.encoder'
    
    enc['layers.0.weight'] = original_model[f'{o_enc}.conv_in.weight']
    enc['layers.0.bias'] = original_model[f'{o_enc}.conv_in.bias']

    map_vae_residual_block(enc, 'layers.1', f'{o_enc}.down.0.block.0', original_model)
    map_vae_residual_block(enc, 'layers.2', f'{o_enc}.down.0.block.1', original_model)
    enc['layers.3.weight'] = original_model[f'{o_enc}.down.0.downsample.conv.weight']
    enc['layers.3.bias'] = original_model[f'{o_enc}.down.0.downsample.conv.bias']

    map_vae_residual_block(enc, 'layers.4', f'{o_enc}.down.1.block.0', original_model)
    map_vae_residual_block(enc, 'layers.5', f'{o_enc}.down.1.block.1', original_model)
    enc['layers.6.weight'] = original_model[f'{o_enc}.down.1.downsample.conv.weight']
    enc['layers.6.bias'] = original_model[f'{o_enc}.down.1.downsample.conv.bias']

    map_vae_residual_block(enc, 'layers.7', f'{o_enc}.down.2.block.0', original_model)
    map_vae_residual_block(enc, 'layers.8', f'{o_enc}.down.2.block.1', original_model)
    enc['layers.9.weight'] = original_model[f'{o_enc}.down.2.downsample.conv.weight']
    enc['layers.9.bias'] = original_model[f'{o_enc}.down.2.downsample.conv.bias']

    map_vae_residual_block(enc, 'layers.10', f'{o_enc}.down.3.block.0', original_model)
    map_vae_residual_block(enc, 'layers.11', f'{o_enc}.down.3.block.1', original_model)

    map_vae_residual_block(enc, 'layers.12', f'{o_enc}.mid.block_1', original_model)
    
    # Middle Attention - FIXED: Squeeze conv weights to fit linear layers
    enc['layers.13.group_norm.weight'] = original_model[f'{o_enc}.mid.attn_1.norm.weight']
    enc['layers.13.group_norm.bias'] = original_model[f'{o_enc}.mid.attn_1.norm.bias']
    q_w = original_model[f'{o_enc}.mid.attn_1.q.weight'].squeeze()
    k_w = original_model[f'{o_enc}.mid.attn_1.k.weight'].squeeze()
    v_w = original_model[f'{o_enc}.mid.attn_1.v.weight'].squeeze()
    enc['layers.13.attention.qkv.weight'] = torch.cat([q_w, k_w, v_w])
    enc['layers.13.attention.wo.weight'] = original_model[f'{o_enc}.mid.attn_1.proj_out.weight'].squeeze()

    q_b = original_model[f'{o_enc}.mid.attn_1.q.bias']
    k_b = original_model[f'{o_enc}.mid.attn_1.k.bias']
    v_b = original_model[f'{o_enc}.mid.attn_1.v.bias']
    enc['layers.13.attention.qkv.bias'] = torch.cat([q_b, k_b, v_b])
    enc['layers.13.attention.wo.bias'] = original_model[f'{o_enc}.mid.attn_1.proj_out.bias']

    map_vae_residual_block(enc, 'layers.14', f'{o_enc}.mid.block_2', original_model)

    # Final Layers - FIXED: Correctly map layers 15, 17, and 18, skipping 16 (SiLU)
    enc['layers.15.weight'] = original_model[f'{o_enc}.norm_out.weight']
    enc['layers.15.bias'] = original_model[f'{o_enc}.norm_out.bias']
    # Layer 16 in encoder.py is SiLU, which has no weights, so we skip it.
    enc['layers.17.weight'] = original_model[f'{o_enc}.conv_out.weight']
    enc['layers.17.bias'] = original_model[f'{o_enc}.conv_out.bias']

    enc['layers.18.weight'] = original_model['first_stage_model.quant_conv.weight']
    enc['layers.18.bias'] = original_model['first_stage_model.quant_conv.bias']

#================================================================================#
    #                             VAE Decoder Mapping                                #
    #================================================================================#
    dec = converted['decoder']
    o_dec = 'first_stage_model.decoder'

    # Initial layers
    dec['layers.0.weight'] = original_model['first_stage_model.post_quant_conv.weight']
    dec['layers.0.bias'] = original_model['first_stage_model.post_quant_conv.bias']
    dec['layers.1.weight'] = original_model[f'{o_dec}.conv_in.weight']
    dec['layers.1.bias'] = original_model[f'{o_dec}.conv_in.bias']

    # Middle Block
    map_vae_residual_block(dec, 'layers.2', f'{o_dec}.mid.block_1', original_model)
    # Middle Attention
    dec['layers.3.group_norm.weight'] = original_model[f'{o_dec}.mid.attn_1.norm.weight']
    dec['layers.3.group_norm.bias'] = original_model[f'{o_dec}.mid.attn_1.norm.bias']
    q_w = original_model[f'{o_dec}.mid.attn_1.q.weight'].squeeze()
    k_w = original_model[f'{o_dec}.mid.attn_1.k.weight'].squeeze()
    v_w = original_model[f'{o_dec}.mid.attn_1.v.weight'].squeeze()
    dec['layers.3.attention.qkv.weight'] = torch.cat([q_w, k_w, v_w])
    dec['layers.3.attention.wo.weight'] = original_model[f'{o_dec}.mid.attn_1.proj_out.weight'].squeeze()
    q_b = original_model[f'{o_dec}.mid.attn_1.q.bias']
    k_b = original_model[f'{o_dec}.mid.attn_1.k.bias']
    v_b = original_model[f'{o_dec}.mid.attn_1.v.bias']
    dec['layers.3.attention.qkv.bias'] = torch.cat([q_b, k_b, v_b])
    dec['layers.3.attention.wo.bias'] = original_model[f'{o_dec}.mid.attn_1.proj_out.bias']
    map_vae_residual_block(dec, 'layers.4', f'{o_dec}.mid.block_2', original_model)

    # Up Block 1 (Corresponds to original's up.3)
    map_vae_residual_block(dec, 'layers.5', f'{o_dec}.up.3.block.0', original_model)
    map_vae_residual_block(dec, 'layers.6', f'{o_dec}.up.3.block.1', original_model)
    map_vae_residual_block(dec, 'layers.7', f'{o_dec}.up.3.block.2', original_model)
    # layers.8 in your model is Upsample, which has no weights.
    dec['layers.9.weight'] = original_model[f'{o_dec}.up.3.upsample.conv.weight']
    dec['layers.9.bias'] = original_model[f'{o_dec}.up.3.upsample.conv.bias']

    # Up Block 2 (Corresponds to original's up.2)
    map_vae_residual_block(dec, 'layers.10', f'{o_dec}.up.2.block.0', original_model)
    map_vae_residual_block(dec, 'layers.11', f'{o_dec}.up.2.block.1', original_model)
    map_vae_residual_block(dec, 'layers.12', f'{o_dec}.up.2.block.2', original_model)
    # layers.13 is Upsample
    dec['layers.14.weight'] = original_model[f'{o_dec}.up.2.upsample.conv.weight']
    dec['layers.14.bias'] = original_model[f'{o_dec}.up.2.upsample.conv.bias']

    # Up Block 3 (Corresponds to original's up.1)
    map_vae_residual_block(dec, 'layers.15', f'{o_dec}.up.1.block.0', original_model)
    map_vae_residual_block(dec, 'layers.16', f'{o_dec}.up.1.block.1', original_model)
    map_vae_residual_block(dec, 'layers.17', f'{o_dec}.up.1.block.2', original_model)
    # layers.18 is Upsample
    dec['layers.19.weight'] = original_model[f'{o_dec}.up.1.upsample.conv.weight']
    dec['layers.19.bias'] = original_model[f'{o_dec}.up.1.upsample.conv.bias']

    # Up Block 4 (Corresponds to original's up.0)
    map_vae_residual_block(dec, 'layers.20', f'{o_dec}.up.0.block.0', original_model)
    map_vae_residual_block(dec, 'layers.21', f'{o_dec}.up.0.block.1', original_model)
    map_vae_residual_block(dec, 'layers.22', f'{o_dec}.up.0.block.2', original_model)
    
    # Final Layers
    dec['layers.23.weight'] = original_model[f'{o_dec}.norm_out.weight']
    dec['layers.23.bias'] = original_model[f'{o_dec}.norm_out.bias']
    # Layer 24 is SiLU
    dec['layers.25.weight'] = original_model[f'{o_dec}.conv_out.weight']
    dec['layers.25.bias'] = original_model[f'{o_dec}.conv_out.bias']
    # Layer 26 is Tanh
    
    #================================================================================#
    #                           Diffusion U-Net Mapping                              #
    #================================================================================#
    diff = converted['diffusion']
    o_diff = 'model.diffusion_model'

    diff['time_embedding.up.weight'] = original_model[f'{o_diff}.time_embed.0.weight']
    diff['time_embedding.up.bias'] = original_model[f'{o_diff}.time_embed.0.bias']
    diff['time_embedding.out.weight'] = original_model[f'{o_diff}.time_embed.2.weight']
    diff['time_embedding.out.bias'] = original_model[f'{o_diff}.time_embed.2.bias']

    diff['unet.encoders.0.0.weight'] = original_model[f'{o_diff}.input_blocks.0.0.weight']
    diff['unet.encoders.0.0.bias'] = original_model[f'{o_diff}.input_blocks.0.0.bias']
    
    encoder_block_indices = [1, 2, 4, 5, 7, 8, 10, 11]
    original_input_blocks = [1, 2, 4, 5, 7, 8, 10, 11]

    for i, b in zip(encoder_block_indices, original_input_blocks):
        res_prefix = f'unet.encoders.{i}.0'
        orig_res_prefix = f'{o_diff}.input_blocks.{b}.0'
        diff[f'{res_prefix}.group_norm_first.weight'] = original_model[f'{orig_res_prefix}.in_layers.0.weight']
        diff[f'{res_prefix}.group_norm_first.bias'] = original_model[f'{orig_res_prefix}.in_layers.0.bias']
        diff[f'{res_prefix}.conv_first.weight'] = original_model[f'{orig_res_prefix}.in_layers.2.weight']
        diff[f'{res_prefix}.conv_first.bias'] = original_model[f'{orig_res_prefix}.in_layers.2.bias']
        diff[f'{res_prefix}.linear_time.weight'] = original_model[f'{orig_res_prefix}.emb_layers.1.weight']
        diff[f'{res_prefix}.linear_time.bias'] = original_model[f'{orig_res_prefix}.emb_layers.1.bias']
        diff[f'{res_prefix}.group_norm_merged.weight'] = original_model[f'{orig_res_prefix}.out_layers.0.weight']
        diff[f'{res_prefix}.group_norm_merged.bias'] = original_model[f'{orig_res_prefix}.out_layers.0.bias']
        diff[f'{res_prefix}.conv_merged.weight'] = original_model[f'{orig_res_prefix}.out_layers.3.weight']
        diff[f'{res_prefix}.conv_merged.bias'] = original_model[f'{orig_res_prefix}.out_layers.3.bias']
        if f'{orig_res_prefix}.skip_connection.weight' in original_model:
            diff[f'{res_prefix}.res_layer.weight'] = original_model[f'{orig_res_prefix}.skip_connection.weight']
            diff[f'{res_prefix}.res_layer.bias'] = original_model[f'{orig_res_prefix}.skip_connection.bias']

        orig_attn_prefix = f'{o_diff}.input_blocks.{b}.1'
        if f'{orig_attn_prefix}.norm.weight' in original_model:
            attn_prefix = f'unet.encoders.{i}.1'
            diff[f'{attn_prefix}.group_norm.weight'] = original_model[f'{orig_attn_prefix}.norm.weight']
            diff[f'{attn_prefix}.group_norm.bias'] = original_model[f'{orig_attn_prefix}.norm.bias']
            diff[f'{attn_prefix}.conv_in.weight'] = original_model[f'{orig_attn_prefix}.proj_in.weight']
            diff[f'{attn_prefix}.conv_in.bias'] = original_model[f'{orig_attn_prefix}.proj_in.bias']
            diff[f'{attn_prefix}.layer_norm_1.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm1.weight']
            diff[f'{attn_prefix}.layer_norm_1.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm1.bias']
            
            q_w, k_w, v_w = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn1.to_q.weight'], original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn1.to_k.weight'], original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn1.to_v.weight']
            diff[f'{attn_prefix}.attention_1.qkv.weight'] = torch.cat([q_w, k_w, v_w])
            diff[f'{attn_prefix}.attention_1.wo.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn1.to_out.0.weight']
            diff[f'{attn_prefix}.attention_1.wo.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn1.to_out.0.bias']
            
            diff[f'{attn_prefix}.layer_norm_2.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm2.weight']
            diff[f'{attn_prefix}.layer_norm_2.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm2.bias']
            
            diff[f'{attn_prefix}.attention_2.q.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn2.to_q.weight']
            k_w_c, v_w_c = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn2.to_k.weight'], original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn2.to_v.weight']
            diff[f'{attn_prefix}.attention_2.kv.weight'] = torch.cat([k_w_c, v_w_c])
            diff[f'{attn_prefix}.attention_2.wo.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn2.to_out.0.weight']
            diff[f'{attn_prefix}.attention_2.wo.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn2.to_out.0.bias']

            diff[f'{attn_prefix}.layer_norm_3.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm3.weight']
            diff[f'{attn_prefix}.layer_norm_3.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm3.bias']
            diff[f'{attn_prefix}.linear_geglu_1.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.ff.net.0.proj.weight']
            diff[f'{attn_prefix}.linear_geglu_1.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.ff.net.0.proj.bias']
            diff[f'{attn_prefix}.linear_geglu_2.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.ff.net.2.weight']
            diff[f'{attn_prefix}.linear_geglu_2.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.ff.net.2.bias']
            diff[f'{attn_prefix}.conv_out.weight'] = original_model[f'{orig_attn_prefix}.proj_out.weight']
            diff[f'{attn_prefix}.conv_out.bias'] = original_model[f'{orig_attn_prefix}.proj_out.bias']

    downsample_map = {3: 3, 6: 6, 9: 9} 
    for i, b in downsample_map.items():
        diff[f'unet.encoders.{i}.0.weight'] = original_model[f'{o_diff}.input_blocks.{b}.0.op.weight']
        diff[f'unet.encoders.{i}.0.bias'] = original_model[f'{o_diff}.input_blocks.{b}.0.op.bias']

    #================================================================================#
    #                           Diffusion U-Net Middle Block                         #
    #================================================================================#
    
    # Middle Block - First Residual
    res_prefix = 'unet.bottom.0'
    orig_res_prefix = f'{o_diff}.middle_block.0'
    diff[f'{res_prefix}.group_norm_first.weight'] = original_model[f'{orig_res_prefix}.in_layers.0.weight']
    diff[f'{res_prefix}.group_norm_first.bias'] = original_model[f'{orig_res_prefix}.in_layers.0.bias']
    diff[f'{res_prefix}.conv_first.weight'] = original_model[f'{orig_res_prefix}.in_layers.2.weight'] # Note: No ".0" here
    diff[f'{res_prefix}.conv_first.bias'] = original_model[f'{orig_res_prefix}.in_layers.2.bias']
    diff[f'{res_prefix}.linear_time.weight'] = original_model[f'{orig_res_prefix}.emb_layers.1.weight']
    diff[f'{res_prefix}.linear_time.bias'] = original_model[f'{orig_res_prefix}.emb_layers.1.bias']
    diff[f'{res_prefix}.group_norm_merged.weight'] = original_model[f'{orig_res_prefix}.out_layers.0.weight']
    diff[f'{res_prefix}.group_norm_merged.bias'] = original_model[f'{orig_res_prefix}.out_layers.0.bias']
    diff[f'{res_prefix}.conv_merged.weight'] = original_model[f'{orig_res_prefix}.out_layers.3.weight']
    diff[f'{res_prefix}.conv_merged.bias'] = original_model[f'{orig_res_prefix}.out_layers.3.bias']

    # Middle Block - Attention
    attn_prefix = 'unet.bottom.1'
    orig_attn_prefix = f'{o_diff}.middle_block.1'
    diff[f'{attn_prefix}.group_norm.weight'] = original_model[f'{orig_attn_prefix}.norm.weight']
    diff[f'{attn_prefix}.group_norm.bias'] = original_model[f'{orig_attn_prefix}.norm.bias']
    diff[f'{attn_prefix}.conv_in.weight'] = original_model[f'{orig_attn_prefix}.proj_in.weight']
    diff[f'{attn_prefix}.conv_in.bias'] = original_model[f'{orig_attn_prefix}.proj_in.bias']
    diff[f'{attn_prefix}.layer_norm_1.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm1.weight']
    diff[f'{attn_prefix}.layer_norm_1.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm1.bias']
    q_w, k_w, v_w = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn1.to_q.weight'], original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn1.to_k.weight'], original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn1.to_v.weight']
    diff[f'{attn_prefix}.attention_1.qkv.weight'] = torch.cat([q_w, k_w, v_w])
    diff[f'{attn_prefix}.attention_1.wo.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn1.to_out.0.weight']
    diff[f'{attn_prefix}.attention_1.wo.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn1.to_out.0.bias']
    diff[f'{attn_prefix}.layer_norm_2.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm2.weight']
    diff[f'{attn_prefix}.layer_norm_2.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm2.bias']
    diff[f'{attn_prefix}.attention_2.q.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn2.to_q.weight']
    k_w_c, v_w_c = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn2.to_k.weight'], original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn2.to_v.weight']
    diff[f'{attn_prefix}.attention_2.kv.weight'] = torch.cat([k_w_c, v_w_c])
    diff[f'{attn_prefix}.attention_2.wo.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn2.to_out.0.weight']
    diff[f'{attn_prefix}.attention_2.wo.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn2.to_out.0.bias']
    diff[f'{attn_prefix}.layer_norm_3.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm3.weight']
    diff[f'{attn_prefix}.layer_norm_3.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm3.bias']
    diff[f'{attn_prefix}.linear_geglu_1.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.ff.net.0.proj.weight']
    diff[f'{attn_prefix}.linear_geglu_1.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.ff.net.0.proj.bias']
    diff[f'{attn_prefix}.linear_geglu_2.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.ff.net.2.weight']
    diff[f'{attn_prefix}.linear_geglu_2.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.ff.net.2.bias']
    diff[f'{attn_prefix}.conv_out.weight'] = original_model[f'{orig_attn_prefix}.proj_out.weight']
    diff[f'{attn_prefix}.conv_out.bias'] = original_model[f'{orig_attn_prefix}.proj_out.bias']

    # Middle Block - Second Residual
    res_prefix = 'unet.bottom.2'
    orig_res_prefix = f'{o_diff}.middle_block.2'
    diff[f'{res_prefix}.group_norm_first.weight'] = original_model[f'{orig_res_prefix}.in_layers.0.weight']
    diff[f'{res_prefix}.group_norm_first.bias'] = original_model[f'{orig_res_prefix}.in_layers.0.bias']
    diff[f'{res_prefix}.conv_first.weight'] = original_model[f'{orig_res_prefix}.in_layers.2.weight'] # Note: No ".0"
    diff[f'{res_prefix}.conv_first.bias'] = original_model[f'{orig_res_prefix}.in_layers.2.bias']
    diff[f'{res_prefix}.linear_time.weight'] = original_model[f'{orig_res_prefix}.emb_layers.1.weight']
    diff[f'{res_prefix}.linear_time.bias'] = original_model[f'{orig_res_prefix}.emb_layers.1.bias']
    diff[f'{res_prefix}.group_norm_merged.weight'] = original_model[f'{orig_res_prefix}.out_layers.0.weight']
    diff[f'{res_prefix}.group_norm_merged.bias'] = original_model[f'{orig_res_prefix}.out_layers.0.bias']
    diff[f'{res_prefix}.conv_merged.weight'] = original_model[f'{orig_res_prefix}.out_layers.3.weight']
    diff[f'{res_prefix}.conv_merged.bias'] = original_model[f'{orig_res_prefix}.out_layers.3.bias']
    
    #================================================================================#
    #                           Diffusion U-Net Decoder                              #
    #================================================================================#
    decoder_block_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    original_output_blocks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    for i, b in zip(decoder_block_indices, original_output_blocks):
        # --- Map Residual Block (sub-module 0) ---
        res_prefix = f'unet.decoders.{i}.0'
        orig_res_prefix = f'{o_diff}.output_blocks.{b}.0'
        
        diff[f'{res_prefix}.group_norm_first.weight'] = original_model[f'{orig_res_prefix}.in_layers.0.weight']
        diff[f'{res_prefix}.group_norm_first.bias'] = original_model[f'{orig_res_prefix}.in_layers.0.bias']
        diff[f'{res_prefix}.conv_first.weight'] = original_model[f'{orig_res_prefix}.in_layers.2.weight']
        diff[f'{res_prefix}.conv_first.bias'] = original_model[f'{orig_res_prefix}.in_layers.2.bias']
        diff[f'{res_prefix}.linear_time.weight'] = original_model[f'{orig_res_prefix}.emb_layers.1.weight']
        diff[f'{res_prefix}.linear_time.bias'] = original_model[f'{orig_res_prefix}.emb_layers.1.bias']
        diff[f'{res_prefix}.group_norm_merged.weight'] = original_model[f'{orig_res_prefix}.out_layers.0.weight']
        diff[f'{res_prefix}.group_norm_merged.bias'] = original_model[f'{orig_res_prefix}.out_layers.0.bias']
        diff[f'{res_prefix}.conv_merged.weight'] = original_model[f'{orig_res_prefix}.out_layers.3.weight']
        diff[f'{res_prefix}.conv_merged.bias'] = original_model[f'{orig_res_prefix}.out_layers.3.bias']
        if f'{orig_res_prefix}.skip_connection.weight' in original_model:
            diff[f'{res_prefix}.res_layer.weight'] = original_model[f'{orig_res_prefix}.skip_connection.weight']
            diff[f'{res_prefix}.res_layer.bias'] = original_model[f'{orig_res_prefix}.skip_connection.bias']

        # --- Conditionally Map Attention Block (sub-module 1) ---
        orig_attn_prefix = f'{o_diff}.output_blocks.{b}.1'
        if f'{orig_attn_prefix}.norm.weight' in original_model:
            attn_prefix = f'unet.decoders.{i}.1'
            diff[f'{attn_prefix}.group_norm.weight'] = original_model[f'{orig_attn_prefix}.norm.weight']
            diff[f'{attn_prefix}.group_norm.bias'] = original_model[f'{orig_attn_prefix}.norm.bias']
            diff[f'{attn_prefix}.conv_in.weight'] = original_model[f'{orig_attn_prefix}.proj_in.weight']
            diff[f'{attn_prefix}.conv_in.bias'] = original_model[f'{orig_attn_prefix}.proj_in.bias']
            diff[f'{attn_prefix}.layer_norm_1.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm1.weight']
            diff[f'{attn_prefix}.layer_norm_1.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm1.bias']
            q_w, k_w, v_w = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn1.to_q.weight'], original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn1.to_k.weight'], original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn1.to_v.weight']
            diff[f'{attn_prefix}.attention_1.qkv.weight'] = torch.cat([q_w, k_w, v_w])
            diff[f'{attn_prefix}.attention_1.wo.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn1.to_out.0.weight']
            diff[f'{attn_prefix}.attention_1.wo.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn1.to_out.0.bias']
            diff[f'{attn_prefix}.layer_norm_2.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm2.weight']
            diff[f'{attn_prefix}.layer_norm_2.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm2.bias']
            diff[f'{attn_prefix}.attention_2.q.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn2.to_q.weight']
            k_w_c, v_w_c = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn2.to_k.weight'], original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn2.to_v.weight']
            diff[f'{attn_prefix}.attention_2.kv.weight'] = torch.cat([k_w_c, v_w_c])
            diff[f'{attn_prefix}.attention_2.wo.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn2.to_out.0.weight']
            diff[f'{attn_prefix}.attention_2.wo.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.attn2.to_out.0.bias']
            diff[f'{attn_prefix}.layer_norm_3.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm3.weight']
            diff[f'{attn_prefix}.layer_norm_3.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.norm3.bias']
            diff[f'{attn_prefix}.linear_geglu_1.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.ff.net.0.proj.weight']
            diff[f'{attn_prefix}.linear_geglu_1.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.ff.net.0.proj.bias']
            diff[f'{attn_prefix}.linear_geglu_2.weight'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.ff.net.2.weight']
            diff[f'{attn_prefix}.linear_geglu_2.bias'] = original_model[f'{orig_attn_prefix}.transformer_blocks.0.ff.net.2.bias']
            diff[f'{attn_prefix}.conv_out.weight'] = original_model[f'{orig_attn_prefix}.proj_out.weight']
            diff[f'{attn_prefix}.conv_out.bias'] = original_model[f'{orig_attn_prefix}.proj_out.bias']
    
        # --- Conditionally Map Upsample Block (sub-module 2) ---

        upsample_sub_idx = 2 if f'{o_diff}.output_blocks.{b}.1.norm.weight' in original_model else 1
        orig_upsample_prefix = f'{o_diff}.output_blocks.{b}.{upsample_sub_idx}'
        if f'{orig_upsample_prefix}.conv.weight' in original_model:
            upsample_prefix = f'unet.decoders.{i}.{upsample_sub_idx}' # The index in our model matches the original
            diff[f'{upsample_prefix}.weight'] = original_model[f'{orig_upsample_prefix}.conv.weight']
            diff[f'{upsample_prefix}.bias'] = original_model[f'{orig_upsample_prefix}.conv.bias']
    #================================================================================#
    #                           Diffusion U-Net Final Layer                          #
    #================================================================================#
    diff['final.norm.weight'] = original_model[f'{o_diff}.out.0.weight']
    diff['final.norm.bias'] = original_model[f'{o_diff}.out.0.bias']
    diff['final.conv.weight'] = original_model[f'{o_diff}.out.2.weight']
    diff['final.conv.bias'] = original_model[f'{o_diff}.out.2.bias']

    return converted