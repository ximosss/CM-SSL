# utils function referenced from github repository of LaBram
# https://github.com/935963004/LaBraM/blob/main/utils.py


# JTwBio_Tokenzier:
#   - te (Temporal_Encoder)
#   - pe (Positioning_Encoding)
#   - je (Joint_Encoder) - include blocks (TransformerEncoderLayer)
#   - quantizer (NormEMAVectorQuantizer)
#   - de (Decoder) - included blocks (TransformerEncoderLayer)

def get_num_layer_for_jtwbio(var_name, hparams):

    num_encoder_layers = hparams.num_transformer_encoder_layers
    num_decoder_layers = hparams.num_transformer_decoder_layers 

    if var_name.startswith("te.") or var_name.startswith("pe."):
        return 0
    elif var_name.startswith("je.encoder.layers"):
        layer_idx = int(var_name.split('.')[3]) 
        return 1 + layer_idx
    elif var_name.startswith("de.decoder.layers"):
        layer_idx = int(var_name.split('.')[3]) 
        return (1 + num_encoder_layers) + layer_idx
    elif var_name.startswith("quantizer.") or var_name.startswith("de.mlp"):
        return 1 + num_encoder_layers + num_decoder_layers
    else:
        return 1 + num_encoder_layers + num_decoder_layers

class LayerDecayValueAssigner(object):
    def __init__(self, values, hparams):

        self.values = values
        self.hparams = hparams 

    def get_scale(self, layer_id):
        if layer_id < 0 or layer_id >= len(self.values):
            print(f"Warning: layer_id {layer_id} out of bounds for values of length {len(self.values)}. Returning 1.0.")
            return 1.0
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_jtwbio(var_name, self.hparams)

def get_parameter_groups_custom(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # no need to decay
        if param.ndim <= 1 or name.endswith(".bias") or name in skip_list:
            group_wd_type = "no_decay"
            this_weight_decay = 0.
        else:
            group_wd_type = "decay"
            this_weight_decay = weight_decay

        # determine layer id and lr scale
        layer_id = None
        lr_scale = 1.0
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            if get_layer_scale is not None and layer_id is not None:
                lr_scale = get_layer_scale(layer_id)

        group_name = f"layer_{layer_id}_{group_wd_type}" if layer_id is not None else group_wd_type
        
        if group_name not in parameter_group_names:
            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": lr_scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": lr_scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    
    for group_name, group_info in parameter_group_names.items():
        print(f" Group: {group_name}, WD: {group_info['weight_decay']:.1e}, \
              LR_Scale: {group_info['lr_scale']:.3f}, Params: {len(group_info['params'])} tensors")
    print("------------------------------------------")
    
    return list(parameter_group_vars.values())
