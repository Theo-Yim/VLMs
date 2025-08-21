"""
Main training script for Ovis model
Based on original Ovis training structure - clean and simple
"""

import json
import os
import pathlib
import sys
from typing import Dict, Optional

import torch
import transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoModel,
    Trainer,
    HfArgumentParser
)

# Add parent directory to path
sys.path.append(str(pathlib.Path(__file__).parent.parent))

# Import original Ovis components
try:
    from ovis.model.configuration_ovis import OvisConfig
    from ovis.model.modeling_ovis import Ovis, VisualTokenizer
    from ovis.util.constants import BEGIN_LINE, END_LINE
    from ovis.util.utils import smart_unit, rank0_print, rankN_print
except ImportError as e:
    print(f"Warning: Could not import Ovis components: {e}")
    print("Make sure you have installed Ovis package")

# Import custom components
from Ovis.src_theo.arguments import ModelArguments, TrainingArguments
from Ovis.src_theo.dataset_theo import SimpleConversationDataset, DataCollatorForMultimodalDataset


def load_model(model_args: ModelArguments, training_args: TrainingArguments):
    """Load model following original Ovis structure exactly"""
    
    model, loading_info = Ovis.from_pretrained(
        training_args.ovis_pretrained_path,
        output_loading_info=True,
        trust_remote_code=True
    )
    
    rankN_print(BEGIN_LINE)
    rankN_print(f'Loading info of Ovis:\n{loading_info}')
    rankN_print(END_LINE)
    
    model.accepts_loss_kwargs = model_args.accepts_loss_kwargs
    
    if model_args.attn_implementation:
        model.llm.config._attn_implementation = model_args.attn_implementation
        model.visual_tokenizer.vit.config._attn_implementation = model_args.attn_implementation
    
    model.llm.config.use_cache = False
    model.config.use_cache = False
    
    rank0_print(BEGIN_LINE)
    rank0_print(f'model.config:\n{model.config}')
    rank0_print(END_LINE)
    
    return model


def construct_model(model_args: ModelArguments, training_args: TrainingArguments):
    """Construct new model following original Ovis structure exactly"""
    
    # 1. Load pretrained llm and text tokenizer
    attn_kwargs = dict()
    if model_args.attn_implementation:
        attn_kwargs['attn_implementation'] = model_args.attn_implementation
    
    llm = AutoModelForCausalLM.from_pretrained(model_args.llm_name_or_path, **attn_kwargs)
    text_tokenizer = AutoTokenizer.from_pretrained(model_args.llm_name_or_path)
    
    # Set pad token if needed
    if text_tokenizer.pad_token_id is None:
        text_tokenizer.pad_token_id = text_tokenizer.eos_token_id
    
    # 2. Construct visual tokenizer
    if model_args.vit_name_or_path is not None:
        vit = AutoModel.from_pretrained(
            model_args.vit_name_or_path,
            trust_remote_code=True
        )
    else:
        # Use default ViT configuration
        from ovis.model.vit.configuration_siglip2_navit import Siglip2NavitConfig
        vit_config = Siglip2NavitConfig(
            hidden_stride=model_args.vit_hidden_stride,
            window_size=model_args.vit_window_size,
            temporal_patch_size=model_args.vit_temporal_patch_size,
            fullatt_block_indexes=model_args.vit_fullatt_block_indexes,
            preserve_original_pe=model_args.vit_preserve_original_pe,
            use_rope=model_args.vit_use_rope,
        )
        vit = AutoModel.from_config(vit_config)
    
    visual_tokenizer = VisualTokenizer(
        vit=vit,
        visual_vocab_size=model_args.visual_vocab_size,
        image_processor_name_or_path=model_args.vit_name_or_path
    )
    
    # 3. Construct ovis config
    ovis_config = OvisConfig(
        multimodal_max_length=training_args.multimodal_max_length,
        conversation_formatter_class=model_args.conversation_formatter_class,
        hidden_size=llm.config.hidden_size,
        visual_vocab_size=model_args.visual_vocab_size,
    )
    
    # 4. Create Ovis model
    model = Ovis(
        config=ovis_config,
        train_from_scratch=True,
        llm=llm,
        text_tokenizer=text_tokenizer,
        visual_tokenizer=visual_tokenizer
    )
    
    model.accepts_loss_kwargs = model_args.accepts_loss_kwargs
    
    if model_args.attn_implementation:
        model.llm.config._attn_implementation = model_args.attn_implementation
        model.visual_tokenizer.vit.config._attn_implementation = model_args.attn_implementation
    
    model.llm.config.use_cache = False
    model.config.use_cache = False
    
    rank0_print(BEGIN_LINE)
    rank0_print(f'model.config:\n{model.config}')
    rank0_print(END_LINE)
    
    return model


def setup_training_modules(model, training_args: TrainingArguments):
    """Setup training modules following original Ovis structure exactly"""
    
    model.requires_grad_(False)
    parameters = []
    
    for module_name_lr in training_args.train_modules.split('|'):
        module_name_lr = module_name_lr.replace(' ', '').split(':')
        module_lr = training_args.learning_rate
        
        if len(module_name_lr) == 2:
            module_name, module_lr = module_name_lr[0], float(module_name_lr[1])
        elif len(module_name_lr) == 1:
            module_name = module_name_lr[0]
        else:
            raise ValueError(f"Invalid train module format: {module_name_lr}")
        
        match module_name:
            case 'all':
                module = model
            case 'llm':
                module = model.llm
            case 'visual_tokenizer':
                module = model.visual_tokenizer
            case 'visual_tokenizer.head':
                module = model.visual_tokenizer.head
            case 'visual_tokenizer.vit':
                module = model.visual_tokenizer.vit
            case 'visual_tokenizer.vit.last_block':
                module = model.visual_tokenizer._get_last_block()
            case 'vte':
                module = model.vte
            case _:
                raise ValueError(f'Invalid train module name: {module_name}')
        
        module.requires_grad_(True)
        parameters.append({'params': module.parameters(), 'lr': module_lr})
        
        rank0_print(f"Training module: {module_name} with lr: {module_lr}")
    
    # Print parameter statistics
    rank0_print(BEGIN_LINE)
    rank0_print(f'#param of model: {smart_unit(model.num_parameters())}')
    rank0_print(f'#param of llm: {smart_unit(model.llm.num_parameters())}')
    rank0_print(f'#param of vit: {smart_unit(model.visual_tokenizer.vit.num_parameters())}')
    rank0_print(f'#param of vte: {smart_unit(model.vte.weight.numel())}')
    rank0_print(f'#dtype of model: {model.dtype}')
    rank0_print(f'#dtype of llm: {model.llm.dtype}')
    rank0_print(f'#dtype of vit: {model.visual_tokenizer.vit.dtype}')
    rank0_print(f'#dtype of vte: {model.vte.weight.dtype}')
    rank0_print(END_LINE)
    
    return parameters


def load_data(model, training_args: TrainingArguments, data_path: str, image_folder: str):
    """Load data using simple dataset"""
    
    dataset = SimpleConversationDataset(
        data_path=data_path,
        image_folder=image_folder,
        model=model,
        training_args=training_args
    )
    
    data_module = dict(
        train_dataset=dataset,
        data_collator=DataCollatorForMultimodalDataset(model.text_tokenizer)
    )
    
    return data_module


def train():
    """Main training function"""
    
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    
    # Save arguments (same as original)
    with training_args.main_process_first(local=False):
        if training_args.process_index == 0:
            def args2dict(args):
                return {k: str(v) for k, v in args.__dict__.items()}
            
            args_log = json.dumps(dict(
                model_args=args2dict(model_args),
                training_args=args2dict(training_args)
            ), ensure_ascii=False, indent=2)
            
            print(args_log)
            os.makedirs(training_args.output_dir, exist_ok=True)
            
            with open(os.path.join(training_args.output_dir, 'model_training_args.json'), 'w', encoding='utf-8') as f:
                f.write(args_log + '\n')
    
    # Load or construct model
    if training_args.ovis_pretrained_path:
        model = load_model(model_args, training_args)
    else:
        model = construct_model(model_args, training_args)
    
    # Setup training modules
    parameters = setup_training_modules(model, training_args)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(parameters, lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
    
    # Load data
    data_path = training_args.data_path or './data/train_data.json'
    image_folder = training_args.image_folder or './data/images'
    
    data_module = load_data(model, training_args, data_path, image_folder)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        **data_module
    )
    
    # Print dataset info (same as original)
    rankN_print(BEGIN_LINE)
    rankN_print(f'model_accepts_loss_kwargs: {trainer.model_accepts_loss_kwargs}')
    rankN_print(END_LINE)
    
    rankN_print(BEGIN_LINE)
    rankN_print('Dataset sample tensor:')
    rankN_print(data_module['train_dataset'][0])
    rankN_print(END_LINE)
    
    rankN_print(BEGIN_LINE)
    rankN_print('Dataset sample input_ids decoding:')
    sample_input_ids = data_module['train_dataset'][0]['input_ids']
    decoded = model.text_tokenizer.decode([x for x in sample_input_ids if x >= 0])
    rankN_print(decoded)
    rankN_print(END_LINE)
    
    rankN_print(BEGIN_LINE)
    rankN_print('Dataset sample labels decoding:')
    sample_labels = data_module['train_dataset'][0]['labels']
    decoded_labels = model.text_tokenizer.decode([x for x in sample_labels if x >= 0])
    rankN_print(decoded_labels)
    rankN_print(END_LINE)
    
    # Training (same as original)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    trainer.save_state()
    
    # Save model
    model.llm.config.use_cache = True
    model.config.use_cache = True
    trainer.save_model()
    
    rank0_print("Training completed!")


if __name__ == "__main__":
    train()