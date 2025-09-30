def split_ovis25_model(world_size):
    """
    Create device mapping for Ovis2.5 model to efficiently distribute across multiple GPUs.
    Architecture:
    - Visual processing: ViT (27 layers) + visual tokenizer head + VTE
    - LLM: Qwen3-8B (36 layers) + embeddings + norm + lm_head
    Data flow: Images -> ViT -> visual_head -> VTE -> merge with text embeddings -> LLM layers -> output
    """
    device_map = {}
    # world_size = torch.cuda.device_count()
    if world_size < 2:
        # Single GPU - put everything on GPU 0
        return "auto"
    # Load config to get layer counts
    llm_layers = 36  # config.llm_config.num_hidden_layers  # 36 for Qwen3-8B
    device_map["visual_tokenizer.vit"] = 0  # Entire SigLIP ViT (27 layers)
    device_map["visual_tokenizer.head"] = 0  # Visual projection head
    device_map["vte"] = 0  # Visual token embeddings
    device_map["llm.model.embed_tokens"] = 0  # Text embeddings (need to merge with visual)
    # Distribute LLM layers across available GPUs
    if world_size == 2:
        # GPU 0: Vision + embeddings + first few LLM layers
        # GPU 1: Remaining LLM layers + norm + lm_head
        layers_gpu0 = 8  # Keep some LLM layers with embeddings
        for i in range(layers_gpu0):
            device_map[f"llm.model.layers.{i}"] = 1  # 0 # Temporary change to 1
        for i in range(layers_gpu0, llm_layers):
            device_map[f"llm.model.layers.{i}"] = 1
        device_map["llm.model.norm"] = 1
        device_map["llm.lm_head"] = 1
    elif world_size == 3:
        layers_per_gpu = [10, 13, 13]  # Slightly front-loaded due to vision processing
        layer_cnt = 0
        for gpu_id, num_layers in enumerate(layers_per_gpu):
            for i in range(num_layers):
                device_map[f"llm.model.layers.{layer_cnt}"] = gpu_id
                layer_cnt += 1
        device_map["llm.model.norm"] = 2
        device_map["llm.lm_head"] = 2
    else:  # world_size >= 4
        # Reserve some layers for GPU 0 (vision processing) and last GPU (output)
        layers_gpu0 = max(4, llm_layers // (world_size + 1))  # Front-load slightly
        layers_last_gpu = max(4, llm_layers // (world_size + 1))
        remaining_layers = llm_layers - layers_gpu0 - layers_last_gpu
        middle_gpus = world_size - 2
        if middle_gpus > 0:
            layers_per_middle_gpu = remaining_layers // middle_gpus
            extra_layers = remaining_layers % middle_gpus
        else:
            layers_per_middle_gpu = 0
            extra_layers = 0
            layers_last_gpu += remaining_layers
        # GPU 0: First layers
        for i in range(layers_gpu0):
            device_map[f"llm.model.layers.{i}"] = 0
        # Middle GPUs: Distribute remaining layers
        layer_cnt = layers_gpu0
        for gpu_id in range(1, world_size - 1):
            num_layers = layers_per_middle_gpu + (1 if gpu_id - 1 < extra_layers else 0)
            for i in range(num_layers):
                device_map[f"llm.model.layers.{layer_cnt}"] = gpu_id
                layer_cnt += 1
        # Last GPU: Final layers + norm + lm_head
        for i in range(layer_cnt, llm_layers):
            device_map[f"llm.model.layers.{i}"] = world_size - 1
        device_map["llm.model.norm"] = world_size - 1
        device_map["llm.lm_head"] = world_size - 1
    return device_map
