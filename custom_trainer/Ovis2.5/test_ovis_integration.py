#!/usr/bin/env python3
"""
Comprehensive test script for Ovis2.5-9B integration
Based on official guide: https://huggingface.co/AIDC-AI/Ovis2.5-9B
"""

import json
import logging
import os
import sys
import tempfile

import torch
from config import SFTTrainingConfig
from data_utils import GroundingParser, OvisDataCollator, OvisDataset
from inference import InferenceConfig, OvisInference
from PIL import Image
from transformers import AutoModelForCausalLM

sys.path.append("..")
from crop_tool import CropTool, parse_and_replace_tool_calls

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model_loading():
    """Test basic model loading with official guide parameters"""
    logger.info("Testing Ovis2.5-9B model loading...")

    try:
        # Load model with official guide parameters
        model = AutoModelForCausalLM.from_pretrained(
            "AIDC-AI/Ovis2.5-9B",
            torch_dtype=torch.bfloat16,  # As per official guide
            trust_remote_code=True,
            device_map="cpu",  # Keep on CPU for testing
            low_cpu_mem_usage=True,
        )

        logger.info(f"✅ Model loaded: {type(model).__name__}")

        # Test text tokenizer (key correction from official guide)
        if hasattr(model, "text_tokenizer"):
            logger.info("✅ Model has text_tokenizer attribute")
            vocab_size = len(model.text_tokenizer.get_vocab())
            logger.info(f"   Text tokenizer vocab size: {vocab_size}")
        else:
            logger.warning("❌ Model missing text_tokenizer attribute")

        # Test preprocessing method
        if hasattr(model, "preprocess_inputs"):
            logger.info("✅ Model has preprocess_inputs method")
        else:
            logger.warning("❌ Model missing preprocess_inputs method")

        return model, True

    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return None, False


def test_official_guide_example():
    """Test the exact example from the official guide"""
    logger.info("Testing official guide example...")

    try:
        # Load model as in official guide
        model = AutoModelForCausalLM.from_pretrained(
            "AIDC-AI/Ovis2.5-9B",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cpu",  # Use CPU for testing
        )

        # Prepare messages exactly as in official guide
        # Use a simple test image instead of downloading
        test_image = Image.new("RGB", (448, 448), color="red")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": test_image},
                    {"type": "text", "text": "What color is this image?"},
                ],
            }
        ]

        # Test preprocessing with official parameters
        input_ids, pixel_values, grid_thws = model.preprocess_inputs(
            messages=messages, add_generation_prompt=True, enable_thinking=True
        )

        logger.info("✅ Preprocessing successful!")
        logger.info(f"   Input IDs shape: {input_ids.shape}")
        logger.info(
            f"   Pixel values shape: {pixel_values.shape if pixel_values is not None else None}"
        )
        logger.info(f"   Grid thws shape: {grid_thws.shape if grid_thws is not None else None}")

        # Test generation with official parameters (short for testing)
        with torch.no_grad():
            outputs = model.generate(
                inputs=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                enable_thinking=True,
                enable_thinking_budget=True,
                max_new_tokens=100,  # Shorter for testing
                thinking_budget=50,  # Shorter for testing
            )

        # Test decoding with model.text_tokenizer (key correction)
        response = model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"✅ Generation successful!")
        logger.info(f"   Response length: {len(response)} characters")
        logger.info(f"   Response preview: {response[:100]}...")

        return True

    except Exception as e:
        logger.error(f"❌ Official guide example failed: {e}")
        return False


def test_crop_tool():
    """Test the crop tool functionality (using existing QwenVL2.5 CropTool)"""
    logger.info("Testing crop tool...")

    try:
        crop_tool = CropTool()

        # Test parsing original format
        original_text = """<think>
        I need to analyze the person on the left.
        {Crop person on left [0.1, 0.2, 0.5, 0.8]}
        Looking at this cropped region, I can see the person clearly.
        </think>
        
        <answer>The person is wearing a red jacket.</answer>"""

        # Use existing parse_and_replace_tool_calls function
        processed_text = parse_and_replace_tool_calls(original_text)

        logger.info("✅ Original format parsing successful!")
        logger.info(f"   Processed text contains tool calls: {'<tool_call>' in processed_text}")

        # Test tool call extraction
        tool_calls = crop_tool.extract_tool_calls(processed_text)
        logger.info(f"   Tool calls extracted: {len(tool_calls)}")

        # Test image cropping
        test_image = Image.new("RGB", (100, 100), color="blue")
        if tool_calls:
            cropped = crop_tool.crop_image(test_image, tool_calls[0]["coordinates"])
            logger.info(f"   Cropped image size: {cropped.size}")
            logger.info("✅ Image cropping successful!")

        # Test format for training (using existing method)
        content_items = crop_tool.format_for_training(processed_text, test_image)
        logger.info(f"   Training format items: {len(content_items)}")
        logger.info(f"   Content types: {[item['type'] for item in content_items]}")

        logger.info("✅ Using existing QwenVL2.5 CropTool successfully!")

        return True

    except Exception as e:
        logger.error(f"❌ Crop tool test failed: {e}")
        return False


def test_grounding_parser():
    """Test the grounding parser for Ovis format"""
    logger.info("Testing grounding parser...")

    try:
        parser = GroundingParser()

        # Test text with grounding elements
        test_text = """
        The image shows <ref>a red car</ref><box>(0.1,0.2),(0.5,0.6)</box> parked next to 
        <ref>a blue building</ref><point>(0.7,0.3)</point>. There are also 
        <ref>two people</ref><box>(0.8,0.1),(0.95,0.4)</box> walking nearby.
        """

        grounding = parser.parse_grounding(test_text)

        logger.info("✅ Grounding parsing successful!")
        logger.info(f"   References: {grounding['refs']}")
        logger.info(f"   Boxes: {grounding['boxes']}")
        logger.info(f"   Points: {grounding['points']}")

        # Verify parsing results
        assert len(grounding["refs"]) == 3
        assert len(grounding["boxes"]) == 2
        assert len(grounding["points"]) == 1

        return True

    except Exception as e:
        logger.error(f"❌ Grounding parser test failed: {e}")
        return False


def test_data_pipeline():
    """Test the data pipeline with dummy data"""
    logger.info("Testing Ovis data pipeline...")

    try:
        # Create dummy data file
        dummy_data = {
            "image_path": "test.jpg",
            "image_id": "test_001",
            "QnA": [
                {
                    "Q": "What is in this image?",
                    "A3": "<think>\nI need to analyze this image and find the red object.\n{Crop red object [0.1, 0.1, 0.5, 0.5]}\nI can see it's a red square.\n</think>\n\n<answer>This image contains a red square.</answer>",
                }
            ],
        }

        # Create temporary files
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(dummy_data) + "\n")
            data_path = f.name

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy image
            test_image = Image.new("RGB", (224, 224), color="green")
            image_path = os.path.join(temp_dir, "test.jpg")
            test_image.save(image_path)

            # Load model for dataset
            model, success = test_model_loading()
            if not success:
                return False

            # Test dataset
            logger.info("Testing OvisDataset...")
            dataset = OvisDataset(
                data_path=data_path,
                image_base_path=temp_dir,
                model=model,
                max_length=512,
                stage="sft",
            )

            logger.info(f"Dataset size: {len(dataset)}")

            # Test getting an item
            if len(dataset) > 0:
                sample = dataset[0]
                logger.info(f"✅ Sample processed successfully!")
                logger.info(f"   Keys: {list(sample.keys())}")
                for key, value in sample.items():
                    if torch.is_tensor(value):
                        logger.info(f"   {key}: {value.shape}")
                    else:
                        logger.info(f"   {key}: {type(value)}")

            # Test data collator
            logger.info("Testing OvisDataCollator...")
            collator = OvisDataCollator(model=model)
            batch = collator([sample])
            logger.info(f"✅ Batch created successfully!")
            logger.info(f"   Batch keys: {list(batch.keys())}")

        # Clean up
        os.unlink(data_path)
        return True

    except Exception as e:
        logger.error(f"❌ Data pipeline test failed: {e}")
        return False


def test_inference_class():
    """Test the Ovis inference class"""
    logger.info("Testing OvisInference class...")

    try:
        # Initialize inference config
        config = InferenceConfig(
            model_path="AIDC-AI/Ovis2.5-9B",  # Use base model
            max_new_tokens=100,  # Short for testing
            thinking_budget=50,
            device="cpu",  # Use CPU for testing
        )

        # This will load the model
        inference = OvisInference(config)
        logger.info("✅ OvisInference initialized successfully!")

        # Test with simple image
        test_image = Image.new("RGB", (224, 224), color="blue")

        result = inference.generate(image=test_image, question="What color is this image?")

        logger.info("✅ Inference generation successful!")
        logger.info(f"   Response length: {len(result['response'])}")
        logger.info(f"   Has thinking content: {bool(result['parsed'].get('think_content'))}")
        logger.info(f"   Has grounding: {result['parsed'].get('has_grounding', False)}")

        return True

    except Exception as e:
        logger.error(f"❌ Inference test failed: {e}")
        return False


def test_lora_compatibility():
    """Test LoRA configuration compatibility with Ovis"""
    logger.info("Testing LoRA compatibility...")

    try:
        from peft import LoraConfig, TaskType, get_peft_model

        # Load model
        model, success = test_model_loading()
        if not success:
            return False

        # Get Ovis LoRA config
        config = SFTTrainingConfig()

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
        )

        logger.info("Applying LoRA to Ovis model...")
        peft_model = get_peft_model(model, lora_config)

        logger.info("✅ LoRA applied successfully!")
        peft_model.print_trainable_parameters()

        return True

    except Exception as e:
        logger.error(f"❌ LoRA test failed: {e}")
        return False


def run_all_tests():
    """Run all Ovis2.5-9B integration tests"""
    logger.info("=" * 60)
    logger.info("RUNNING OVIS2.5-9B INTEGRATION TESTS")
    logger.info("Based on official guide: https://huggingface.co/AIDC-AI/Ovis2.5-9B")
    logger.info("=" * 60)

    tests = [
        ("Model Loading", test_model_loading),
        ("Official Guide Example", test_official_guide_example),
        ("Crop Tool", test_crop_tool),
        ("Grounding Parser", test_grounding_parser),
        ("Data Pipeline", test_data_pipeline),
        ("Inference Class", test_inference_class),
        ("LoRA Compatibility", test_lora_compatibility),
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info(f"\n{'=' * 40}")
        logger.info(f"TEST: {test_name}")
        logger.info(f"{'=' * 40}")

        try:
            if test_name == "Model Loading":
                model, success = test_func()
                results[test_name] = success
            else:
                results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = False

    # Print summary
    logger.info(f"\n{'=' * 60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'=' * 60}")

    passed = 0
    failed = 0

    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{test_name:<25}: {status}")
        if success:
            passed += 1
        else:
            failed += 1

    logger.info(f"\nTotal: {passed + failed}, Passed: {passed}, Failed: {failed}")

    if failed == 0:
        logger.info("🎉 All tests passed! Ovis2.5-9B integration ready!")
        logger.info("\nNext steps:")
        logger.info("1. Install dependencies: pip install -r requirements.txt")
        logger.info("2. Prepare your data in JSONL format")
        logger.info("3. Run training: python train_ovis25.py --stage sft")
    else:
        logger.warning(f"⚠️  {failed} test(s) failed. Check the logs above.")
        logger.info("\nTroubleshooting:")
        logger.info("1. Ensure all dependencies are installed")
        logger.info("2. Check CUDA availability if using GPU")
        logger.info("3. Verify model download completed successfully")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
