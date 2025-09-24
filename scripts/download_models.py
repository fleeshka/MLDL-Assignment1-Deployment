import os
import warnings
import logging
from transformers import pipeline

# Disable symlink warnings for Windows compatibility
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Reduce logging verbosity
logging.getLogger("transformers").setLevel(logging.ERROR)


def save_models():
    """
    Save pre-trained transformer models to local directory for offline deployment.
    Saves both model weights and tokenizers for classification and text generation tasks.
    """

    # Create directories for model storage
    os.makedirs("../code/models/classifier", exist_ok=True)
    os.makedirs("../code/models/generator", exist_ok=True)

    # Save sentiment analysis classifier
    print("Saving classification model...")
    try:
        # Load pre-trained sentiment analysis model
        classifier = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english"  # SST-2 fine-tuned model
        )

        # Save model and tokenizer to local directory
        classifier.model.save_pretrained("code/models/classifier/")
        classifier.tokenizer.save_pretrained("code/models/classifier/")
        print("Classifier saved to code/models/classifier/")

        # Verify model functionality
        test_result = classifier("I love this product!")
        print(f"Model test output: {test_result}")

    except Exception as e:
        print(f"Classifier error: {e}")
        return

    # Save text generation model
    print("Saving generation model...")
    try:
        # Load pre-trained text-to-text generation model
        generator = pipeline("text2text-generation", model="t5-small")

        # Save model and tokenizer
        generator.model.save_pretrained("code/models/generator/")
        generator.tokenizer.save_pretrained("code/models/generator/")
        print("Generator saved to code/models/generator/")

        # Verify model functionality with translation task
        test_result = generator("translate English to French: Hello world")
        print(f"Model test output: {test_result[0]['generated_text']}")

    except Exception as e:
        print(f"Generator error: {e}")
        return

    print("All models saved successfully")
    print("Model locations:")
    print(" - code/models/classifier/")
    print(" - code/models/generator/")


if __name__ == "__main__":
    save_models()