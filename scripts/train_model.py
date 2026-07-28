"""
TensorFlow Model Training Script.

Trains the document classification model using keyword-based training data.
Run this script to train and save the model before using classification features.

Usage:
    python scripts/train_model.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.ml.classification_service import MLPipeline
from backend.app.core.logging import setup_logging
from loguru import logger


def main():
    """Main training function."""
    setup_logging()
    logger.info("=" * 60)
    logger.info("Document Classification Model Training")
    logger.info("=" * 60)

    pipeline = MLPipeline()

    # Prepare data
    logger.info("Step 1: Preparing training data...")
    texts, labels = pipeline.prepare_training_data()
    logger.info(f"Prepared {len(texts)} training samples")

    # Train model
    logger.info("Step 2: Training TensorFlow model...")
    success = pipeline.train_model()

    if success:
        logger.info("Step 3: Model trained and saved successfully!")
        logger.info(f"Model saved to: models/document_classifier/tf_model")

        # Evaluate
        logger.info("Step 4: Evaluating model...")
        metrics = pipeline.evaluate_model()
        if "accuracy" in metrics:
            logger.info(f"Model accuracy: {metrics['accuracy']:.2%}")
        logger.info("Training complete!")
    else:
        logger.error("Model training failed!")
        logger.error("Ensure TensorFlow is installed: pip install tensorflow")
        sys.exit(1)


if __name__ == "__main__":
    main()