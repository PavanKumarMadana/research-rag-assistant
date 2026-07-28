"""
Classification Service Module.

Provides TensorFlow-based document classification and ML pipeline management.
"""

import os
import pickle
import time
import json
from typing import Optional
import numpy as np

from loguru import logger

from backend.app.core.config import settings


# Define categories
DOCUMENT_CATEGORIES = [
    "Artificial Intelligence",
    "Machine Learning",
    "Computer Vision",
    "Natural Language Processing",
    "Robotics",
    "Cyber Security",
    "Cloud Computing",
    "Uncategorized",
]

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Artificial Intelligence": [
        "artificial intelligence", "AI", "intelligent agent", "reasoning", "knowledge representation",
        "expert system", "neural network", "deep learning", "cognitive", "intelligent system",
    ],
    "Machine Learning": [
        "machine learning", "supervised learning", "unsupervised learning", "reinforcement learning",
        "classification", "regression", "clustering", "training data", "model training",
        "feature extraction", "gradient descent", "overfitting", "cross-validation",
    ],
    "Computer Vision": [
        "computer vision", "image recognition", "object detection", "image segmentation",
        "convolutional", "visual", "image processing", "face recognition", "scene understanding",
        "optical flow", "image classification", "video analysis",
    ],
    "Natural Language Processing": [
        "natural language", "NLP", "text classification", "sentiment analysis", "named entity recognition",
        "machine translation", "text generation", "language model", "transformer", "BERT",
        "GPT", "text mining", "information extraction", "question answering",
    ],
    "Robotics": [
        "robotics", "robot", "autonomous", "manipulator", "sensor fusion", "SLAM",
        "motion planning", "robot control", "actuator", "kinematics", "humanoid",
    ],
    "Cyber Security": [
        "cyber security", "cybersecurity", "security", "encryption", "malware", "intrusion detection",
        "firewall", "authentication", "vulnerability", "threat detection", "network security",
        "cryptography", "cyber attack", "ransomware",
    ],
    "Cloud Computing": [
        "cloud computing", "cloud", "distributed system", "virtualization", "container",
        "microservices", "serverless", "scalability", "load balancing", "cloud service",
        "AWS", "Azure", "GCP", "infrastructure as code",
    ],
}


class ClassificationService:
    """Service for document classification using keyword matching and ML."""

    def __init__(self) -> None:
        """Initialize classification service."""
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.categories = DOCUMENT_CATEGORIES
        self._load_or_init_model()

    def _load_or_init_model(self) -> None:
        """Load existing model or initialize fallback keyword classifier."""
        model_path = settings.MODEL_PATH
        model_dir = os.path.dirname(model_path)

        # Try to load TensorFlow model
        tf_model_path = os.path.join(model_dir, "document_classifier.keras")
        vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
        encoder_path = os.path.join(model_dir, "label_encoder.pkl")

        if os.path.exists(tf_model_path) and os.path.exists(vectorizer_path):
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(tf_model_path)
                with open(vectorizer_path, "rb") as f:
                    self.vectorizer = pickle.load(f)
                with open(encoder_path, "rb") as f:
                    self.label_encoder = pickle.load(f)
                logger.info("Loaded trained TensorFlow classification model")
                return
            except Exception as e:
                logger.warning(f"Could not load TF model: {e}")

        logger.info("Using keyword-based classification (no trained model found)")

    def classify_text(self, text: str) -> dict:
        """Classify document text into a category.

        Args:
            text: Document text to classify.

        Returns:
            dict: Classification result with category and confidence.
        """
        if not text or not text.strip():
            return {
                "predicted_category": "Uncategorized",
                "confidence": 0.0,
                "probabilities": {cat: 0.0 for cat in self.categories},
            }

        # If TF model is available, use it
        if self.model is not None and self.vectorizer is not None:
            return self._classify_with_tf(text)

        # Otherwise use keyword-based classification
        return self._classify_with_keywords(text)

    def _classify_with_tf(self, text: str) -> dict:
        """Classify using TensorFlow model.

        Args:
            text: Text to classify.

        Returns:
            dict: Classification result.
        """
        try:
            import tensorflow as tf

            # Vectorize the text
            text_vector = self.vectorizer.transform([text])
            # Convert to dense for TF (handle sparse)
            text_dense = text_vector.toarray()

            # Predict
            predictions = self.model.predict(text_dense, verbose=0)[0]
            predicted_idx = int(np.argmax(predictions))
            confidence = float(predictions[predicted_idx])

            # Get probabilities for all categories
            probabilities = {}
            for i, cat in enumerate(self.categories):
                if i < len(predictions):
                    probabilities[cat] = float(predictions[i])

            predicted_category = self.categories[predicted_idx] if predicted_idx < len(self.categories) else "Uncategorized"

            return {
                "predicted_category": predicted_category,
                "confidence": confidence,
                "probabilities": probabilities,
            }

        except Exception as e:
            logger.error(f"TF classification failed: {e}")
            return self._classify_with_keywords(text)

    def _classify_with_keywords(self, text: str) -> dict:
        """Classify using keyword matching.

        Args:
            text: Text to classify.

        Returns:
            dict: Classification result.
        """
        text_lower = text.lower()
        scores = {}

        for category, keywords in CATEGORY_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                count = text_lower.count(keyword)
                score += count

            # Normalize by text length
            normalized_score = score / max(len(text.split()), 1) * 1000
            scores[category] = normalized_score
        scores["Uncategorized"] = 0.0

        # Get the best category
        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]

        # If no keywords matched, classify as Uncategorized
        if best_score == 0:
            best_category = "Uncategorized"
            scores["Uncategorized"] = 1.0

        # Calculate probabilities (softmax-like normalization)
        total_score = sum(scores.values()) or 1
        probabilities = {
            cat: (score / total_score) for cat, score in scores.items()
        }

        confidence = probabilities[best_category]

        return {
            "predicted_category": best_category,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    def get_model_info(self) -> dict:
        """Get information about the classification model.

        Returns:
            dict: Model information.
        """
        return {
            "model_type": "tensorflow" if self.model else "keyword_fallback",
            "categories": self.categories,
            "num_categories": len(self.categories),
            "is_trained": self.model is not None,
        }


class MLPipeline:
    """Machine Learning pipeline for training the document classifier."""

    def __init__(self) -> None:
        """Initialize ML pipeline."""
        self.categories = DOCUMENT_CATEGORIES

    def prepare_training_data(self) -> tuple:
        """Prepare training data from keyword definitions.

        Returns:
            tuple: (texts, labels) for training.
        """
        texts = []
        labels = []

        # Generate synthetic training data from keywords
        for category, keywords in CATEGORY_KEYWORDS.items():
            # Create sample sentences for each keyword
            for keyword in keywords:
                # Multiple variations
                texts.extend([
                    f"This document discusses {keyword} in detail.",
                    f"The application of {keyword} is explored in this paper.",
                    f"A comprehensive study of {keyword} and its implications.",
                    f"This research focuses on {keyword} techniques and methods.",
                    f"Recent advances in {keyword} are presented in this work.",
                    f"An overview of {keyword} approaches and algorithms.",
                    f"The role of {keyword} in modern systems is analyzed.",
                    f"Experimental results for {keyword} are evaluated.",
                ])
                labels.extend([category] * 8)

        uncategorized_samples = [
            "This document contains general technical background information.",
            "The paper discusses broad system observations without a specific field.",
            "This report includes mixed topics and introductory concepts.",
            "The work provides general research context and future opportunities.",
        ]
        texts.extend(uncategorized_samples)
        labels.extend(["Uncategorized"] * len(uncategorized_samples))

        logger.info(f"Prepared {len(texts)} training samples across {len(self.categories)} categories")
        return texts, labels

    def train_model(self) -> bool:
        """Train a TensorFlow text classification model.

        Returns:
            bool: True if training succeeded.
        """
        try:
            import tensorflow as tf
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import LabelEncoder
            import pickle

            logger.info("Starting TensorFlow model training...")

            # Prepare data
            texts, labels = self.prepare_training_data()

            # Vectorize text
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            X = vectorizer.fit_transform(texts)

            # Encode labels
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(labels)

            # Convert to categorical
            y_categorical = tf.keras.utils.to_categorical(y, num_classes=len(self.categories))

            # Build model
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(X.shape[1],)),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(len(self.categories), activation="softmax"),
            ])

            model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

            # Train
            history = model.fit(
                X.toarray(),
                y_categorical,
                epochs=50,
                batch_size=16,
                validation_split=0.2,
                verbose=1,
            )

            # Save model
            model_dir = os.path.dirname(settings.MODEL_PATH)
            tf_model_path = os.path.join(model_dir, "document_classifier.keras")
            model.save(tf_model_path)

            # Save vectorizer and encoder
            with open(os.path.join(model_dir, "vectorizer.pkl"), "wb") as f:
                pickle.dump(vectorizer, f)
            with open(os.path.join(model_dir, "label_encoder.pkl"), "wb") as f:
                pickle.dump(label_encoder, f)

            logger.info(f"Model trained and saved to {tf_model_path}")

            # Save training history
            history_path = os.path.join(model_dir, "training_history.json")
            with open(history_path, "w") as f:
                json.dump({
                    "accuracy": [float(x) for x in history.history["accuracy"]],
                    "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
                    "loss": [float(x) for x in history.history["loss"]],
                    "val_loss": [float(x) for x in history.history["val_loss"]],
                    "final_accuracy": float(history.history["accuracy"][-1]),
                    "final_val_accuracy": float(history.history["val_accuracy"][-1]),
                }, f, indent=2)

            return True

        except ImportError as e:
            logger.error(f"TensorFlow not installed: {e}")
            return False
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False

    def evaluate_model(self) -> dict:
        """Evaluate the trained model performance.

        Returns:
            dict: Evaluation metrics.
        """
        try:
            import tensorflow as tf
            from sklearn.metrics import classification_report, confusion_matrix
            import numpy as np

            model_dir = os.path.dirname(settings.MODEL_PATH)
            tf_model_path = os.path.join(model_dir, "document_classifier.keras")
            vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
            encoder_path = os.path.join(model_dir, "label_encoder.pkl")

            if not all(os.path.exists(p) for p in [tf_model_path, vectorizer_path, encoder_path]):
                return {"error": "Trained model not found"}

            model = tf.keras.models.load_model(tf_model_path)
            with open(vectorizer_path, "rb") as f:
                vectorizer = pickle.load(f)
            with open(encoder_path, "rb") as f:
                label_encoder = pickle.load(f)

            # Generate test data
            texts, labels = self.prepare_training_data()
            X_test = vectorizer.transform(texts).toarray()
            y_test = label_encoder.transform(labels)

            # Predict
            predictions = model.predict(X_test, verbose=0)
            y_pred = np.argmax(predictions, axis=1)

            # Metrics
            report = classification_report(
                y_test, y_pred,
                target_names=self.categories,
                output_dict=True,
                zero_division=0,
            )

            accuracy = np.mean(y_pred == y_test)

            return {
                "accuracy": float(accuracy),
                "classification_report": report,
                "num_classes": len(self.categories),
            }

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {"error": str(e)}
