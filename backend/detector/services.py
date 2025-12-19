from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

class DeepfakeModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
            cls._instance = super(DeepfakeModel, cls).__new__(cls)
            cls.processor = ViTImageProcessor.from_pretrained(model_name)
            cls.model = ViTForImageClassification.from_pretrained(model_name)
        return cls._instance

    def predict(self, image_file):
        image = Image.open(image_file).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze()
            
        # Map class indices to labels
        labels = self.model.config.id2label
        prediction_idx = torch.argmax(probs).item()
        label_name = labels[prediction_idx].lower()
        results = {labels[i]: float(probs[i]) for i in range(len(probs))}
        
        # Determine the top result
        top_label = max(results, key=results.get)
        return {
            "prediction": label_name,
            "confidence": round(float(probs[prediction_idx]) * 100, 2),
            "raw_scores": results
        }