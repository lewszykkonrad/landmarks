import torchvision.models as models
from PIL import Image, ImageTk
from transformers import AutoFeatureExtractor, ResNetModel, ResNetForImageClassification
import torch
import datasets
from datasets import load_dataset


image = Image.open("../../faces/high_quality_dataset/hot/image_1.jpg")

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-101")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-101")

inputs = feature_extractor(image, return_tensors="pt")

# with torch.no_grad():
#     logits = model(**inputs).logits

# # model predicts one of the 1000 ImageNet classes
# predicted_label = logits.argmax(-1).item()
# print(model.config.id2label[predicted_label])

print(inputs)