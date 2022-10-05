from PIL import Image, ImageTk
from transformers import AutoFeatureExtractor, ResNetForImageClassification
import torch
from datasets import load_dataset

image = Image.open("../../faces/high_quality_dataset/hot/image_1.jpg") 

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

inputs = feature_extractor(image, return_tensors="pt")

embedder = model.base_model

with torch.no_grad():
    output = embedder(**inputs)
    
image_embedding = output.pooler_output.flatten()
print(type(image_embedding[0]))

