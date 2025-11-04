# Hat Detection Project

A computer vision project that uses a pre-trained YOLOS model to detect if a person is wearing a hat in images. The project highlights detected hats and provides a textual indication of presence.

## Features
### What it does
- Detects hats in images
- Draws bounding boxes around hats
- Returns a message if a hat is detected

## Technologies Used
- Python
- PyTorch
- Transformers (Hugging Face)
- PIL (Python Imaging Library)

## Installation
```bash
pip install torch transformers pillow
```

## Usage

### 1. Import Libraries
```python
from PIL import Image, ImageDraw
from transformers import YolosImageProcessor, YolosForObjectDetection
import torch
```

### 2. Load Model and Processor
```python
model_name = 'valentinafeve/yolos-fashionpedia'
processor = YolosImageProcessor.from_pretrained(model_name)
model = YolosForObjectDetection.from_pretrained(model_name)
```

### 3. Load and Process Image
```python
image = Image.open('mulher.webp').convert('RGB')
inputs = processor(images=image, return_tensors='pt')
outputs = model(**inputs)
```

### 4. Post-process and Detect Hats
```python
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

draw = ImageDraw.Draw(image)
hat_detected = False

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    if score > 0.7:
        label_name = model.config.id2label[label.item()]
        if label_name.lower() == "hat":
            hat_detected = True
            box = [round(i, 2) for i in box.tolist()]
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], box[1]), f"{label_name} ({score:.2f})", fill="red")

if hat_detected:
    print("A person is wearing a hat.")
else:
    print("A person is NOT wearing a hat.")

image.show()
image.save("result_hat.png")
```

## Example
- Input: `mulher.webp`
- Output: `result_hat.png` with detected hats highlighted
