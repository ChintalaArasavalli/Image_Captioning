# Image_Captioning

# 🖼️ Image Captioning using ViT-GPT2

## 👩‍🎓 Student Details
- **Name**: Chintala Arasavalli  
- **College**: St. Francis College for Women  
- **Department**: B.Com  
- **Email**: arasavallichintala@gmail.com

---

## 📌 Overview
This project implements an image captioning system using a pre-trained Vision Transformer (ViT) and GPT-2 model. The system generates human-like descriptive captions for any input image. It uses Hugging Face's powerful `nlpconnect/vit-gpt2-image-captioning` model and runs seamlessly in Google Colab.

---

## 🧠 Problem Statement
Understanding visual content and converting it into meaningful text is a core challenge in AI. This project addresses that challenge by generating accurate, real-time captions for images using a pre-trained deep learning model, which is useful in accessibility, content generation, and surveillance.

---

## 🚀 Tools & Technologies Used
- **Model**: Vision Transformer (ViT) + GPT-2 (via Hugging Face)
- **Libraries**: `transformers`, `torch`, `Pillow`
- **Platform**: Google Colab (GPU/CPU)
- **Language**: Python

---

## 📥 Installation
Install required packages using:
```bash
pip install transformers torch Pillow
```

---

## 🧪 How It Works

```python
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch

# Load the model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Caption prediction function
def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values.to(device)
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [pred.strip() for pred in preds]

# Example usage
predict_step(["/content/sample_image.png"])
```

---

## ✅ Sample Output

> **Input**: Image of a cat lying on a bed  
> **Generated Caption**: `"a cat sitting on a bed with a blanket"`

---

## 🔮 Future Scope
- Fine-tune the model with a custom image dataset  
- Deploy it as a web app using Streamlit or Flask  
- Extend to video captioning or multi-language generation

---

## 📚 References
- Hugging Face ViT-GPT2: https://huggingface.co/nlpconnect/vit-gpt2-image-captioning  
- Transformers Docs: https://huggingface.co/docs/transformers  
- GitHub: https://github.com/Chintalaharika5

---

## 🧑‍💻 Developed By
**Chintala Arasavalli**  
Email: arasavallichintala@gmail.com

