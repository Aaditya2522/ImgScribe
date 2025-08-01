import os
import torch
from torch.utils.data import Dataset, Subset
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

# 🧠 Set memory fragmentation handling
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ✅ Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🖥️ Using device:", device)


# 🧩 Custom Dataset
class CaptionDataset(Dataset):
    def __init__(self, image_folder, caption_file, processor):
        self.image_folder = image_folder
        self.processor = processor
        self.images = []
        self.captions = []

        with open(caption_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|", 1)
                if len(parts) == 2:
                    filename, caption = parts
                    image_path = os.path.join(image_folder, filename)
                    if os.path.isfile(image_path):
                        self.images.append(image_path)
                        self.captions.append(caption)

        if len(self.images) == 0:
            raise ValueError("❌ No valid image-caption pairs found.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        caption = self.captions[idx]
        encoding = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32,  # 💡 Lower max_length = less memory
        )
        encoding = {k: v.squeeze(0).to(device) for k, v in encoding.items()}
        encoding["labels"] = encoding["input_ids"].clone()
        return encoding


# 🚀 Train function
def train_model():
    image_dir = "Flickr8k_Dataset/Images"
    caption_file = "Flickr8k_Dataset/captions.txt"

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base", use_fast=False
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    dataset = CaptionDataset(image_dir, caption_file, processor)

    # 🔍 Limit samples for testing (increase later)
    dataset = Subset(dataset, range(500))
    print(f"✅ Loaded {len(dataset)} samples.")

    # 🛠 Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        logging_dir="./logs",
        save_strategy="epoch",
        remove_unused_columns=False,
        fp16=True if torch.cuda.is_available() else False,  # mixed precision
        report_to="none",
    )

    data_collator = DataCollatorWithPadding(tokenizer=processor.tokenizer)

    # 🧠 Clear memory
    torch.cuda.empty_cache()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=None,  # ✅ avoids FutureWarning
    )

    print("🚀 Training started...")
    trainer.train()

    model.save_pretrained("./blip-caption-model")
    processor.save_pretrained("./blip-caption-model")
    print("✅ Training complete and model saved!")


if __name__ == "__main__":
    train_model()
