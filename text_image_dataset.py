import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class TextImageDataset(Dataset):
    """
    Dataset for text-conditioned image generation.

    Expected directory structure:
        data_path/
            images/
                img_0001.jpg
                img_0002.jpg
                ...
            captions/
                img_0001.txt
                img_0002.txt
                ...

    Each .txt file should contain a single line of text caption in Japanese or English.
    """

    def __init__(self, data_path, transform=None, tokenizer=None, max_text_len=77):
        """
        Args:
            data_path: Root directory containing 'images' and 'captions' folders
            transform: Torchvision transforms for images
            tokenizer: Transformers tokenizer for text
            max_text_len: Maximum text sequence length
        """
        self.data_path = data_path
        self.images_dir = os.path.join(data_path, 'images')
        self.captions_dir = os.path.join(data_path, 'captions')
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

        # Get list of image files
        self.image_files = sorted([f for f in os.listdir(self.images_dir)
                                   if f.endswith(('.jpg', '.jpeg', '.png', '.JPEG'))])

        print(f"Found {len(self.image_files)} images in {self.images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Load caption
        caption_name = os.path.splitext(img_name)[0] + '.txt'
        caption_path = os.path.join(self.captions_dir, caption_name)

        if os.path.exists(caption_path):
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
        else:
            # Default caption if file doesn't exist
            caption = ""
            print(f"Warning: Caption file not found: {caption_path}")

        # Tokenize text
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                caption,
                padding='max_length',
                truncation=True,
                max_length=self.max_text_len,
                return_tensors='pt'
            )
            input_ids = tokens['input_ids'].squeeze(0)
            attention_mask = tokens['attention_mask'].squeeze(0)
        else:
            # Return dummy tokens if tokenizer not provided
            input_ids = torch.zeros(self.max_text_len, dtype=torch.long)
            attention_mask = torch.zeros(self.max_text_len, dtype=torch.long)

        return image, input_ids, attention_mask


class ImageNetWithCaptions(Dataset):
    """
    Wrapper for ImageNet dataset that generates simple captions from class names.
    Useful for quick testing with existing ImageNet data.

    This generates captions like "a photo of a [class_name]" in Japanese.
    """

    def __init__(self, imagenet_dataset, tokenizer, max_text_len=77, class_names=None):
        """
        Args:
            imagenet_dataset: torchvision.datasets.ImageFolder dataset
            tokenizer: Transformers tokenizer
            max_text_len: Maximum text sequence length
            class_names: Optional dict mapping class indices to Japanese names
        """
        self.imagenet_dataset = imagenet_dataset
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.class_names = class_names

    def __len__(self):
        return len(self.imagenet_dataset)

    def __getitem__(self, idx):
        image, class_idx = self.imagenet_dataset[idx]

        # Generate simple caption from class index
        if self.class_names and class_idx in self.class_names:
            class_name = self.class_names[class_idx]
        else:
            class_name = f"クラス{class_idx}"

        caption = f"{class_name}の写真"

        # Tokenize
        tokens = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_len,
            return_tensors='pt'
        )
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)

        return image, input_ids, attention_mask
