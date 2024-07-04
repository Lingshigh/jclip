import jittor as jt
from PIL import Image
import jclip as clip
import os
from tqdm import tqdm
import argparse

# Set to use CUDA if available
jt.flags.use_cuda = 1

# Argument parser for command-line options
parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='A')
args = parser.parse_args()

# Load the CLIP model and preprocess function
model, preprocess = clip.load("ViT-B-32.pkl")

# Read class names from the file and process them
classes = open('Dataset/classes.txt').read().splitlines()
new_classes = []
for c in classes:
    c = c.split(' ')[0]
    if c.startswith('Animal'):
        c = c[7:]
    if c.startswith('Thu-dog'):
        c = c[8:]
    if c.startswith('Caltech-101'):
        c = c[12:]
    if c.startswith('Food-101'):
        c = c[9:]
    c = 'a photo of ' + c
    new_classes.append(c)

# Tokenize the class names and encode them into text features
text = clip.tokenize(new_classes)
text_features = model.encode_text(text)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Directory for images based on the split
split = 'TestSet' + args.split
imgs_dir = 'Dataset/' + split
imgs = os.listdir(imgs_dir)

# Open the result file for writing predictions
with open('result.txt', 'w') as save_file:
    preds = []
    with jt.no_grad():
        for img in tqdm(imgs):
            img_path = os.path.join(imgs_dir, img)
            try:
                # Load and preprocess the image
                image = Image.open(img_path)
                image = preprocess(image).unsqueeze(0)

                # Encode the image into features
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # Calculate text probabilities and get top 5 predictions
                text_probs = (100.0 * image_features @ text_features.transpose(0, 1)).softmax(dim=-1)
                _, top_labels = text_probs[0].topk(5)
                preds.append(top_labels)

                # Save top 5 predictions to the file
                save_file.write(img + ' ' + ' '.join([str(p.item()) for p in top_labels]) + '\n')
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

print("Processing completed. Predictions saved to result.txt")