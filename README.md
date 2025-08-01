# ImgScribe

ImgScribe — A Smart Image Caption Generator using BLIP

Objective:
To build a deep learning application that automatically generates meaningful captions for images using a pretrained BLIP (Bootstrapped Language Image Pretraining) model and the Flickr8k dataset. The model is trained and deployed with an aesthetically enhanced Streamlit GUI.

Tools & Technologies Used:

Dataset: Flickr8k (Images + Captions)
Model: Salesforce/blip-image-captioning-base (Pretrained)
Frameworks/Libraries:
PyTorch
Hugging Face Transformers
scikit-learn
PIL (Python Imaging Library)
Streamlit
Device: GPU (8GB VRAM) using CUDA
Interface: Streamlit GUI (Dark Themed)

Folder Structure:
Image Caption Generator/
│
├── Flickr8k_Dataset/
│   ├── Images/                 # 8000 images
│   └── captions.txt            # Cleaned image-caption pairs (pipe-separated)
│
├── blip-caption-model/         # Saved model and processor after training
│
├── model.py                    # Training script
├── app.py                      # Streamlit GUI code
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

Features:
Uses BLIP model with vision-language pretraining
Fine-tuned on Flickr8k with ~6000 samples
Custom PyTorch Dataset for handling captions and images
GPU-accelerated training with memory optimization
Image caption generation using GUI
Streamlit-based dark-themed interface
Fully offline and local deployment possible

model.py Summary:
Loads and preprocesses dataset
Uses Hugging Face Trainer for training
Optimized for GPU with memory tweaks (expandable segments, lower max_length, batch size = 1)
Trains and saves BLIP model locally

app.py Summary (GUI):
Allows uploading of image files (.jpg/.png)
Preprocesses image and generates caption using the trained BLIP model
Modern UI using Streamlit with dark theme
Display of image and predicted caption

How to Run:
Install dependencies:
pip install -r requirements.txt
Train the model (optional if model is already trained):

python model.py
💻 Run the app:
streamlit run app.py

Deployment Notes:
Can be hosted locally (http://localhost:8501)
To expose via HTTPS:
Use services like ngrok or Cloudflare Tunnel
Or deploy on platforms like Streamlit Cloud, Heroku, or Vercel with HTTPS support

**IN ORDER TO DOWNLOAD THE DATASET**

Visit the Official Dataset Link:

Go to https://forms.illinois.edu/sec/1713398

Fill out the short form to request access to the Flickr8k Dataset.

Download the Files:
After submitting the form, you’ll receive access to download the following:

Flickr8k_Dataset.zip — contains 8091 images.

Flickr8k_text.zip — contains caption files.

Extract the Files:

Unzip Flickr8k_Dataset.zip to a folder named Flickr8k_Dataset/Images

Unzip Flickr8k_text.zip to get Flickr8k.token.txt or Flickr8k.caption.txt

Prepare captions.txt:

Create a new file named captions.txt with the format:
filename.jpg|caption
Example:
1000268201_693b08cb0e.jpg|A child in a pink dress climbs stairs.
