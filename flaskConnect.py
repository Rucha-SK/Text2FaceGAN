from flask import Flask, request, jsonify, make_response
import torch
import torch.nn as nn
from PIL import Image
from io import BytesIO
import base64
import sys
import os

from flask_cors import CORS

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import logging

# Configure logging to output to the terminal
logging.basicConfig(level=logging.INFO)

project_folder = os.path.abspath("Text2FaceGAN/FGTD/scripts")

# Append the path
sys.path.append(project_folder)

print(sys.path) 

from FGTD.scripts.preprocess.dataset import get_weighted_dataloader
from FGTD.scripts.preprocess.extract_zip import extract_zip
from FGTD.scripts.text_encoder.sentence_encoder import SentenceEncoder

sentence_encoder = SentenceEncoder('cpu')

class Generator(nn.Module):
    '''
    The Generator Network
    '''

    def __init__(self, noise_size, feature_size, num_channels, embedding_size, reduced_dim_size):
        super(Generator, self).__init__()
        self.reduced_dim_size = reduced_dim_size

        self.projection = nn.Sequential(
            nn.Linear(in_features = embedding_size, out_features = reduced_dim_size),
            nn.BatchNorm1d(num_features = reduced_dim_size),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True)
        )

        self.layer = nn.Sequential(
            nn.ConvTranspose2d(noise_size + reduced_dim_size, feature_size * 8, 4, 1, 0, bias = False),
            nn.BatchNorm2d(feature_size * 8),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),

            # state size (ngf*4) x 4 x 4
            nn.ConvTranspose2d(feature_size * 8, feature_size * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(feature_size * 4),
            nn.ReLU(True),

            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(feature_size * 4, feature_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size * 2),
            nn.ReLU(True),

            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(feature_size * 2, feature_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(True),

            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(feature_size, feature_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(True),

            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(feature_size, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()

        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.0002, betas = (0.5, 0.5))

    def forward(self, noise, text_embeddings):
        encoded_text = self.projection(text_embeddings)
        print("encoded text shape:" ,encoded_text.shape, flush=True)
        concat_input = torch.cat([noise, encoded_text], dim = 1).unsqueeze(2).unsqueeze(2)
        print("concat_input shape:" ,concat_input.shape, flush=True)
        output = self.layer(concat_input)
        return output


# Generator2
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels)
        )

    def forward(self, x):
        return x + self.conv(x)

class Generator2(nn.Module):
    def __init__(self, noise_size, feature_size, num_channels, embedding_size, reduced_dim_size):
        super(Generator2, self).__init__()

        # Efficient text feature processing
        self.projection = nn.Sequential(
            nn.Linear(embedding_size, reduced_dim_size),
            nn.LayerNorm(reduced_dim_size),
            nn.LeakyReLU(0.2)
        )

        # Efficient image processing with focus on details
        self.image_encoder = nn.Sequential(
            nn.Conv2d(num_channels, feature_size, 3, 1, 1),
            nn.GroupNorm(8, feature_size),  # More memory efficient than InstanceNorm
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_size, feature_size * 2, 3, 1, 1),
            nn.GroupNorm(8, feature_size * 2),
            nn.LeakyReLU(0.2)
        )

        # Efficient detail enhancement
        self.detail_branch = nn.Sequential(
            nn.Conv2d(feature_size * 2, feature_size * 4, 3, 1, 1),
            nn.GroupNorm(8, feature_size * 4),
            nn.LeakyReLU(0.2),

            # Single residual connection for detail preservation
            ResidualBlock(feature_size * 4),

            nn.Conv2d(feature_size * 4, feature_size * 2, 3, 1, 1),
            nn.GroupNorm(8, feature_size * 2),
            nn.LeakyReLU(0.2)
        )

        # Lightweight attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(feature_size * 2 + reduced_dim_size, feature_size * 2, 1, 1),
            nn.Sigmoid()
        )

        # Final refinement
        self.output_conv = nn.Sequential(
            nn.Conv2d(feature_size * 2, feature_size, 3, 1, 1),
            nn.GroupNorm(8, feature_size),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_size, num_channels, 3, 1, 1),
            nn.Tanh()
        )

        # Learnable detail enhancement weight
        self.detail_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_size * 2, 1, 1),
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')  # Fixed the typo
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # Fixed the initialization for bias

    def forward(self, image, text_embeddings):
        # Process text embeddings
        text_features = self.projection(text_embeddings)
        text_features = text_features.view(-1, text_features.size(1), 1, 1)
        text_features = text_features.expand(-1, -1, image.size(2), image.size(3))

        # Extract and enhance image features
        image_features = self.image_encoder(image)
        enhanced_features = self.detail_branch(image_features)

        # Apply attention with text features
        combined = torch.cat([enhanced_features, text_features], dim=1)
        attention_weights = self.attention(combined)
        attended_features = enhanced_features * attention_weights

        # Generate refined output
        refined = self.output_conv(attended_features)

        # Dynamic residual connection
        detail_weight = torch.sigmoid(self.detail_weight(attended_features)) * 0.5
        output = (1 - detail_weight) * image + detail_weight * refined

        return output

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels)
        )

    def forward(self, x):
        return x + self.conv(x)
noise_size = 100  # Adjust according to your noise size
feature_size = 64  # Adjust as per your model architecture
num_channels = 3  # Change if generating grayscale images
embedding_size = 768  # Change based on your text embedding size
reduced_dim_size = 256  # Change based on your architecture

# Create an instance of your Generator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# generator = Generator(noise_size, feature_size, num_channels, embedding_size, reduced_dim_size)

generator = Generator(100,128, 3, 768, 256).to('cpu')
print("Generator structure:", generator)
generator.load_state_dict(torch.load('generator_checkpoint_epoch_14.pt', map_location=torch.device('cpu')))
generator.eval()

generator2 = Generator2(100, 128, 3, 768, 256).to(device)
generator2.load_state_dict(torch.load('generator2_epoch20.pt', map_location=device))
generator2.eval()


app = Flask(__name__)

# Configure CORS
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('flask_cors').level = logging.DEBUG

def generate_image_from_text(text_description):
    # with torch.no_grad():
    #     test_noise = torch.randn(size=(1, 100)).to('cpu')
    #     test_embeddings = sentence_encoder.convert_text_to_embeddings([text_description])
    #     print(f"Test noise shape: {test_noise.shape}")
    #     print(f"Text embeddings shape: {test_embeddings.shape}")
    #     test_embeddings = test_embeddings.to('cpu')  # Change if using GPU
    #     image_tensor = generator(test_noise, test_embeddings).detach()
    #     print(f"Image tensor shape: {image_tensor.shape}")
    #     print(f"Max value in image tensor: {image_tensor.max()}, Min value in image tensor: {image_tensor.min()}")
    #     generated_image = transforms.ToPILImage()((image_tensor.squeeze() * 0.5 + 0.5).clamp(0, 1).cpu())
    # return generated_image

    with torch.no_grad():
        noise = torch.randn(1, 100).to(device)
        text_embeddings = sentence_encoder.convert_text_to_embeddings([text_description]).to(device)
        print(f"Test noise shape: {noise.shape}")
        print(f"Text embeddings shape: {text_embeddings.shape}")
        initial_image = generator(noise, text_embeddings).detach()
        refined_image = generator2(initial_image, text_embeddings).detach()
        print(f"Max value in image tensor: {refined_image.max()}, Min value in image tensor: {refined_image.min()}")
        output_image = transforms.ToPILImage()((refined_image.squeeze() * 0.5 + 0.5).clamp(0, 1).cpu())
    return output_image

@app.after_request
def add_header(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/')
def home():
    return "Hello, Flask!"    

@app.route('/generate-image', methods=['POST','OPTIONS'])
def generate_image():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "http://localhost:3000")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response

    data = request.get_json()
    text_description = data['description']
    image = generate_image_from_text(text_description)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    response = jsonify({'image_url': f"data:image/jpeg;base64,{img_str}"})
    return response

if __name__ == "__main__":
    app.run(debug=True, port=5000)
