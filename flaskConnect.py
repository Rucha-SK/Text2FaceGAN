from flask import Flask, request, jsonify
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



project_folder = os.path.abspath("/T2FFrontend/FGTD/scripts")

# Append the path
sys.path.append(project_folder)

print(sys.path) 

from preprocess import get_weighted_dataloader, extract_zip
from text_encoder.sentence_encoder import SentenceEncoder

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

noise_size = 100  # Adjust according to your noise size
feature_size = 64  # Adjust as per your model architecture
num_channels = 3  # Change if generating grayscale images
embedding_size = 768  # Change based on your text embedding size
reduced_dim_size = 256  # Change based on your architecture

# Create an instance of your Generator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(noise_size, feature_size, num_channels, embedding_size, reduced_dim_size)
# Load model with weights_only set to True
# Load the state dict and filter out non-matching keys
#pretrained_dict = torch.load("generator_checkpoint_epoch_14.pt", map_location=device)



#model_dict = generator.state_dict()

# Filter out unnecessary keys
#pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}

# Overwrite entries in the existing state dict
#model_dict.update(pretrained_dict) 
# Load the new state dict
#generator.load_state_dict(model_dict)
#generator.load_state_dict(torch.load("generator_checkpoint_epoch_14.pt", map_location='cpu', weights_only=True))

#generator.eval()  # Set the generator to evaluation mode

generator = Generator(100,128, 3, 768, 256).to('cpu')
print("Generator structure:", generator)
generator.load_state_dict(torch.load('generator_checkpoint_epoch_14.pt', map_location=torch.device('cpu')))

generator.eval()


app = Flask(__name__)

CORS(app)

def generate_image_from_text(text_description):
    with torch.no_grad():
        test_noise = torch.randn(size=(1, 100)).to('cpu')
        test_embeddings = sentence_encoder.convert_text_to_embeddings([text_description])
        print(f"Test noise shape: {test_noise.shape}")
        print(f"Text embeddings shape: {test_embeddings.shape}")
        test_embeddings = test_embeddings.to('cpu')  # Change if using GPU

        image_tensor = generator(test_noise, test_embeddings).detach()

        print(f"Image tensor shape: {image_tensor.shape}")
        print(f"Max value in image tensor: {image_tensor.max()}, Min value in image tensor: {image_tensor.min()}")

        generated_image = transforms.ToPILImage()((image_tensor.squeeze() * 0.5 + 0.5).clamp(0, 1).cpu())
    
    return generated_image

@app.route('/')
def home():
    return "Hello, Flask!"    

@app.route('/generate-image', methods=['POST'])
def generate_image():
    print('Received request to generate image.',file=sys.stdout)
    data = request.get_json()
    text_description = data['description']
    
    # Generate the image
    image = generate_image_from_text(text_description)
    
    # Save or convert image to base64
    image.save("generated_image.jpg")  
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({'image_url': f"data:image/jpeg;base64,{img_str}"})

if __name__ == "__main__":
    app.run(debug=False)
