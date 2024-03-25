from urllib.request import urlopen
from PIL import Image
import timm
import torch
import matplotlib.pyplot as plt

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)

# Assuming 'model' is already defined and loaded with pretrained weights
for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}")


# Example: Print the actual values of the weights for a specific layer
layer_name = ['blocks.11.mlp.fc1.weight', 'blocks.11.mlp.fc1.bias',
               'blocks.11.mlp.fc2.weight', 'blocks.11.mlp.fc2.bias']


  # Example layer name
for name, param in model.named_parameters():
    if name in layer_name:
        print(f"{name}: values=\n{param.data}")

layer_name = 'blocks.0.attn.qkv.weight'  # Example layer name
for name, param in model.named_parameters():
    if name in layer_name:
        # Flatten the tensor to 1D for histogram plotting
        values = param.data.cpu().numpy().flatten()
        plt.hist(values, bins=100)
        plt.title(f"Histogram of weights for {name}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()
