import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Dataset definition
class OasisDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Recursively gather all .png images from the subdirectories
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.png'):
                    self.samples.append(os.path.join(dirpath, filename))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name = self.samples[idx]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image

# Data preparation
transform = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

oasis_data = OasisDataset("/Users/abhinavsingh/Desktop/Pattern-and-Machine-learning", transform)
data_loader = DataLoader(oasis_data, batch_size=32, shuffle=True)

# VAE model definition
class VAE(nn.Module):
    def __init__(self, h_dim, z_dim):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(256*256, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)

        # Decoder
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, 256*256)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        return torch.clamp(torch.sigmoid(self.fc4(h)), 0, 1)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 256*256))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(400, 20).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 256*256), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def visualize_samples(data_loader, model, num_samples=5):
    model.eval()
    data_iter = iter(data_loader)
    data = next(data_iter).to(device)

    with torch.no_grad():
        recon, _, _ = model(data)

        data = data.cpu().numpy()
        recon = recon.cpu().numpy()

        fig, axs = plt.subplots(2, num_samples, figsize=(15, 6))
        for i in range(num_samples):
            axs[0, i].imshow(data[i].reshape(256, 256), cmap='gray')
            axs[1, i].imshow(recon[i].reshape(256, 256), cmap='gray')
            axs[0, i].axis('off')
            axs[1, i].axis('off')

        axs[0, 0].set_title('Original Images')
        axs[1, 0].set_title('Reconstructed Images')
        plt.tight_layout()
        plt.show()

# Training function
def train(epoch, epochs):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(data_loader.dataset),
            100. * batch_idx / len(data_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_loader.dataset)))

epochs = 10
for epoch in range(1, epochs + 1):
    train(epoch, epochs)
    visualize_samples(data_loader, model)  # Visualize after each epoch

