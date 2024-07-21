import torch
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from enhanced_vae_model import loss_function  # Import the loss function from your separate file

class VAETrainer:
    def __init__(self, model, optimizer, data_loader, device, kld_weight=1.0):
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.device = device
        self.kld_weight = kld_weight

    def train(self, epochs, log_interval=10):
        self.model.train()
        train_losses = []
        total_batches = len(self.data_loader) * epochs
        
        with tqdm(total=total_batches, desc='Training', unit='batch') as pbar:
            for epoch in range(1, epochs + 1):
                total_loss = 0
                for batch_idx, (data, _) in enumerate(self.data_loader):
                    data = data.to(self.device)
                    self.optimizer.zero_grad()
                    recon_batch, mu, logvar = self.model(data)
                    loss = loss_function(recon_batch, data, mu, logvar, self.kld_weight)
                    loss.backward()
                    total_loss += loss.item()
                    self.optimizer.step()
                    
                    if (batch_idx + 1) % log_interval == 0:
                        pbar.set_postfix({'loss': loss.item() / len(data)})
                    pbar.update(1)
                
                average_loss = total_loss / len(self.data_loader.dataset)
                train_losses.append(average_loss)
                pbar.set_description(f'Epoch {epoch}/{epochs}')
        
        return train_losses

class LossPlotter:
    @staticmethod
    def plot_losses(losses, scale='log'):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(losses) + 1), losses, marker='o')
        plt.title(f'Average Loss per Epoch ({scale.capitalize()} Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.yscale(scale)
        plt.grid(True, which="both", ls="--")
        plt.show()

# Example usage
if __name__ == "__main__":
    # Assuming model, optimizer, train_loader, and device are already defined
    trainer = VAETrainer(model, optimizer, train_loader, device, kld_weight=1.0)
    train_losses = trainer.train(epochs=100, log_interval=50)
    
    plotter = LossPlotter()
    plotter.plot_losses(train_losses, scale='linear')  # Change 'linear' to 'log' for log scale