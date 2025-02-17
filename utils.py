import torch

def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
    print(f"Model loaded from {filename}")
