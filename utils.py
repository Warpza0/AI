def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image)

def load_model(model_path, device):
    model = NeuralNet(input_size=28*28, num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate_model(model, data_loader, device):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    
    acc = float(num_correct) / float(num_samples) * 100
    return acc