import torch
import torch.nn as nn
import os

# 1. Re-declare Model Architecture
class LiveCallConv1dDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=80, stride=4, padding=38),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)

# 2. Load PyTorch weights
model_path = r"models\master_tier1.pth"  # or live_call_conv1d_model.pth
onnx_output_path = r"models\streaming_model_int8.onnx"

print(f"📦 Loading PyTorch weights from: {model_path}")
model = LiveCallConv1dDetector()
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# 3. Create dummy input tensor matching expected window size (Batch=1, Channels=1, Samples=1600)
dummy_input = torch.randn(1, 1, 1600)

print(f"⚙️ Exporting model to ONNX format at: {onnx_output_path}")
os.makedirs("models", exist_ok=True)

torch.onnx.export(
    model,
    dummy_input,
    onnx_output_path,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("✅ ONNX model exported successfully!")