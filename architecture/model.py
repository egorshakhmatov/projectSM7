"""GRU for three nonnegative delay forecasts, independent of CSV loading."""
import io
import torch
from torch import nn


class TrafficGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(3, 32, num_layers=1, batch_first=True)
        self.output = nn.Linear(32, 3)

    def forward(self, history):
        if history.ndim != 3 or history.shape[1:] != (8, 3):
            raise ValueError("Ожидается вход [размер пакета, 8, 3].")
        _, hidden = self.gru(history)
        return nn.functional.softplus(self.output(hidden[0]))


def verify_model():
    """Technical test on fixed tensors; no dataset or traffic forecast."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = TrafficGRU().cpu()
        parameters = sum(p.numel() for p in model.parameters())
        if parameters != 3651:
            raise RuntimeError(f"Неверное число параметров: {parameters}")
        history = torch.linspace(0, 4, 48, dtype=torch.float32).reshape(2, 8, 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        loss = nn.MSELoss()(model(history), torch.tensor([[0.5, 1.0, 2.0], [1.5, 0.7, 1.0]]))
        loss.backward()
        optimizer.step()
        if not any(not torch.equal(value, before[key]) for key, value in model.state_dict().items()):
            raise RuntimeError("Веса не изменились после шага обучения.")
        model.eval()
        with torch.inference_mode():
            expected = model(history)
        if not torch.isfinite(expected).all() or (expected < 0).any():
            raise RuntimeError("Неверные прогнозируемые задержки.")
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = TrafficGRU().cpu()
        restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
        restored.eval()
        with torch.inference_mode():
            if not torch.equal(expected, restored(history)):
                raise RuntimeError("Результат изменился после загрузки весов.")
        return {"parameters": parameters, "device": "cpu", "output_shape": list(expected.shape), "training_step": True, "weights_roundtrip": True, "forecast": False}
