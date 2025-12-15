import lightgbm as lgb
import torch
from models.NCF import NeuralCollaborativeFiltering
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class NCFWrapper:
    """
    args 없이 직접 하이퍼파라미터를 전달하여 NeuralCollaborativeFiltering을 래핑
    """
    def __init__(self, field_dims, embed_dim=16, mlp_dims=[16,32],
                 batchnorm=True, dropout=0.2,
                 epochs=5, batch_size=1024, lr=1e-3, weight_decay=0.0, device=None):
        
        # 모델 하이퍼파라미터
        self.model_kwargs = {
            "field_dims": field_dims,
            "embed_dim": embed_dim,
            "mlp_dims": mlp_dims,
            "batchnorm": batchnorm,
            "dropout": dropout
        }

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._is_fitted = False
        self._model = None
        self.field_dims=field_dims
    def _build_model(self):
        # NeuralCollaborativeFiltering 수정 필요:
        # args 대신 **kwargs 사용
        model = NeuralCollaborativeFiltering(self.model_kwargs, self.field_dims)
        return model.to(self.device)

    def to(self, device):
        self._model.to(device)

    def fit(self, X_train, y_train, X_valid, y_valid,
        epochs=5, batch_size=1024, device=None, loss_fn=None, verbose=True):
    
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._build_model()
        loss_fn =  torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)

        train_dataset = TensorDataset(torch.LongTensor(X_train), torch.FloatTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_valid is not None and y_valid is not None:
            valid_loader = DataLoader(TensorDataset(torch.LongTensor(X_valid), torch.FloatTensor(y_valid)),
                                    batch_size=batch_size, shuffle=False)
        else:
            valid_loader = None

        for epoch in range(epochs):
            self._model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = self._model(xb).view(-1)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg_train_loss = total_loss / len(train_loader.dataset)

            
            msg = f"Epoch {epoch+1}/{epochs} Train Loss: {avg_train_loss:.4f}"
            if valid_loader is not None:
                self._model.eval()
                val_preds = []
                with torch.no_grad():
                    for xb, yb in valid_loader:
                        xb = xb.to(device)
                        val_preds.append(self._model(xb).view(-1).cpu().numpy())
                val_preds = np.concatenate(val_preds)
                val_loss = np.sqrt(((val_preds - y_valid)**2).mean())
                msg += f" | Valid RMSE: {val_loss:.4f}"
            print(msg)
        self._is_fitted=True
        return self


    
    def _predict_model(self, loader):
        self._model.eval()
        preds = []
        with torch.no_grad():
            for xb in loader:
                if isinstance(xb, (list, tuple)):
                    xb = xb[0]
                xb = xb.to(self.device)
                out = self._model(xb).view(-1).cpu().numpy()
                preds.append(out)
        return np.concatenate(preds, axis=0)

    def predict(self, X):
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        test_loader = DataLoader(TensorDataset(torch.LongTensor(X)), batch_size=self.batch_size, shuffle=False)
        return self._predict_model(test_loader)
