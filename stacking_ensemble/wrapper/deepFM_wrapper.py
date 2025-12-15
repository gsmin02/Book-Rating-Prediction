import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models.DeepFM import DeepFM 

class DeepFMWrapper:
    """
    DeepFM 모델을 래핑하여 fit/predict 지원
    """
    def __init__(self, field_dims, embed_dim=16, mlp_dims=[16,32],
                 batchnorm=True, dropout=0.2,
                 epochs=5, batch_size=1024, lr=1e-3, weight_decay=0.0, device=None):
        
        self.model_kwargs = {
            "embed_dim": embed_dim,
            "mlp_dims": mlp_dims,
            "batchnorm": batchnorm,
            "dropout": dropout
        }
        self.field_dims = field_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._is_fitted = False
        self._model = None

    def _build_model(self):
        model = DeepFM(self.model_kwargs, self.field_dims)
        return model.to(self.device)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, verbose=True):
        self._model = self._build_model()
        self._model.train()

        loss_fn = torch.nn.MSELoss()  # RMSE는 MSE 후 sqrt
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        train_dataset = TensorDataset(torch.LongTensor(X_train), torch.FloatTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if X_valid is not None and y_valid is not None:
            valid_loader = DataLoader(TensorDataset(torch.LongTensor(X_valid), torch.FloatTensor(y_valid)),
                                      batch_size=self.batch_size, shuffle=False)
        else:
            valid_loader = None

        for epoch in range(self.epochs):
            self._model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                preds = self._model(xb).view(-1)
                loss = torch.sqrt(loss_fn(preds, yb))  # RMSE
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)

            avg_train_loss = total_loss / len(train_loader.dataset)
            msg = f"Epoch {epoch+1}/{self.epochs} Train RMSE: {avg_train_loss:.4f}"

            if valid_loader is not None:
                self._model.eval()
                val_preds = []
                with torch.no_grad():
                    for xb, yb in valid_loader:
                        xb = xb.to(self.device)
                        val_preds.append(self._model(xb).view(-1).cpu().numpy())
                val_preds = np.concatenate(val_preds)
                val_loss = np.sqrt(((val_preds - y_valid)**2).mean())
                msg += f" | Valid RMSE: {val_loss:.4f}"
            
            if verbose:
                print(msg)

        self._is_fitted = True
        return self

    def _predict_model(self, model, loader):
        model.eval()
        preds = []
        with torch.no_grad():
            for xb in loader:
                if isinstance(xb, (list, tuple)):
                    xb = xb[0]
                xb = xb.to(self.device)
                out = model(xb).view(-1).cpu().numpy()
                preds.append(out)
        return np.concatenate(preds, axis=0)

    def predict(self, X):
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        test_loader = DataLoader(TensorDataset(torch.LongTensor(X)), batch_size=self.batch_size, shuffle=False)
        return self._predict_model(self._model, test_loader)
