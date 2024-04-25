import torch


from HySpecLab.metrics import UnmixingLoss
from HySpecLab.metrics.regularization import SimilarityLoss
from HySpecLab.unmixing import ContrastiveUnmixing
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def train(model:ContrastiveUnmixing, n_endmembers:int, dataset:Dataset, n_batchs:int = 64, n_epochs:int = 100, lr=1e-3, similarity_weight=1, sparse_weight=1e-1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = UnmixingLoss() 
    similarity_reg = SimilarityLoss(n_endmembers, temperature=.1, reduction='mean')

    dataloader = DataLoader(dataset, batch_size=int(len(dataset)/n_batchs), shuffle=True)

    epoch_iterator = tqdm(
            range(n_epochs),
            leave=True,
            unit="epoch",
            postfix={"tls": "%.4f" % -1},
        )

    for _ in epoch_iterator:
        epoch_loss = 0.
        for i, (x) in enumerate(dataloader):
            x = x.to(device)
            optimizer.zero_grad()
            y = model(x)
            sparse_reg = model.sparse_gate.regularize() if model.sparse_gate is not None else 0
            loss = (2*criterion(y, x)) + similarity_weight*similarity_reg(model.ebk) +  sparse_weight*sparse_reg
            epoch_loss += loss.detach().item()

            loss.backward()
            optimizer.step()

        epoch_iterator.set_postfix(tls="%.4f" % (epoch_loss/(i+1)))