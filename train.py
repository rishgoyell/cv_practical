import time
import torch.optim as optim

from data import circlesData
from models import circleParametrizer

torch.manual_seed(111)

def get_train_val_test_split(datasize):
    datainds = torch.randperm(datasize)
    split = [int(datasize * 0.8), int(datasize * 0.9)]
    return datainds[:split[0]], datainds[split[0]:split[1]], datainds[split[1]:]

def run_epoch(M, DL, train=True):
    M.eval()
    if train:
        M.train()

    # epoch stats
    num_batches = 0
    total_loss = 0

    for X, Y in DL:
        X = X.to(device).unsqueeze(1)
        Y = Y.to(device)

        # forward pass
        if train:
            pred_params = M(X, Y)
        else:
            with torch.no_grad():
                pred_params = M(X, Y)
        loss = criterion(pred_params, Y)

        # gradient step
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss/num_batches


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # define model
    CP = circleParametrizer(spatial=True, device=device).to(device)

    # define datasets
    traininds, valinds, testinds = get_train_val_test_split(50000)
    traindata = circlesData('/scratch/rgoyal6/scale/data.pth', traininds)
    valdata = circlesData('/scratch/rgoyal6/scale/data.pth', valinds)
    testdata = circlesData('/scratch/rgoyal6/scale/data.pth', testinds)

    # define dataloaders
    trainloader = DataLoader(traindata, batch_size=32, shuffle=True, num_workers=2)
    valloader = DataLoader(valdata, batch_size=32, shuffle=False, num_workers=2)
    testloader = DataLoader(testdata, batch_size=32, shuffle=False, num_workers=2)

    # peripherals
    num_epochs = 150
    lr = 0.001
    criterion = nn.MSELoss()
    optimizer = optim.Adam(CP.parameters(), lr=lr, weight_decay=0.0005)

    total_loss = 0
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        start_time = time.time()
        loss = run_epoch(CP, trainloader, train=True)
        val_loss = run_epoch(CP, testloader, train=False)
        if best_val_loss > val_loss and epoch > 20:
            print(f'Model saved at epoch {epoch}, val loss improved from {best_val_loss} to {val_loss}')
            best_val_loss = val_loss
            state = {'model': CP.state_dict(), 'optimizer':optimizer.state_dict(), 'loss': best_val_loss, 'epoch': epoch}
        print(f'Epoch: {epoch}; Time: {time.time() - start_time : .2f}; Train Loss: {loss: .4f}; Val Loss: {val_loss : .4f}')

    # performance on test
    CP.load_state_dict(state['model'])
    test_loss = run_epoch(CP, testloader, train=False)
    print(f'Test Loss: {test_loss}')

    # save best model
    torch.save(state, 'model.pth')
