import neuranet as nnt
import neuranet.nn as nn


# define the train dataset : 
X_train = nnt.rand(1000, 3)
y_train = nnt.Tensor(nnt.dot(X_train, nnt.Tensor([1, -2, 1]).T))

# build model using Sequential: 
model = nn.Sequential(
    nn.Linear(3, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

# define the loss function: 
mse = nn.MSELoss(model.layers())

# define the optimizer: 
opt = nnt.optim.GD(model.layers(), lr=0.1)

# traning loop: 
for epoch in range(100):

    for i in range(0, 1000, 10):

        opt.zero_grad()

        y_predi = model(nnt.Tensor(X_train[i:i+10]))

        loss = mse(y_predi, nnt.Tensor(y_train[i:i+10]))

        if i%1000 == 0:
            print(loss.item())

        loss.backward()

        opt.step()


layers = model.layers()

print("\nModel Parameters:")
print(model(1))

# Print the true parameters:
print("\nTrue Parameters:")
print(nnt.Tensor([1, -2, 1]).T)

