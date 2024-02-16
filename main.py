import neuranet as nnt
import neuranet.nn as nn


# # define the train dataset : 
# X_train = nnt.rand(1000, 3)
# y_train = nnt.Tensor(nnt.dot(X_train, nnt.Tensor([1, -2, 1]).T))

X_train = nnt.Tensor([[0, 0], 
                      [0, 1], 
                      [1, 0], 
                      [1, 1]])

y_train = nnt.Tensor([0, 1, 1, 0]).T 

# build model using Sequential: 
model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

# define the loss function: 
mse = nn.MSELoss(model.layers())

# define the optimizer: 
opt = nnt.optim.GD(model.layers(), lr=0.1)

# traning loop: 
for epoch in range(1000):

    for i in range(4):
        opt.zero_grad()

        y_predi = model(nnt.Tensor(X_train[i]))

        loss = mse(y_predi, nnt.Tensor(y_train[i]))

        if epoch%10 == 0:
            print(loss.item())

        loss.backward()

        opt.step()


layers = model.layers()

print("\nModel Parameters:")
print(model(X_train))

# Print the true parameters:
print("\nTrue Parameters:")
print(nnt.Tensor([1, -2, 1]).T)

