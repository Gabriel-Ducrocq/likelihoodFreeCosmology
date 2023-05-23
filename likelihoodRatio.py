import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from bridge import BrownianBridge
import math
import matplotlib.pyplot as plt


def loss_function_classif(predicted_proba, true_labels):
    """
    label 1 if data has the right distribution, 0 otherwise
    """
    n_pos = torch.sum(true_labels)
    n_neg = torch.sum(1-true_labels)
    return -(torch.sum(torch.log(predicted_proba+1e-15)*true_labels)/(n_pos+1e-10) + torch.sum(torch.log(1-predicted_proba+1e-15)*(1-true_labels))/(n_neg+1e-10))

class MLPClassif(nn.Module):
    def __init__(self, in_dim, out_dim, intermediate_dim, device, num_hidden_layers = 1):
        super(MLPClassif, self).__init__()
        self.flatten = nn.Flatten()
        self.num_hidden_layers = num_hidden_layers
        self.compression_layer = torch.nn.Linear(9996, 6)
        ##Using leaky RELU !
        ##Removing the activation of the first Layer !!
        self.input_layer = nn.Linear(in_dim, intermediate_dim, device=device)
        #self.input_layer.to(device)
        self.output_layer = nn.Sequential(nn.Dropout(0.0), nn.Linear(intermediate_dim, out_dim, device=device), torch.nn.Sigmoid())
        #self.output_layer.to(device)
        self.list_intermediate = [nn.Sequential(nn.Dropout(0.0), nn.Linear(intermediate_dim, intermediate_dim, device=device), torch.nn.LeakyReLU())
                             for _ in range(num_hidden_layers)]
        self.linear_relu_stack = nn.Sequential(*[layer for layer in self.list_intermediate])
        #self.linear_relu_stack.to(device)
        self.in_dim = in_dim


    def forward(self, cls, theta):
        #x = self.flatten(x)
        compressed_cls = self.compression_layer(cls)
        x = torch.concat([compressed_cls, theta], dim=-1)
        x = self.input_layer(x)
        hidden = self.linear_relu_stack(x)
        output = self.output_layer(hidden)
        #plt.hist(logits.detach().numpy().flatten())
        #plt.show()
        #print(logits)
        return output

class data_set_classif(Dataset):
    def __init__(self, theta, cls, labels):
        self.cls = cls
        self.theta = theta
        self.labels = labels

    def __len__(self):
        return len(self.cls)

    def __getitem__(self, index):
        return self.theta[index], self.cls[index], self.labels[index]

scale = torch.tensor(np.array([l*(l+1)/(2*np.pi) for l in range(2, 2501)]), dtype=torch.float32)
all_theta = []
all_cls_tt = []
all_cls_ee = []
all_cls_bb = []
all_cls_te = []
for i in range(1, 4):
    if i == 1:
        all_theta.append(np.load("data/polarizationGaussianPrior/all_theta.npy"))
        all_cls_tt.append(np.load("data/polarizationGaussianPrior/all_cls_tt_hat.npy"))
        all_cls_ee.append(np.load("data/polarizationGaussianPrior/all_cls_ee_hat.npy"))
        all_cls_bb.append(np.load("data/polarizationGaussianPrior/all_cls_bb_hat.npy"))
        all_cls_te.append(np.load("data/polarizationGaussianPrior/all_cls_te_hat.npy"))
        print(np.load("data/polarizationGaussianPrior/all_theta.npy").shape)
    else:
        all_theta.append(np.load("data/polarizationGaussianPrior/all_theta" + str(i) + ".npy"))
        all_cls_tt.append(np.load("data/polarizationGaussianPrior/all_cls_tt_hat" + str(i) + ".npy"))
        all_cls_ee.append(np.load("data/polarizationGaussianPrior/all_cls_ee_hat" + str(i) + ".npy"))
        all_cls_bb.append(np.load("data/polarizationGaussianPrior/all_cls_bb_hat" + str(i) + ".npy"))
        all_cls_te.append(np.load("data/polarizationGaussianPrior/all_cls_te_hat" + str(i) + ".npy"))
        print(np.load("data/polarizationGaussianPrior/all_theta" + str(i) + ".npy").shape)

all_cls_tt = np.vstack(all_cls_tt)
all_cls_ee = np.vstack(all_cls_ee)
all_cls_bb = np.vstack(all_cls_bb)
all_cls_te = np.vstack(all_cls_te)
all_theta = np.vstack(all_theta)

all_theta = torch.tensor(all_theta[:, :], dtype=torch.float32)
all_cls_tt = torch.tensor(all_cls_tt[:, 2:], dtype=torch.float32) * scale
all_cls_ee = torch.tensor(all_cls_ee[:, 2:], dtype=torch.float32) * scale
all_cls_bb = torch.tensor(all_cls_bb[:, 2:], dtype=torch.float32) * scale
all_cls_te = torch.tensor(all_cls_te[:, 2:], dtype=torch.float32) * scale
all_cls = torch.tensor(np.hstack([all_cls_tt, all_cls_ee, all_cls_bb, all_cls_te]), dtype=torch.float32)

training_theta = all_theta[:19369]
training_cls = all_cls[:19369]

test_theta = all_theta[19369:]
test_cls = all_cls[19369:]

all_cls = torch.tensor(np.hstack([all_cls_tt, all_cls_ee, all_cls_bb, all_cls_te]), dtype=torch.float32)

COSMO_PARAMS_MEAN_PRIOR = np.array([0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561]) # Prior mean
COSMO_PARAMS_SIGMA_PRIOR = np.array([0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071]) # Prior std

fake_theta_train = np.random.normal(loc=COSMO_PARAMS_MEAN_PRIOR, scale=COSMO_PARAMS_SIGMA_PRIOR, size=(19369, 6))
fake_theta_test = np.random.normal(loc=COSMO_PARAMS_MEAN_PRIOR, scale=COSMO_PARAMS_SIGMA_PRIOR, size=(2000, 6))

training_theta_fake= torch.tensor(fake_theta_train, dtype=torch.float32)
test_theta_fake = torch.tensor(fake_theta_test, dtype=torch.float32)

training_theta = torch.concat([training_theta, training_theta_fake], dim=0)
test_theta = torch.concat([test_theta, test_theta_fake], dim=0)

std_train = torch.std(training_cls, dim=0)
mean_train = torch.mean(training_cls, dim=0)

std_test = torch.std(test_cls, dim=0)
mean_test = torch.mean(test_cls, dim=0)

#Change the normalization of test set here !
training_cls = (training_cls - mean_train)/std_train
training_cls = torch.concat([training_cls, training_cls], dim=0)

test_cls = (test_cls - mean_train)/std_train
test_cls = torch.concat([test_cls, test_cls], dim=0)



std_theta_train = torch.std(training_theta, dim=0)
mean_theta_train = torch.mean(training_theta, dim=0)

std_theta_test = torch.std(test_theta, dim=0)
mean_theta_test = torch.mean(test_theta, dim=0)

training_theta = (training_theta - mean_theta_train)/std_theta_train
test_theta = (test_theta - mean_theta_test)/std_theta_test

labels_train = torch.ones((2*19369, 1))
labels_train[19369:] = torch.zeros((19369,1))

labels_test = torch.ones((2*2000, 1))
labels_test[2000:] = torch.zeros((2000,1))

batch_size = 50
batch_size_test = 2000
test_dataset = data_set_classif(test_theta, test_cls, labels_test)
training_dataset = data_set_classif(training_theta, training_cls, labels_train)


#mlp = MLP(2499, 5, 1024, "cpu", 10)
mlpClassif = MLPClassif(7, 1, 256, "cpu", 3)

all_losses_train = []
all_losses_test = []
all_accuracies_train = []
optimizer = torch.optim.AdamW(mlpClassif.parameters(), lr=0.00003)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=100)

loader_test = iter(DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True))
for test_theta, test_cls, labels_test in loader_test:
    # inp = torch.cat([test_cls, test_theta[:, 0:6]], dim=1)

    predicted_proba = mlpClassif.forward(test_cls, test_theta[:, :1])
    # cat = (predicted_proba[:, 0].detach().numpy() > 1 / 2).astype(int)
    # cat_true = (labels_test.detach().numpy()).astype(int)
    # print("Accuracy total:", np.mean(cat == cat_true[:, 0]))
    test_loss = loss_function_classif(predicted_proba, labels_test)
    all_losses_test.append(test_loss.detach())
    # scheduler.step(np.mean(epoch_loss))

print("Test loss:", test_loss)
all_epoch_losses = []
for epoch in range(10000):
    mlpClassif.train()
    epoch_loss = []
    loader = iter(DataLoader(training_dataset, batch_size=batch_size, shuffle=True))
    for batch_training_theta, batch_training_cls, batch_training_labels in loader:
        # batch_training_cls = batch_training_cls + 0.2*torch.randn_like(batch_training_cls)
        #inp = torch.cat([batch_training_cls, batch_training_theta[:, 4:5]], dim=1)
        #predicted_proba = mlpClassif.forward(inp)
        predicted_proba = mlpClassif.forward(batch_training_cls, batch_training_theta[:, :1])
        cat = (predicted_proba[:, 0].detach().numpy() > 1 / 2).astype(int)
        cat_true = (batch_training_labels[:, 0].detach().numpy()).astype(int)
        all_accuracies_train.append(np.mean(cat == cat_true))
        l2_pen = 0
        for name, p in mlpClassif.named_parameters():
                l2_pen += torch.sum(p ** 2)

        loss = loss_function_classif(predicted_proba, batch_training_labels) + 0.01*l2_pen
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        all_losses_train.append(loss.detach())
        epoch_loss.append(loss.detach())

    mlpClassif.eval()
    loader_test = iter(DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True))
    for test_theta, test_cls, labels_test in loader_test:
        #inp = torch.cat([test_cls, test_theta[:, 0:6]], dim=1)

        predicted_proba = mlpClassif.forward(test_cls, test_theta[:, :1])
        #cat = (predicted_proba[:, 0].detach().numpy() > 1 / 2).astype(int)
        #cat_true = (labels_test.detach().numpy()).astype(int)
        #print("Accuracy total:", np.mean(cat == cat_true[:, 0]))
        test_loss = loss_function_classif(predicted_proba, labels_test)
        all_losses_test.append(test_loss.detach())
        scheduler.step(test_loss)
        # scheduler.step(np.mean(epoch_loss))
        all_epoch_losses.append(np.mean(epoch_loss))

    print("Epoch number:", epoch)
    print("Epoch training loss:", np.mean(epoch_loss))
    print("Epoch training acc:", np.mean(all_accuracies_train))
    print("Test loss:", test_loss)
    torch.save(mlpClassif, "data/classif/mlpclassif")
    # print("Learning rate:", get_lr(optimizer))
    print("\n")






