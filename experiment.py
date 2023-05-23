import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from bridge import BrownianBridge
import math
import matplotlib.pyplot as plt



class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Get the positional encoding of times t
        :param time: torch.tensor (N_batch,) of float, corresponding to the sampling times
        :return: torch.tensor (N_batch, dim), the positionql encodings.
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class dataSetLikelihoodFree(Dataset):

    def __init__(self, data, param, perturbed_param, times):
        self.data = data
        self.param = param
        self.perturbed_param = perturbed_param
        self.times = times

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.param[idx], self.perturbed_param[idx], self.times[idx]



class Network(nn.Module):
    def __init__(self, dim_in_compression, dim_out_compression, dims_in, dims_out):
        super().__init__()
        n_layers = len(dims_out)
        assert dims_out[:-1] == dims_in[1:], "Dimension of subsequent layer not matching"
        self.compression_layer = torch.nn.Linear(dim_in_compression, dim_out_compression)
        self.dim_in_out = zip(dims_in, dims_out)
        self.layers = nn.ModuleList([])
        for i, dims in enumerate(self.dim_in_out):
            dim_in, dim_out = dims
            self.layers.append(torch.nn.Linear(dim_in, dim_out))
            if i != n_layers - 1:
                self.layers.append(torch.nn.LeakyReLU())
            else:
                print("NOT LAST LAYER ?")

    def forward(self, cls, theta):
        compressed_cls = self.compression_layer(cls)
        x = torch.concatenate([compressed_cls, theta], dim=-1)
        for h in self.layers:
            x = h(x)

        return x


def l2_loss(true_data, pred_data):
    return torch.mean((true_data - pred_data) ** 2)


def generate_dataset(brid, dataset):
    N_sample, dimension = dataset.shape
    times = torch.rand(size=(N_sample, 1))
    perturbed = brid.sample(times, torch.randn(size=(N_sample, dimension)), dataset)
    return times, perturbed

brid = BrownianBridge(6, a=1, b=4)
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


std_train = torch.std(training_cls, dim=0)
mean_train = torch.mean(training_cls, dim=0)

std_test = torch.std(test_cls, dim=0)
mean_test = torch.mean(test_cls, dim=0)

#Change the normalization of test set here !
training_cls = (training_cls - mean_train)/std_train
test_cls = (test_cls - mean_train)/std_train


std_theta_train = torch.std(training_theta, dim=0)
mean_theta_train = torch.mean(training_theta, dim=0)

std_theta_test = torch.std(test_theta, dim=0)
mean_theta_test = torch.mean(test_theta, dim=0)

training_theta = (training_theta - mean_theta_train)/std_theta_train
test_theta = (test_theta - mean_theta_test)/std_theta_test
pos_embbed = SinusoidalPositionEmbeddings(training_cls.shape[1])
if False:
    times_train, perturbed_theta_train = generate_dataset(brid, training_theta)
    times_test, perturbed_theta_test = generate_dataset(brid, test_theta)

    train_pos_embedding = pos_embbed(times_train)[:, 0, :]
    test_pos_embedding = pos_embbed(times_test)[:, 0, :]
    training_set = dataSetLikelihoodFree(training_cls, training_theta, perturbed_theta_train, train_pos_embedding)
    test_set = dataSetLikelihoodFree(test_cls, test_theta, perturbed_theta_test, test_pos_embedding)
    data_test = test_cls + test_pos_embedding

    batch_size = 50
    net = Network(9996, 6, [12, 1024, 1024, 1024],[1024, 1024, 1024, 6])
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    pred_test = net.forward(data_test, perturbed_theta_test)
    loss_test = l2_loss(test_theta, pred_test)
    print("loss test:", loss_test)
    for n_epoch in range(5000):
        data_loader = DataLoader(training_set, shuffle=True, batch_size=batch_size)
        data_loader_test = DataLoader(test_set)
        for cls, theta, perturbed_theta, time_embed in iter(data_loader):
            optimizer.zero_grad()
            data = cls + time_embed
            pred_theta = net.forward(data, perturbed_theta)
            loss = l2_loss(theta, pred_theta)
            loss.backward()
            optimizer.step()


        pred_test = net.forward(data_test, perturbed_theta_test)
        loss_test = l2_loss(test_theta, pred_test)
        print("N epoch:", n_epoch)
        print("loss test:", loss_test)
        torch.save(net, "network")
        print("\n")

network = torch.load("network")
times = torch.linspace(0, 1, 1000)[1:, None]
#times_emb = pos_embbed(times)[:, 0, :]

obs = torch.ones((10000,test_cls.shape[1]))*test_cls[100:101, :]
print("obs shape:", obs.shape)
#print("times_embedd", times_emb.shape)
n_param = 2
#Also for the 10th example !
_, sample = brid.euler_maruyama(torch.randn(size=(10000, 6)), times, 1, network, obs)

print(np.corrcoef(sample.T))
plt.hist(sample[:, n_param].detach().numpy(), density=True, alpha=0.5)
plt.hist(test_theta[:, n_param], alpha=0.5, density=True)
plt.axvline(x=test_theta[100, n_param])
plt.show()


















