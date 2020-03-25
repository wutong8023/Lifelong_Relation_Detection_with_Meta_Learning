import argparse
import numpy as np
import torch
from torch import nn, autograd as ag
import matplotlib.pyplot as plt
from copy import deepcopy


def experiment(run, plot=True):
    seed = 0
    inner_step_size = 0.02  # stepsize in inner SGD
    inner_epochs = 1  # number of epochs of each inner SGD
    outer_stepsize_reptile = 0.1  # stepsize of outer optimization, i.e., meta-optimization
    outer_stepsize_maml = 0.01
    n_iterations = 30000  # number of outer updates; each iteration we sample one task and update on it

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # Define task distribution
    x_all = np.linspace(-5, 5, 50)[:, None]  # All of the x points
    n_train = 10  # Size of training minibatches

    def gen_task():
        "Generate classification problem"
        phase = rng.uniform(low=0, high=2 * np.pi)
        ampl = rng.uniform(0.1, 5)
        f_randomsine = lambda x: np.sin(x + phase) * ampl
        return f_randomsine

    # Define model. Reptile paper uses ReLU, but Tanh gives slightly better results
    model = nn.Sequential(
        nn.Linear(1, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )

    def to_torch(x):
        return ag.Variable(torch.Tensor(x))

    def train_on_batch(x, y):
        x = to_torch(x)
        y = to_torch(y)
        model.zero_grad()
        ypred = model(x)
        loss = (ypred - y).pow(2).mean()
        loss.backward()
        for param in model.parameters():
            param.data -= inner_step_size * param.grad.data

    def predict(x):
        x = to_torch(x)
        return model(x).data.numpy()

    # Choose a fixed task and minibatch for visualization
    f_plot = gen_task()
    xtrain_plot = x_all[rng.choice(len(x_all), size=n_train)]

    # Training loop
    for iteration in range(n_iterations):
        weights_before = deepcopy(model.state_dict())  # 取出训练前的参数留存

        # Generate task
        f = gen_task()
        y_all = f(x_all) # x和y

        # Do SGD on this task
        inds = rng.permutation(len(x_all))  # 随机排列一个序列，或返回一个排列范围。
        train_ind = inds[:-1 * n_train]  # 训练数据
        val_ind = inds[-1 * n_train:]       # Val contains 1/5th of the sine wave

        for _ in range(inner_epochs):  # 内层循环
            for start in range(0, len(train_ind), n_train):  # N_TRAIN 是step
                mbinds = train_ind[start:start + n_train]  # 取出一组index
                x = to_torch(x_all[mbinds])
                y = to_torch(y_all[mbinds])
                model.zero_grad()
                ypred = model(x)  # 预测y
                loss = (ypred - y).pow(2).mean()  # 求loss
                loss.backward()  # 反向传播，自动求导出梯度
                for param in model.parameters():
                    param.data -= inner_step_size * param.grad.data  # 根据梯度信息，手动step更新梯度

        if run == 'MAML':
            outer_step_size = outer_stepsize_maml * (1 - iteration / n_iterations)  # linear schedule
            for start in range(0, len(val_ind), n_train):
                dpinds = val_ind[start:start + n_train]
                x = to_torch(x_all[dpinds])
                y = to_torch(y_all[dpinds])

                # Compute the grads
                model.zero_grad()
                y_pred = model(x)
                loss = (y_pred - y).pow(2).mean()
                loss.backward()

                # Reload the model
                model.load_state_dict(weights_before)

                # SGD on the params
                for param in model.parameters():
                    param.data -= outer_step_size * param.grad.data
        else:
            # Interpolate between current weights and trained weights from this task
            # I.e. (weights_before - weights_after) is the meta-gradient
            weights_after = model.state_dict()  # 经过inner_epoch轮次的梯度更新后weights
            outerstepsize = outer_stepsize_reptile * (1 - iteration / n_iterations)  # linear schedule
            model.load_state_dict({name: weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize
                                   for name in weights_before})

        # Periodically plot the results on a particular task and minibatch
        if plot and iteration == 0 or (iteration + 1) % 1000 == 0:
            plt.cla()
            f = f_plot
            weights_before = deepcopy(model.state_dict())  # save snapshot before evaluation
            plt.plot(x_all, predict(x_all), label="pred after 0", color=(0, 0, 1))
            for inneriter in range(32):
                train_on_batch(xtrain_plot, f(xtrain_plot))
                if (inneriter + 1) % 8 == 0:
                    frac = (inneriter + 1) / 32
                    plt.plot(x_all, predict(x_all), label="pred after %i" % (inneriter + 1), color=(frac, 0, 1 - frac))
            plt.plot(x_all, f(x_all), label="true", color=(0, 1, 0))
            lossval = np.square(predict(x_all) - f(x_all)).mean()
            plt.plot(xtrain_plot, f(xtrain_plot), "x", label="train", color="k")
            plt.ylim(-4, 4)
            plt.legend(loc="lower right")
            plt.pause(0.01)
            model.load_state_dict(weights_before)  # restore from snapshot
            print(f"-----------------------------")
            print(f"iteration               {iteration + 1}")
            print(f"loss on plotted curve   {lossval:.3f}")  # would be better to average loss over a set of examples, but this is optimized for brevity


def main():
    parser = argparse.ArgumentParser(description='MAML and Reptile Sine wave regression example.')
    parser.add_argument('--run', dest='run', default='Reptile') # MAML, Reptile
    args = parser.parse_args()

    experiment(args.run)


if __name__ == '__main__':
    main()