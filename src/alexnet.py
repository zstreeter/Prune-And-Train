import jax.numpy as jnp
import flax.linen as nn


class AlexNet(nn.Module):
    num_classes: int = 200

    def setup(self):
        self.conv1 = nn.Conv(features=96, kernel_size=(11, 11), strides=(4, 4))
        self.conv2 = nn.Conv(features=256, kernel_size=(5, 5))
        self.conv3 = nn.Conv(features=384, kernel_size=(3, 3))
        self.conv4 = nn.Conv(features=384, kernel_size=(3, 3))
        self.conv5 = nn.Conv(features=256, kernel_size=(3, 3))
        self.fc1 = nn.Dense(features=4096)
        self.fc2 = nn.Dense(features=4096)
        self.fc3 = nn.Dense(features=self.num_classes)

    def __call__(self, x):
        x = nn.relu(self.conv1(x))
        x = nn.max_pool(x, (3, 3), (2, 2))
        x = nn.relu(self.conv2(x))
        x = nn.max_pool(x, (3, 3), (2, 2))
        x = nn.relu(self.conv3(x))
        x = nn.relu(self.conv4(x))
        x = nn.relu(self.conv5(x))
        x = nn.max_pool(x, (3, 3), (2, 2))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        x = self.fc3(x)
        return x
