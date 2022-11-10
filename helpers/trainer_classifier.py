import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class Trainer:
    def __init__(self, model, opt):
        self.model = model
        self.loss_fn = torch.nn.MSELoss()
        self.device = opt['device']

        self.optimizer = torch.optim.Adam(model.parameters(), lr=opt['learning_rate'])

        self.epochs = opt['n_epochs'] if 'n_epochs' in opt else 1
        self.batch_size = opt['batch_size'] if 'batch_size' in opt else 1
        self.learning_rate = opt['learning_rate'] if 'learning_rate' in opt else 1e-3
        self.sample_interval = opt['sample_interval'] if 'sample_interval' in opt else 50

        self.opt = opt

        self.rank = 0

        self.use_checkpoint = opt['load_checkpoint'] if 'load_checkpoint' in opt else False
        self.path_checkpoint = os.path.join('trained_classifier', 'checkpoint.pt')
        # self.path_checkpoint = opt['path_checkpoint'] if 'path_checkpoint' in opt else None

    def load_checkpoint(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def save_checkpoint(self, path, test_dataset=None, loss=None, model=None):
        if model is None:
            model = self.model

        state_dict = {
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'configuration': self.opt,
            'test_dataset': test_dataset,
            'train_loss': [l[0] for l in loss] if loss is not None else None,
            'test_loss': [l[1] for l in loss] if loss is not None else None,
            'accuracy': [l[2] for l in loss] if loss is not None else None
        }
        torch.save(state_dict, path)

    def train(self, train_data, train_labels, test_data, test_labels):
        if train_data.shape[0] != train_labels.shape[0]:
            raise RuntimeError("Train data and labels must have the same number of samples.")
        if test_data.shape[0] != test_labels.shape[0]:
            raise RuntimeError("Test data and labels must have the same number of samples.")

        loss_train = 9e9
        loss_test = 9e9
        accuracy = 0
        loss = []

        test_data = test_data.to(self.device)
        test_labels = test_labels.to(self.device)

        for epoch in range(self.epochs):
            # Train
            # shuffle train_data and train_labels
            idx = torch.randperm(train_data.shape[0])
            train_data = train_data[idx]
            train_labels = train_labels[idx]
            for batch in range(0, train_data.shape[0], self.batch_size):
                # Check if remaining samples are enough for a batch and adjust if not
                if batch + self.batch_size > train_data.shape[0]:
                    batch_size = train_data.shape[0] - batch
                else:
                    batch_size = self.batch_size

                data = train_data[batch_size:batch_size + self.batch_size].to(self.device)
                labels = train_labels[batch_size:batch_size + self.batch_size].to(self.device)
                loss_train = self.batch_train(data, labels)

                # Test
                loss_test, accuracy = self.test(test_data, test_labels)

                loss.append((loss_train, loss_test, accuracy))

                # save checkpoint every n epochs
                if epoch % self.sample_interval == 0:
                    self.save_checkpoint(self.path_checkpoint, torch.concat((test_labels, test_data), dim=1), loss)

            print(f"Epoch [{epoch + 1}/{self.epochs}]: "
                  f"Loss train: {loss_train:.4f}, "
                  f"Loss test: {loss_test:.4f}, "
                  f"Accuracy: {accuracy:.4f}")

        if self.rank == 0:
            self.save_checkpoint(self.path_checkpoint, None, loss, None)

        return loss

    def batch_train(self, data, labels):
        self.model.train()
        data, labels = data.to(self.device), labels.to(self.device)

        # calc loss
        # shape of data: (batch_size, channels, 1, sequence_length)
        # shape of labels/output: (batch_size, n_conditions)
        output = self.model(data.view(data.shape[0], 1, 1, data.shape[-1]))
        loss = self.loss_fn(output, labels)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def test(self, data, labels):
        self.model.eval()
        data, labels = data.to(self.device), labels.to(self.device)

        output = self.model(data.view(data.shape[0], 1, 1, data.shape[-1]))
        loss = self.loss_fn(output, labels)

        # accuracy
        output = output.round()
        accuracy = (output == labels).sum() / labels.shape[0]

        return loss.item(), accuracy.item()

    def print_log(self, current_epoch, train_loss, test_loss, test_accuracy):
        print(
            "[Epoch %d/%d] [Train loss: %f] [Test loss: %f] [Accuracy: %f]"
            % (current_epoch, self.epochs,
               train_loss, test_loss, test_accuracy)
        )


class DDPTrainer(Trainer):

    def __init__(self, model, opt):
        super(Trainer, self).__init__()

        # training configuration
        super().__init__(model, opt)

        self.world_size = opt['world_size'] if 'world_size' in opt else 1

    def set_ddp_framework(self):
        # set ddp generator and discriminator
        self.model.to(self.rank)
        self.model = DDP(self.model, device_ids=[self.rank])

        # set ddp optimizer
        opt_state = self.optimizer.state_dict()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(opt_state)

    def set_device(self, rank):
        self.rank = rank
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else f'cpu:{rank}')

    def save_checkpoint(self, path_checkpoint=None, test_dataset=None, loss=None, model=None):
        if self.rank == 0:
            super().save_checkpoint(path_checkpoint, test_dataset=test_dataset, loss=loss, model=self.model.module)
        # dist.barrier()

    def print_log(self, current_epoch, train_loss, test_loss, test_accuracy):
        # average the loss across all processes before printing
        reduce_tensor = torch.tensor([train_loss, test_loss, test_accuracy], dtype=torch.float32, device=self.device)
        dist.all_reduce(reduce_tensor, op=dist.ReduceOp.SUM)
        reduce_tensor /= self.world_size
        if self.rank == 0:
            super().print_log(current_epoch, reduce_tensor[0], reduce_tensor[1], reduce_tensor[2])
