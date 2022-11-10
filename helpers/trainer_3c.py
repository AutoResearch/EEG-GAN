import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class Trainer:
    def __init__(self, classifier, critic, opt):
        self.device = opt['device']
        self.classifier = classifier.to(self.device)
        self.critic = critic
        if self.critic is not None:
            self.critic = self.critic.to(self.device)

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=opt['learning_rate'])

        self.epochs = opt['n_epochs'] if 'n_epochs' in opt else 1
        self.batch_size = opt['batch_size'] if 'batch_size' in opt else 1
        self.learning_rate = opt['learning_rate'] if 'learning_rate' in opt else 1e-3
        self.sample_interval = opt['sample_interval'] if 'sample_interval' in opt else 50

        self.opt = opt

        self.rank = 0

        self.use_checkpoint = opt['load_checkpoint'] if 'load_checkpoint' in opt else False
        self.path_checkpoint = os.path.join('trained_3c', 'checkpoint.pt')

    def load_checkpoint(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.classifier.load_state_dict(state_dict['classifier'])
        self.critic.load_state_dict(state_dict['critic'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def save_checkpoint(self, path, test_dataset=None, loss=None, classifier=None):
        if classifier is None:
            classifier = self.classifier

        state_dict = {
            'classifier': classifier.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'configuration': self.opt,
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

        test_data = test_data.to(self.device)
        test_labels = test_labels.to(self.device)

        # compute scores on test data
        ones_test = torch.ones_like(test_labels)
        zeros_test = torch.zeros_like(test_labels)
        scores = self.compute_scores(test_data, ones_test, zeros_test)
        test_data = test_data.view(-1, 1, 1, test_data.shape[-1])
        scores = scores.view(-1, 2, 1, 1).repeat(1, 1, 1, test_data.shape[-1]).to(self.device)
        test_data = torch.concat((test_data, scores), dim=1).to(self.device)

        loss_train = 9e9
        loss_test = 9e9
        accuracy = 0
        loss = []

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
                real_labels = train_labels[batch_size:batch_size + self.batch_size].to(self.device)

                # get scores for all types of conditions from critic and attach to data
                ones = torch.ones_like(real_labels)
                zeros = torch.zeros_like(real_labels)
                scores = self.compute_scores(data, ones, zeros)
                data = data.view(-1, 1, 1, data.shape[-1])
                scores = scores.view(-1, 2, 1, 1).repeat(1, 1, 1, data.shape[-1]).to(self.device)
                data = torch.concat((data, scores), dim=1)
                loss_train = self.batch_train(data, real_labels)

                # Test
                loss_test, accuracy = self.test(test_data, test_labels)

                loss.append((loss_train, loss_test, accuracy))

            # save checkpoint every n epochs
            if epoch % self.sample_interval == 0:
                self.save_checkpoint(self.path_checkpoint, None, loss)

            print(f"Epoch [{epoch + 1}/{self.epochs}]: "
                  f"Loss train: {loss_train:.4f}, "
                  f"Loss test: {loss_test:.4f}, "
                  f"Accuracy: {accuracy:.4f}")

        if self.rank == 0:
            self.save_checkpoint(self.path_checkpoint, None, loss, None)

        return loss

    def batch_train(self, data, labels):
        self.classifier.train()
        data, labels = data.to(self.device), labels.to(self.device)

        # calc loss
        # shape of data: (batch_size, channels, 1, sequence_length)
        # shape of labels/output: (batch_size, n_conditions)
        output = self.classifier(data)
        loss = self.loss_fn(output, labels)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def test(self, data, labels):
        self.classifier.eval()
        data, labels = data.to(self.device), labels.to(self.device)

        output = self.classifier(data)
        loss = self.loss_fn(output, labels)

        # accuracy
        output = output.round()
        accuracy = (output == labels).sum() / labels.shape[0]

        return loss.item(), accuracy.item()

    def compute_scores(self, data, real_labels, *args):
        """Compute the scores for the given data and all combinations of labels."""
        labels = [real_labels]
        labels.extend(args)
        score = torch.zeros((real_labels.shape[0], len(labels)))
        for j, label in enumerate(labels):
            batch_labels = label.view(-1, 1, 1, 1).repeat(1, 1, 1, data.shape[1])
            batch_data = data.view(-1, 1, 1, data.shape[1])
            batch_data = torch.cat((batch_data, batch_labels), dim=1).to(self.device)
            validity = self.critic(batch_data)
            score[:, j] = validity[:, 0]

        return score

    def print_log(self, current_epoch, train_loss, test_loss, test_accuracy):
        print(
            "[Epoch %d/%d] [Train loss: %f] [Test loss: %f] [Accuracy: %f]"
            % (current_epoch, self.epochs,
               train_loss, test_loss, test_accuracy)
        )


class DDPTrainer(Trainer):

    def __init__(self, classifier, critic, opt):
        super(Trainer, self).__init__()

        # training configuration
        super().__init__(classifier, critic, opt)

        self.world_size = opt['world_size'] if 'world_size' in opt else 1

    def set_ddp_framework(self):
        # set ddp generator and discriminator
        self.classifier.to(self.rank)
        self.classifier = DDP(self.classifier, device_ids=[self.rank])

        # set ddp optimizer
        opt_state = self.optimizer.state_dict()
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(opt_state)

    def set_device(self, rank):
        self.rank = rank
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else f'cpu:{rank}')

    def save_checkpoint(self, path_checkpoint=None, test_dataset=None, loss=None, classifier=None):
        if self.rank == 0:
            super().save_checkpoint(path_checkpoint, test_dataset=test_dataset, loss=loss, classifier=self.classifier.module)
        # dist.barrier()

    def print_log(self, current_epoch, train_loss, test_loss, test_accuracy):
        # average the loss across all processes before printing
        reduce_tensor = torch.tensor([train_loss, test_loss, test_accuracy], dtype=torch.float32, device=self.device)
        dist.all_reduce(reduce_tensor, op=dist.ReduceOp.SUM)
        reduce_tensor /= self.world_size
        if self.rank == 0:
            super().print_log(current_epoch, reduce_tensor[0], reduce_tensor[1], reduce_tensor[2])
