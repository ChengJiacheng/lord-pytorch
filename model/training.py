import os
import itertools
from tqdm import tqdm

import numpy as np

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.modules import LordModel, VGGDistance
from model.utils import AverageMeter, NamedTensorDataset


class Lord:

	def __init__(self, config):
		super().__init__()

		self.config = config
		self.model = LordModel(config)
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	def train(self, imgs, classes, model_dir, tensorboard_dir):
		data = dict(
			img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
			img_id=torch.from_numpy(np.arange(imgs.shape[0])),
			class_id=torch.from_numpy(classes.astype(np.int64))
		)

		dataset = NamedTensorDataset(data)
		data_loader = DataLoader(
			dataset, batch_size=self.config['train']['batch_size'],
			shuffle=True, sampler=None, batch_sampler=None,
			num_workers=1, pin_memory=True, drop_last=True
		)

		self.model.init()
		self.model.to(self.device)

		criterion = VGGDistance(self.config['perceptual_loss']['layers']).to(self.device)

		optimizer = Adam([
			{
				'params': self.model.generator.parameters(),
				'lr': self.config['train']['learning_rate']['generator']
			},
			{
				'params': itertools.chain(self.model.content_embedding.parameters(), self.model.class_embedding.parameters()),
				'lr': self.config['train']['learning_rate']['latent']
			}
		], betas=(0.5, 0.999))

		scheduler = CosineAnnealingLR(
			optimizer,
			T_max=self.config['train']['n_epochs'] * len(data_loader),
			eta_min=self.config['train']['learning_rate']['min']
		)

		summary = SummaryWriter(log_dir=tensorboard_dir)

		train_loss = AverageMeter()
		for epoch in range(self.config['train']['n_epochs']):
			self.model.train()
			train_loss.reset()

			pbar = tqdm(iterable=data_loader)
			for batch in pbar:
				batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

				optimizer.zero_grad()
				out = self.model(batch['img_id'], batch['class_id'])

				content_penalty = torch.sum(out['content_code'] ** 2, dim=1).mean()
				loss = criterion(out['img'], batch['img']) + self.config['content_decay'] * content_penalty

				loss.backward()
				optimizer.step()
				scheduler.step()

				train_loss.update(loss.item())
				pbar.set_description_str('epoch #{}'.format(epoch))
				pbar.set_postfix(loss=train_loss.avg)

			pbar.close()
			torch.save(self.model.state_dict(), os.path.join(model_dir, 'model.pth'))

			fixed_sample_img = self.evaluate(dataset, randomized=False)
			random_sample_img = self.evaluate(dataset, randomized=True)

			summary.add_scalar(tag='loss', scalar_value=train_loss.avg, global_step=epoch)
			summary.add_image(tag='sample-fixed', img_tensor=fixed_sample_img, global_step=epoch)
			summary.add_image(tag='sample-random', img_tensor=random_sample_img, global_step=epoch)

		summary.close()

	def evaluate(self, dataset, n_samples=5, randomized=False):
		self.model.eval()

		if randomized:
			random = np.random
		else:
			random = np.random.RandomState(seed=1234)

		img_idx = torch.from_numpy(random.choice(len(dataset), size=n_samples, replace=False))

		samples = dataset[img_idx]
		samples = {name: tensor.to(self.device) for name, tensor in samples.items()}

		blank = torch.ones_like(samples['img'][0])
		output = [torch.cat([blank] + list(samples['img']), dim=2)]
		for i in range(n_samples):
			converted_imgs = [samples['img'][i]]

			for j in range(n_samples):
				out = self.model(samples['img_id'][[j]], samples['class_id'][[i]])
				converted_imgs.append(out['img'][0])

			output.append(torch.cat(converted_imgs, dim=2))

		return torch.cat(output, dim=1)
