import os
import torch
import time
import pickle
import argparse
import torch.nn as nn
from Decoder import RNN
from utils import get_cnn
import matplotlib.pyplot as plt
from Vocabulary import Vocabulary
from torchvision import transforms
from torch.autograd import Variable
from Preprocess import load_captions
from DataLoader import DataLoader, shuffle_data


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-model')
	parser.add_argument('-dir', type = str, default = 'train')
	parser.add_argument('-save_iter', type = int, default = 10)
	parser.add_argument('-learning_rate', type=float, default = 1e-5)
	parser.add_argument('-epoch', type=int)
	parser.add_argument('-gpu_device', type=int)
	parser.add_argument('-hidden_dim', type=int, default = 512)
	parser.add_argument('-embedding_dim', type=int, default = 512)

	args = parser.parse_args()
	print(args)
	train_dir = args.dir
	threshold = 5

	captions_dict = load_captions(train_dir)
	vocab = Vocabulary(captions_dict, threshold)
	with open(os.path.join(args.model, 'vocab.pkl'), 'wb') as f:
		pickle.dump(vocab, f)
		print('dictionary dump')
	transform = transforms.Compose([transforms.Resize((224, 224)), 
									transforms.ToTensor(),
									transforms.Normalize((0.5, 0.5, 0.5),
														 (0.5, 0.5, 0.5))
									])

	dataloader = DataLoader(train_dir, vocab, transform)
	data = dataloader.gen_data()
	print(train_dir + ' loaded')

	# embedding_dim = 512
	vocab_size = vocab.index
	hidden_dim = 512
	# learning_rate = 1e-3
	model_name = args.model
	cnn = get_cnn(architecture = model_name, embedding_dim = args.embedding_dim)
	lstm = RNN(embedding_dim = args.embedding_dim, hidden_dim = args.hidden_dim, 
			   vocab_size = vocab_size)
	
	if torch.cuda.is_available():
		with torch.cuda.device(args.gpu_device):
			cnn.cuda()
			lstm.cuda()
			# iteration = args.epoch
			# cnn_file = 'iter_' + str(iteration) + '_cnn.pkl'
			# lstm_file = 'iter_' + str(iteration) + '_lstm.pkl'
			# cnn.load_state_dict(torch.load(os.path.join(model_name, cnn_file)))
			# lstm.load_state_dict(torch.load(os.path.join(model_name, lstm_file)))
	
	criterion = nn.CrossEntropyLoss()
	params = list(cnn.linear.parameters()) + list(lstm.parameters()) 
	optimizer = torch.optim.Adam(params, lr = args.learning_rate)
	num_epochs = 100000
	
	for epoch in range(num_epochs):
		shuffled_images, shuffled_captions = shuffle_data(data, seed = epoch)
		num_captions = len(shuffled_captions)
		loss_list = []
		tic = time.time()
		for i in range(num_captions):
			image_id = shuffled_images[i]
			image = dataloader.get_image(image_id)
			image = image.unsqueeze(0)
						
			if torch.cuda.is_available():
				with torch.cuda.device(args.gpu_device):
					image = Variable(image).cuda()
					caption = torch.cuda.LongTensor(shuffled_captions[i])
			else:
				image = Variable(image)
				caption = torch.LongTensor(shuffled_captions[i])

			caption_train = caption[:-1] # remove <end>
			cnn.zero_grad()
			lstm.zero_grad()
			
			cnn_out = cnn(image)
			lstm_out = lstm(cnn_out, caption_train)
			loss = criterion(lstm_out, caption)
			loss.backward()
			optimizer.step()
			loss_list.append(loss)
		toc = time.time()
		avg_loss = torch.mean(torch.Tensor(loss_list))	
		print('epoch %d avg_loss %f time %.2f mins' 
			%(epoch, avg_loss, (toc-tic)/60))		
		if epoch % args.save_iter == 0:

			torch.save(cnn.state_dict(), os.path.join(model_name, 'iter_%d_cnn.pkl'%(epoch)))
			torch.save(lstm.state_dict(), os.path.join(model_name, 'iter_%d_lstm.pkl'%(epoch)))

