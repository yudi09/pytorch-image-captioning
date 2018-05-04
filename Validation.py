import os
import torch
import time
import pickle
import argparse
from PIL import Image
import torch.nn as nn
from utils import get_cnn
from Decoder import RNN
from Vocabulary import Vocabulary
from torch.autograd import Variable
from torchvision import transforms
from DataLoader import DataLoader, shuffle_data

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-dir', type = str, default = 'dev')
	parser.add_argument('-model')
	parser.add_argument('-epoch', type=int)
	parser.add_argument('-gpu_device', type=int)
	args = parser.parse_args()

	with open(os.path.join(args.model, 'vocab.pkl'), 'rb') as f:
	    vocab = pickle.load(f)

	transform = transforms.Compose([transforms.Resize((224, 224)), 
	                                transforms.ToTensor(),
	                                transforms.Normalize((0.5, 0.5, 0.5),
	                                                     (0.5, 0.5, 0.5))
	                                ])
	dataloader = DataLoader(args.dir, vocab, transform)
	data = dataloader.gen_data()
	print(args.dir + ' loaded')

	embedding_dim = 512
	vocab_size = vocab.index
	hidden_dim = 512
	model_name = args.model
	criterion = nn.CrossEntropyLoss()
	cnn = get_cnn(architecture = model_name, embedding_dim = embedding_dim)
	lstm = RNN(embedding_dim = embedding_dim, hidden_dim = hidden_dim, 
	           vocab_size = vocab_size)

	if torch.cuda.is_available():
		with torch.cuda.device(args.gpu_device):
			cnn.cuda()
			lstm.cuda()
	
	for iteration in range(0, 240, 10):
		cnn_file = 'iter_' + str(iteration) + '_cnn.pkl'
		lstm_file = 'iter_' + str(iteration) + '_lstm.pkl'
		cnn.load_state_dict(torch.load(os.path.join(model_name, cnn_file)))
		lstm.load_state_dict(torch.load(os.path.join(model_name, lstm_file)))

		cnn.eval()
		lstm.eval()
		
		images, captions = data
		num_captions = len(captions)
		loss_list = []
		# tic = time.time()
		with torch.no_grad():
			for i in range(num_captions):
				image_id = images[i]
				image = dataloader.get_image(image_id)
				image = image.unsqueeze(0)
							
				if torch.cuda.is_available():
					with torch.cuda.device(args.gpu_device):
						image = Variable(image).cuda()
						caption = torch.cuda.LongTensor(captions[i])
				else:
					image = Variable(image)
					caption = torch.LongTensor(captions[i])

				caption_train = caption[:-1] # remove <end>
				
				loss = criterion(lstm(cnn(image), caption_train), caption)
				
				loss_list.append(loss)
				# avg_loss = torch.mean(torch.Tensor(loss_list))
				# print('ex %d / %d avg_loss %f' %(i+1, num_captions, avg_loss), end='\r')
		# toc = time.time()
		avg_loss = torch.mean(torch.Tensor(loss_list))
		print('%d %f' %(iteration, avg_loss))