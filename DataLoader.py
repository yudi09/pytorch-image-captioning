import os
import json
import nltk
import time
import torch
from PIL import Image

class DataLoader():
	def __init__(self, dir_path, vocab, transform):
		self.images = None
		self.captions_dict = None
		# self.data = None
		self.vocab = vocab
		self.transform = transform
		self.load_captions(dir_path)
		self.load_images(dir_path)
	
	def load_captions(self, captions_dir):
		caption_file = os.path.join(captions_dir, 'captions.txt')
		captions_dict = {}
		with open(caption_file) as f:
			for line in f:
				cur_dict = json.loads(line)
				for k, v in cur_dict.items():
					captions_dict[k] = v
		self.captions_dict = captions_dict
	
	def load_images(self, images_dir):
		files = os.listdir(images_dir)
		images = {}
		for cur_file in files:
			ext = cur_file.split('.')[1]
			if ext == 'jpg':
				images[cur_file] = self.transform(Image.open(os.path.join(images_dir, cur_file)))
		self.images = images
	
	def caption2ids(self, caption):
		vocab = self.vocab
		tokens = nltk.tokenize.word_tokenize(caption.lower())
		vec = []
		vec.append(vocab.get_id('<start>'))
		vec.extend([vocab.get_id(word) for word in tokens])
		vec.append(vocab.get_id('<end>'))
		return vec
	
	def gen_data(self):
		images = []
		captions = []
		for image_id, cur_captions in self.captions_dict.items():
			num_captions = len(cur_captions)
			images.extend([image_id] * num_captions)
			for caption in cur_captions:
				captions.append(self.caption2ids(caption))
		# self.data = images, captions
		data = images, captions
		return data

	def get_image(self, image_id):
		return self.images[image_id]
			
def shuffle_data(data, seed=0):
	images, captions = data
	shuffled_images = []
	shuffled_captions = []
	num_images = len(images)
	torch.manual_seed(seed)
	perm = list(torch.randperm(num_images))
	for i in range(num_images):
		shuffled_images.append(images[perm[i]])
		shuffled_captions.append(captions[perm[i]])
	return shuffled_images, shuffled_captions

# def make_minibatches(self, data, minibatch_size=1, seed=0):
	
# def get_batch(self,):