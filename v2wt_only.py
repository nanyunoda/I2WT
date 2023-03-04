#!/usr/bin/env python
# coding: utf-8


from scenedetect import VideoManager, SceneManager, StatsManager
from scenedetect import open_video
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images, write_scene_list_html

##
#     초기값 설정
#     threshold_value : 이미지 추출을 위한 threshold 값
#     video_name : 변환하고자 하는 원본 동영상 이름
#     video_path : 변환하고자 하는 원본 동영상 풀 path (동영상을 rawData 디렉토리에 저장)
#     img_style : 변환하고자 하는 이미지 스타일 ( Hayao, Shinkai, Hosoda, Paprika )
#     gpu_use : GPU CUDA를 사용할 경우 0, CPU를 사용하는 경우 -1 

threshold_value = 30
video_name = 'test1'
video_path = './rawData/' + video_name + '.mp4'
img_style = 'paprika'
gpu_use = -1

video_manager = VideoManager([video_path])
### VideoManager is deprecated and will be removed.
#
#video = open_video(video_path)

stats_manager = StatsManager()
scene_manager = SceneManager(stats_manager)

scene_manager.add_detector(ContentDetector(threshold = threshold_value))

video_manager.set_downscale_factor()
### VideoManager is deprecated and will be removed.
#
#video.set_downscale_factor()

video_manager.start()
scene_manager.detect_scenes(frame_source=video_manager)
### VideoManager is deprecated and will be removed.
#
#video.start()
#scene_manager.detect_scenes(frame_source=video)

# result
stats_path = './result/result_' + video_name + '_' + str(threshold_value) + '.csv'
with open(stats_path, 'w') as f:
    stats_manager.save_to_csv(f, video_manager.get_base_timecode())
### VideoManager is deprecated and will be removed.
#
#    stats_manager.save_to_csv(f, video.get_base_timecode())

scene_list = scene_manager.get_scene_list()
print(f'{len(scene_list)} scenes detected!')

test_imgs = 'test_img/' + video_name + '_' + img_style + '_' + str(threshold_value)

save_images(
    scene_list,
    video_manager,
### VideoManager is deprecated and will be removed.
#
#    video,
    num_images = 1,
    image_name_template='$SCENE_NUMBER',
    output_dir = test_imgs)

result_html = './result/result_'+ video_name + '_' + img_style + '_' + str(threshold_value) + '.html'
write_scene_list_html(result_html, scene_list)

for scene in scene_list:
    start, end = scene

    # your code
    print(f'{start.get_seconds()} - {end.get_seconds()}')

'''
import torch
import os
import numpy as np
import argparse
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from network.Transformer import Transformer

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default = test_imgs)
parser.add_argument('--load_size', default = 450)
parser.add_argument('--model_path', default = './pretrained_model')
parser.add_argument('--style', default = img_style)
parser.add_argument('--output_dir', default = 'test_output/'+ video_name + '_' + img_style + '_' + str(threshold_value))
parser.add_argument('--gpu', type=int, default = gpu_use)

opt = parser.parse_args()

valid_ext = ['.jpg', '.png']

if not os.path.exists(opt.output_dir): os.mkdir(opt.output_dir)

# load pretrained model
model = Transformer()
model.load_state_dict(torch.load(os.path.join(opt.model_path, opt.style + '_net_G_float.pth')))
model.eval()

if opt.gpu > -1:
	print('GPU mode')
	model.cuda()
else:
	print('CPU mode')
	model.float()

for files in os.listdir(opt.input_dir):
	ext = os.path.splitext(files)[1]
	if ext not in valid_ext:
		continue
	# load image
	input_image = Image.open(os.path.join(opt.input_dir, files)).convert("RGB")
	# resize image, keep aspect ratio
	h = input_image.size[0]
	w = input_image.size[1]
	ratio = h *1.0 / w
	if ratio > 1:
		h = opt.load_size
		w = int(h*1.0/ratio)
	else:
		w = opt.load_size
		h = int(w * ratio)
        
#   input_image = input_image.resize((h, w), Image.BICUBIC)
#   Deprecation Warning 에 따라 Resampling.BICUBIC 을 BICUBIC 대신 사용함
	input_image = input_image.resize((h, w), Image.Resampling.BICUBIC)

	input_image = np.asarray(input_image)
	# RGB -> BGR
	input_image = input_image[:, :, [2, 1, 0]]
	input_image = transforms.ToTensor()(input_image).unsqueeze(0)
	# preprocess, (-1, 1)
	input_image = -1 + 2 * input_image 
	if opt.gpu > -1:
		input_image = Variable(input_image, volatile=True).cuda()
	else:
		input_image = Variable(input_image, volatile=True).float()
####       나중에 고민해보자
####    UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.

#		with torch.no_grad(): 
#			input_image = input_image.float()
#		return input_image

	# forward
	output_image = model(input_image)
	output_image = output_image[0]
	# BGR -> RGB
	output_image = output_image[[2, 1, 0], :, :]
	# deprocess, (0, 1)
	output_image = output_image.data.cpu().float() * 0.5 + 0.5
	# save
	vutils.save_image(output_image, os.path.join(opt.output_dir, files[:-4] + '_' + opt.style + '.jpg'))
'''

print('Done!')


