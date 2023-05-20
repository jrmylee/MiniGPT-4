import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
import copy


# ========================================
#             Model Initialization
# ========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--dir", required=False, help="directory to images if necessary")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

images = []
questions = ["Are there dark clouds in the sky?", "Is it possible that this image was taken by a camera?", "Is the image blurry?", "If there are buildings in the image, do they look realistic?" ]
if args.dir:
    # walk through the directory and put all of the files into the images array
    for root, dirs, files in os.walk(args.dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".JPG") or file.endswith(".PNG") or file.endswith(".JPEG"):
                images.append(os.path.join(root, file))



model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')


responses = {}
num_beams = 2
temperature = 1
for image in images:
    print("NEXT IMAGE BABY \n\n\n\n")
    img_list = []
    chat_state = copy.deepcopy(CONV_VISION.copy())
    chat.upload_img(image, chat_state, img_list)
    post_image_chat_state = copy.deepcopy(chat_state)
    responses[image] = {}
    specific_dict = responses[image]
    for question in questions:
        print("NEXT QN UNHHHH \n\n\n\n")
        chat.ask(question, chat_state)
        llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
        specific_dict[question] = llm_message
        chat_state = post_image_chat_state

# write the dictionary into a csv
# the csv will have the columns image, question, response
import csv
rows = []
rows.append(["image", "question", "response"])
for image in responses.keys():
    for question in responses[image].keys():
        rows.append([image, question, responses[image][question]])
# write the rows into a csv
with open('responses.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(rows)
