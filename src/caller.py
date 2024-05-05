import torch
from transformers import CLIPProcessor, CLIPTokenizer


class Caller(object):

    def __init__(self, args):
        self.args = args
        self.image_processor = CLIPProcessor.from_pretrained(args.clip_pretrained_model)
        self.text_processor = CLIPTokenizer.from_pretrained(args.clip_pretrained_model)

    def __call__(self, batch):
        pixel_values = self.image_processor(images=[item['image'] for item in batch], return_tensors="pt")[
            'pixel_values']

        text_output = self.text_processor([item['text'] + ' [SEP] ' + item['caption'] for item in batch], padding=True,
                                          return_tensors="pt", truncation=True)

        caption_output = self.text_processor([item['caption'] for item in batch], padding=True, return_tensors="pt",
                                             truncation=True)
        labels = torch.LongTensor([item['label'] for item in batch])

        new_batch = {}
        new_batch['pixel_values'] = pixel_values,
        new_batch['input_ids'] = text_output['input_ids']
        new_batch['attention_mask'] = text_output['attention_mask']

        new_batch['input_ids_caption'] = caption_output['input_ids']
        new_batch['attention_mask_caption'] = caption_output['attention_mask']
        new_batch['labels'] = labels

        return new_batch
