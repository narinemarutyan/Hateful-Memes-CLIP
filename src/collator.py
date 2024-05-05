import torch
from transformers import CLIPProcessor, CLIPTokenizer, AutoTokenizer


class CustomCollator(object):

    def __init__(self, args, fine_grained_labels, multilingual_tokenizer_path='none'):
        self.args = args
        self.fine_grained_labels = fine_grained_labels
        self.image_processor = CLIPProcessor.from_pretrained(args.clip_pretrained_model)
        self.text_processor = CLIPTokenizer.from_pretrained(args.clip_pretrained_model)
        if multilingual_tokenizer_path != 'none':
            self.text_processor = AutoTokenizer.from_pretrained(multilingual_tokenizer_path)

    def __call__(self, batch):
        pixel_values = self.image_processor(images=[item['image'] for item in batch], return_tensors="pt")[
            'pixel_values']
        if self.args.caption_mode == 'replace_text':
            text_output = self.text_processor([item['caption'] for item in batch], padding=True, return_tensors="pt",
                                              truncation=True)
        elif self.args.caption_mode == 'concat_with_text':
            text_output = self.text_processor([item['text'] + ' [SEP] ' + item['caption'] for item in batch],
                                              padding=True, return_tensors="pt", truncation=True)
        else:
            text_output = self.text_processor([item['text'] for item in batch], padding=True, return_tensors="pt",
                                              truncation=True)

        if self.args.dataset in ['original', 'masked', 'inpainted', 'tamil']:
            caption_output = self.text_processor([item['caption'] for item in batch], padding=True, return_tensors="pt",
                                                 truncation=True)
            labels = torch.LongTensor([item['label'] for item in batch])
        if self.args.dataset in ['original', 'masked', 'inpainted']:
            idx_memes = torch.LongTensor([item['idx_meme'] for item in batch])
            idx_images = torch.LongTensor([item['idx_image'] for item in batch])
            idx_texts = torch.LongTensor([item['idx_text'] for item in batch])

        batch_new = {}
        batch_new['pixel_values'] = pixel_values,
        batch_new['input_ids'] = text_output['input_ids']
        batch_new['attention_mask'] = text_output['attention_mask']
        if self.args.dataset in ['original', 'masked', 'inpainted', 'tamil']:
            batch_new['input_ids_caption'] = caption_output['input_ids']
            batch_new['attention_mask_caption'] = caption_output['attention_mask']
            batch_new['labels'] = labels
        if self.args.dataset in ['original', 'masked', 'inpainted']:
            batch_new['idx_memes'] = idx_memes
            batch_new['idx_images'] = idx_images
            batch_new['idx_texts'] = idx_texts

        if self.args.dataset in ['original', 'masked', 'inpainted', 'prop']:
            # if self.args.labels.startswith('fine_grained'):
            for label in self.fine_grained_labels:
                batch_new[label] = torch.LongTensor([item[label] for item in batch])

        if self.args.dataset == 'prop':
            batch_new['labels'] = torch.LongTensor([item['labels'] for item in batch])

        return batch_new
