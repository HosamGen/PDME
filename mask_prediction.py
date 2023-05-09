import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from bert.modeling_bert import BertModel
from bert.tokenization_bert import BertTokenizer
from LAVT_RIS.lib import segmentation
from PIL import Image

# image_path = './LAVT-RIS/testimages/mug_in_hand.jpg'
# prompt = 'the blue mug on the hand'
# weights = './LAVT-RIS//checkpoints/refcoco.pth'
# device = 'cuda:0'

class LAVTRISInference():
    def __init__(self, transforms, weights_path, device):
        self.transforms = transforms
        self.weights = weights_path
        self.device = device


    def infer(self, image_path, prompt) -> np.ndarray:
        self.img = Image.open(image_path)
        self.img_ndarray = np.array(self.img)
        self.original_w, self.original_h = self.img.size
        self.img = self.transforms(self.img).unsqueeze(0)
        self.img = self.img.to(self.device)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        prompt_tokenized = tokenizer.encode(text=prompt, add_special_tokens=True)[:20] # Truncate to 20 tokens
        # Pad tokenized prompt
        padded_sent_toks = [0] * 20
        padded_sent_toks[:len(prompt_tokenized)] = prompt_tokenized
        # Prompt token mask: 1 for real words; 0 for padded tokens
        attention_mask = [0] * 20
        attention_mask[:len(prompt_tokenized)] = [1]*len(prompt_tokenized)
        # Convert tokens and mask to tensors
        padded_sent_toks = torch.tensor(padded_sent_toks).unsqueeze(0)  # (1, 20)
        attention_mask = torch.tensor(attention_mask).unsqueeze(0)  # (1, 20)
        padded_sent_toks = padded_sent_toks.to(self.device)  # for inference (input)
        attention_mask = attention_mask.to(self.device)  # for inference (input)

        # Construct mini arguments class for LAVT-RIS model
        
        class args:
            swin_type = 'base'
            window12 = True
            mha = ''
            fusion_drop = 0.00
            pretrained_swin_weights = "./LAVT_RIS/checkpoints/swin_base_patch4_window12_384_22k.pth"
            image_size = 512

        # Load Model

        single_model = segmentation.__dict__['lavt'](pretrained='', args=args)
        single_model.to(self.device)
        model_class = BertModel
        single_bert_model = model_class.from_pretrained('bert-base-uncased')
        single_bert_model.pooler = None

        checkpoint = torch.load(self.weights, map_location='cpu')
        single_bert_model.load_state_dict(checkpoint['bert_model'])
        single_model.load_state_dict(checkpoint['model'])
        model = single_model.to(self.device)
        bert_model = single_bert_model.to(self.device)

        # Inference

        bert_model.eval()
        model.eval()
        with torch.no_grad():
            last_hidden_states = bert_model(padded_sent_toks, attention_mask=attention_mask)[0]
            embedding = last_hidden_states.permute(0, 2, 1)
            output, features = model(self.img, embedding, l_mask=attention_mask.unsqueeze(-1))
            output = output.argmax(1, keepdim=True)  # (1, 1, 480, 480)
            print(output.shape)
            output = F.interpolate(output.float(), (self.original_h, self.original_w))  # 'nearest'; resize to the original image size
            output = output.squeeze()  # (orig_h, orig_w)
            output = output.cpu().data.numpy()  # (orig_h, orig_w)

        int_output = output.astype(np.uint8)
        pil_image = Image.fromarray(int_output.astype(np.uint8) * 255)
        output_folder = "./LAVT_RIS/maskoutput/"
        basename = os.path.basename(image_path)
        letters = basename.split(".")[-2]
        filename = letters + "_" + prompt.replace(" ", "_") + "_mask"
        pil_image.save(output_folder + filename + ".jpg")

        # Overlay the mask on the image
        visualization = self.overlay_davis(self.img_ndarray, output)
        visualization = Image.fromarray(visualization)
        output_folder = "./LAVT_RIS/overlayoutput/"
        filename = letters + "_" + prompt.replace(" ", "_") + "_overlay"
        visualization.save(output_folder + filename + ".jpg")

        return output, features


    # Save Results
    def overlay_davis(self, image, mask, colors=[[0, 0, 0], [255, 0, 0]], cscale=1, alpha=0.4):
        from scipy.ndimage.morphology import binary_dilation

        colors = np.reshape(colors, (-1, 3))
        colors = np.atleast_2d(colors) * cscale

        im_overlay = image.copy()
        object_ids = np.unique(mask).astype(int)

        for object_id in object_ids[1:]:
            # Overlay color on  binary mask
            foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
            binary_mask = mask == object_id

            # Compose image
            im_overlay[binary_mask] = foreground[binary_mask]

            # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
            countours = binary_dilation(binary_mask) ^ binary_mask
            # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
            im_overlay[countours, :] = 0

        return im_overlay.astype(image.dtype)
