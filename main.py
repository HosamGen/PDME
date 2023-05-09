from optimization.image_editor import ImageEditor
from optimization.arguments import get_arguments

from PIL import Image
from mask_prediction import LAVTRISInference
import torchvision.transforms as T


if __name__ == "__main__":

    image_transforms = T.Compose(
        [
        T.Resize(512),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    args = get_arguments()

    mask_prompt = args.mask_prompt
    image_path = args.init_image
    lavt_weights = args.lavt_weights
    device = "cuda:" + str(args.gpu_id)

    LAVTRISInference = LAVTRISInference(image_transforms, lavt_weights, device)
    # Get output mask (pre-binarization) and features (x_c1, x_c2, x_c3, x_c4)
    LAVTRIS_mask, decoding_features = LAVTRISInference.infer(image_path, mask_prompt)
    
    print("Mask Type: ", type(LAVTRIS_mask), "Mask Shape: ", LAVTRIS_mask.shape)
    
    image_editor = ImageEditor(args, LAVTRIS_mask, decoding_features) # Add extra imageditor arguments for y4 y3 y2 y1 into UNET.
    image_editor.edit_image_by_prompt()
    image_editor.reconstruct_image()