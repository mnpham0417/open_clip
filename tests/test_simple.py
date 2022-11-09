
import torch
from PIL import Image
from open_clip import tokenizer
import pytest
import open_clip
import os
import timm
os.environ["CUDA_VISIBLE_DEVICES"] = ""

@pytest.mark.parametrize("jit", [False, True])
def test_inference(jit):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32', jit=jit)
    
    assert False
    current_dir = os.path.dirname(os.path.realpath(__file__))

    image = preprocess(Image.open(current_dir + "/../docs/CLIP.png")).unsqueeze(0)
    text = tokenizer.tokenize(["a diagram", "a dog", "a cat"])

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    assert text_probs.cpu().numpy()[0].tolist() == [1.0, 0.0, 0.0]

if __name__ == "__main__":
    print("Load model")
    rn50 = timm.create_model('resnet50', pretrained=False, num_classes=1000)
    rn50.load_state_dict(torch.load("/scratch/mp5847/ben_ckpt/iclr-caption-paper/yfcc-intlab-ours-strict-2_2m.pt")['state_dict'])
    # model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='/scratch/mp5847/ben_ckpt/iclr-caption-paper/yfcc-vl-ours-strict-2_2m.pt')

    print("Done")