import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataset import *
from PIL import Image
from torchvision import transforms
import torch

def join_objects(img_ids, aug_img, dataset):
  boxes_list = list()
  jmp = int(aug_img.shape[0] / (len(img_ids) + 1))

  for i, img_id in enumerate(img_ids):
    path = dataset.image_reference(img_id)
    k_img = Image.open(path)
    img = dataset.load_image(img_id)
    img = np.asarray(img)
    
    mask = get_mask(k_img)
    idx = np.nonzero(mask)

    aug_img[add_idx(idx, i*jmp, i*jmp)] = img[idx]
    aug_img = aug_img.astype(int)
    ann_path = (path[:-4] + ".xml").replace("images","annots")

    boxes, w, h = dataset.extract_boxes(ann_path)
    for j, box in enumerate(boxes):
      box = np.array(box)
      box += i * jmp
      boxes[j] = box
    boxes_list.append(boxes)

  fig,ax = plt.subplots(1)
  ax.imshow(aug_img)
  draw_boxes(boxes_list, ax)


def get_mask(input_img):
  # https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
  model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
  model.eval()
  
  preprocess = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  input_tensor = preprocess(input_img)
  input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

  # move the input and model to GPU for speed if available
  if torch.cuda.is_available():
      input_batch = input_batch.to('cuda')
      model.to('cuda')

  with torch.no_grad():
      output = model(input_batch)['out'][0]
  output_predictions = output.argmax(0)

  # create a color pallette, selecting a color for each class
  palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
  colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
  colors = (colors % 255).numpy().astype("uint8")

  # plot the semantic segmentation predictions of 21 classes in each color
  # Note: modified the resizing
  r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_img.size)
  print(r.size)
  return r


def draw_boxes(boxes_list, ax):
  # TODO: support for multiple boxes in annotation file
  for boxes in boxes_list:
    for box in boxes:   
      # w,h are for the whole image
      xmin, ymin, xmax, ymax = box
      rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1,edgecolor='r', fill=False)
      ax.add_patch(rect)
  plt.show()


def add_idx(idx, x_off, y_off):
  idx = list(idx)
  idx[1] += x_off
  idx[0] += y_off
  return tuple(idx)
  
   
def show_images(img_ids, dataset):
  for img_id in img_ids:
    path = dataset.image_reference(img)
    fig, ax = plt.subplots(1)
    img = Image.open(path)
    plt.imshow(img)


#aug_img = np.zeros((1000,1000,3))

train_set = KangarooDataset()
train_set.load_kangaroos('kangaroo', is_train=True)
train_set.prepare()
print('training set size: %d' % len(train_set.image_ids))

backg = Image.open("grass.webp")
backg = backg.resize((900, 900))

aug_img = np.asarray(backg)

img_ids = [9, 71]
show_images(img_ids, train_set)
join_objects(img_ids, aug_img.copy(), train_set)

