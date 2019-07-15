sm = nn.Softmax()
test_transforms = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
import matplotlib.pyplot as plt
def im_convert(tensor):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image= image.numpy()
    image = image.transpose(1,2,0)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.clip(0, 1)
    return image
def preprocess(path):
  img = cv2.imread(path)
  img = test_transforms(img)
  img = img.unsqueeze(0)
  return img
def cam(model,path):
  img = preprocess(path)
  fmap,logits = model(img.to('cuda'))
  params = list(model.parameters())
  weight_softmax = model.linear.weight.detach().cpu().numpy()
  logits = sm(logits)
  idx = np.argmax(logits.detach().cpu().numpy())
  bz, nc, h, w = fmap.shape
  out = np.dot(fmap.detach().cpu().numpy().reshape((nc, h*w)).T,weight_softmax[idx,:].T)
  cam = out.reshape(h,w)
  cam = cam - np.min(cam)
  cam_img = cam / np.max(cam)
  cam_img = np.uint8(255*cam_img)
  out = cv2.resize(cam_img, (im_size,im_size))
  heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
  img = im_convert(img)
  result = heatmap * 0.5 + img*0.8*255
  cv2.imwrite('/content/1.png',result)
  result1 = heatmap * 0.5/255 + img*0.8
  r,g,b = cv2.split(result1)
  result1 = cv2.merge((b,g,r))
  plt.imshow(result1)
  plt.show()
  
