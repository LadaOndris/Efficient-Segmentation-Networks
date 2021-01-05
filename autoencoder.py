import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize
import pickle
from tqdm import tqdm

class CAE(nn.Module):
  def __init__(self):
    super().__init__()

    self.pool = nn.MaxPool2d(kernel_size=2)
    self.l1 = nn.Sequential(
      nn.Conv2d(1,32,kernel_size=3, padding=2),
      nn.ReLU(),
      )

    self.l2 = nn.Sequential(
      nn.Conv2d(32,16,kernel_size=3, padding=2),
      nn.ReLU(),
      )

    self.l3 = nn.Sequential(
      nn.Conv2d(16,8,kernel_size=3, padding=2),
      nn.ReLU(),
      )

    self.l4 = nn.Sequential(
      nn.Conv2d(8,4,kernel_size=3, padding=1),
      nn.ReLU(),
      )

    self.l5 = nn.Sequential(
      nn.Conv2d(4,1,kernel_size=3, padding=1),
      nn.ReLU(),
      )

    self.drop_out = nn.Dropout(p=0.2)

    self.up1 = nn.Sequential(
      nn.ConvTranspose2d(1,4,kernel_size=3, stride=2),
      nn.ReLU(),
      )

    self.up2 = nn.Sequential(
      nn.ConvTranspose2d(4,8,kernel_size=3, stride=2),
      nn.ReLU(),
      )

    self.up3 = nn.Sequential(
      nn.ConvTranspose2d(8,16,kernel_size=3, stride=2),
      nn.ReLU(),
      )

    self.up4 = nn.Sequential(
      nn.ConvTranspose2d(16,32,kernel_size=3, stride=2),
      nn.ReLU(),
      )

    self.up5 = nn.Sequential(
      nn.ConvTranspose2d(32,1,kernel_size=2, stride=2, padding=(7,15)),
      nn.Sigmoid(),
      )

  def forward(self, x):
    x = self.l1(x)
    x = self.pool(x)
    x = self.l2(x)
    x = self.pool(x)
    x = self.l3(x)
    x = self.pool(x)
    x = self.l4(x)
    x = self.pool(x)
    x = self.l5(x)
    x = self.pool(x)
    x = self.drop_out(x)

    bottleneck = torch.flatten(x)

    x = self.up1(x)
    x = self.up2(x)
    x = self.up3(x)
    x = self.up4(x)
    x = self.up5(x)

    return x, bottleneck

##### DATALOADER #####
trf = T.Compose([T.ToTensor()])
from torch.utils.data import Dataset, DataLoader, sampler
from pathlib import Path

class BDD100K(Dataset):
  def __init__(self,img_dir,gt_dir,seg_model=None,gt=False,pytorch=True):
    super().__init__()
    # Loop through the files in red folder and combine, into a dictionary, the other bands
    self.files = [self.combine_files(f, gt_dir) for f in img_dir.iterdir() if not f.is_dir()]
    self.pytorch = pytorch
    self.gt = gt
      
  def combine_files(self, img_file: Path, gt_dir):
    files = {'img': img_file,
             'gt': Path(str(gt_dir/img_file.name).split('.')[0] + '_drivable_id.png')}
    return files
                                     
  def __len__(self):
    return len(self.files)

  def __getitem__(self, index):
    if self.gt:
      trf2 = T.Compose([T.Resize((176,320)),T.ToTensor()])
      return 0, trf2(Image.open(self.files[index]['gt']))
    else:
      datas = pickle.load(open('./dataset/inform/bdd100k_inform.pkl', "rb"))
      img = Image.open(self.files[index]['img'])
      image = np.asarray(img, np.float32)
      image = resize(image, (176,320), order=1, preserve_range=True)

      image -= datas['mean']
      # image = image.astype(np.float32) / 255.0
      image = image[:, :, ::-1]  # revert to RGB
      image = image.transpose((2, 0, 1))  # HWC -> CHW

      image = torch.from_numpy(image.copy())

      segmentation.eval()
      y = segmentation(image.unsqueeze(0))
      y = y.cpu().data[0].numpy()
      y = y.transpose(1, 2, 0)

      y = np.asarray(np.argmax(y, axis=2), dtype=np.float32)
      y[y==2] = 0
      y = torch.from_numpy(y.copy()).unsqueeze(0)

      return trf(img), y
##############

if __name__ == "__main__":
  from builders.model_builder import build_model

  autoencoder = CAE().cuda()
  segmentation = build_model('FastSCNN',num_classes=3)
  checkpoint = torch.load('./checkpoint/bdd100k/FastSCNNbs200gpu1_train/model_8.pth')
  segmentation.load_state_dict(checkpoint['model'])
  
  train_ds = BDD100K( Path('./dataset/bdd100k/images/100k/train/'),
                     Path('./dataset/bdd100k/drivable_maps/labels/train/'),
                     segmentation,True
                     )
  valid_ds = BDD100K( Path('./dataset/bdd100k/images/100k/val/'),
                     Path('./dataset/bdd100k/drivable_maps/labels/val/'),
                     segmentation,True
                     )

  train_dl = DataLoader(train_ds, batch_size=12, shuffle=True)
  valid_dl = DataLoader(valid_ds, batch_size=12, shuffle=True)

  def train(model,dl,criterion,optimizer,epochs):
    l = []
    for epoch in range(1,epochs+1):
      train_loss = 0.0
      t = tqdm(total=len(dl),desc='Episodes')
      i=0
      for _, images in dl:
        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs,images)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*images.size(0)
        i+=1
        if i%50 == 0:
          torch.save(model, "./checkpoint/autoencoder/last_autoencoder.pt")
        t.set_description(f'Episodes (loss: {round(float(loss),6)})')
        t.update(1)
      t.close()
      train_loss = train_loss/len(dl)
      print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
      torch.save(model, "./checkpoint/autoencoder/autoencoder_"+str(epoch)+".pt")
      l.append(train_loss)
    return l

  ######TRAINING##########
  training_mode = True
  if training_mode:
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(autoencoder.parameters(),lr=0.0001)
    train_loss = train(autoencoder, valid_dl, loss_fn, opt, epochs=5)

    pickle.dump(train_loss, open("./checkpoint/autoencoder/loss_stats_autoencoder.pkl","wb"))
    torch.save(autoencoder,"./checkpoint/autoencoder/last_autoencoder.pt")

  #######TESTING###########
  trained_autoencoder = torch.load("./checkpoint/autoencoder/last_autoencoder.pt")
  trans = T.ToPILImage(mode='RGB')
  trans2 =T.ToPILImage(mode='L')
  x,y = train_ds[2]
  plt.imshow(trans2(x.squeeze())); plt.show()
  plt.imshow(trans2(y.squeeze())); plt.show()
  print(y.shape,y)

  start = time.time()
  pred, _ = trained_autoencoder(y.unsqueeze(0))
  print(time.time()-start,'seconds')
  plt.imshow(trans2(pred.squeeze())); plt.show()
