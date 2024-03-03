from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from model import AttentionR2Unet
import time
import os
import torch
import argparse
from data_loader import CustomLoader
from dataset import SegmentationDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def prepare_plot(origImage, origMask, predMask, i):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))

    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage.cpu())
    ax[1].imshow(origMask.cpu())
    ax[2].imshow(predMask.cpu())

    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")
    # set the layout of the figure and save it
    figure.tight_layout()
    file = "result"+str(i)+".png"
    figure.savefig(os.path.join(out_path, file))

def test(testLoader, model_path, out_path):
    # Load model and its state dictionary for inference
    model = AttentionR2Unet()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    #set model to evaluation no gradient mode
    model.eval()
    imgs = []
    ground_truth = []
    prediction_tests = []
    with torch.no_grad():
        for (i,(x,y)) in enumerate(testLoader):
          imgs.append(x.to(device))
          ground_truth.append(y.to(device))
          prediction_tests.append(torch.sigmoid(model(imgs[i])))

    # Store the prediction outputs
    for i in range(len(prediction_tests)):
        min = prediction_tests[i].min()
        max = prediction_tests[i].max()
        threshold = torch.tensor([(min+max)/2]).to(device)
        results = (prediction_tests[i]>threshold).float()*1
        image = imgs[i].permute(0,2,3,1)
        gt = ground_truth[i].permute(0,2,3,1)
        pred = results.permute(0,2,3,1)
        prepare_plot(image.squeeze(), gt.squeeze(),pred.squeeze(), i, out_path)



def train(model_path, trainLoader, valLoader, BATCH_SIZE = 8, NUM_EPOCHS = 50, RESUME=True):
    # initialize model
    model = AttentionR2Unet()
    model.to(device)
    lossFunc = torch.nn.BCELoss()
    opt = Adam(model.parameters(), lr=1e-3)
    scheduler = ExponentialLR(opt, gamma=0.9)
    # load model from file if checkpoint found
    if RESUME==True:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print("Found checkpoint: ")
    else:
        epoch = 0

    # Training
    trainSteps = len(trainLoader) // BATCH_SIZE
    valSteps = len(valLoader) // BATCH_SIZE
    H = {"train_loss": [], "val_loss": []}
    print("[INFO] training the network...")
    startTime = time.time()
    for e in (range(NUM_EPOCHS)):
        print("Epoch", epoch + e +1)
        model.train()

        totalTrainLoss = 0
        totalValLoss = 0

        print("Forward Pass")
        for (i, (x, y)) in enumerate(trainLoader):
            (x, y) = (x.to(device), y.to(device))
            pred = model(x)
            pred_probs = torch.sigmoid(pred)
            pred_flat = pred_probs.view(pred_probs.size(0),-1)
            y_flat = y.view(y.size(0),-1)
            loss = lossFunc(pred_flat, y_flat)
            opt.zero_grad()
            loss.backward()
            opt.step()

            totalTrainLoss += loss

        # validation
        print("Backward Pass")
        with torch.no_grad():
            model.eval()
            for (x, y) in valLoader:
                (x, y) = (x.to(device), y.to(device))
                pred = model(x)
                pred_probs = torch.sigmoid(pred)
                pred_flat = pred_probs.view(pred_probs.size(0),-1)
                y_flat = y.view(y.size(0),-1)
                totalValLoss += lossFunc(pred_flat, y_flat)

        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        torch.save({
        'epoch': epoch + e + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss':loss,
        }, model_path)

        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(epoch + e + 1, epoch + NUM_EPOCHS))
        print("Train loss: {:.6f}, Valss: {:.4f}".format(avgTrainLoss, avgValLoss))
        scheduler.step()

    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))

def main(BATCH_SIZE, NUM_EPOCHS, RESUME, model_path, images_folder, masks_folder, train, test, out_path):
    dataloader = CustomLoader(images_folder, masks_folder)
    images = dataloader.iter_images()
    masks = dataloader.iter_masks()
    trainLoader, valLoader, testLoader = dataloader.get_items(images, masks)
    if test == True:
        test(testLoader, model_path, out_path)
    if train == True:
        train(BATCH_SIZE, NUM_EPOCHS, RESUME, model_path, trainLoader, valLoader)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #paths
    parser.add_argument('--model_path', type=str, required=False, help='Model path to resume training/ path to save. Default = Root path ', default = "r2unet.pt")
    parser.add_argument('--images_folder', type=str, required=True, help='Path to the images folder')
    parser.add_argument('--masks_folder', type=str, required=True, help='Path to the masks folder')
    parser.add_argument('--epochs', required=False, help='number of epochs to train. Default=50')
    parser.add_argument('--batch_size', required=False, help='Batch size. Default = 8')
    parser.add_argument('--out_path', type=str, required=True, help="Path to store the ouput inferences")
    #other params
    parser.add_argument('--test', required= True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--resume_training', default = True, help='Resume the training process')
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.batch_size
    RESUME = args.resume_training
    model_path = args.model_path
    images_folder = args.images_folder
    masks_folder = args.masks_folder
    test = args.test
    train = args.train
    out_path = args.out_path
    main(BATCH_SIZE, NUM_EPOCHS, RESUME, model_path, images_folder, masks_folder, test, train, out_path)
