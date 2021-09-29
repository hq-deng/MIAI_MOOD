import os
from PIL import Image
from model import resnet18,DeepLabHeadV3Plus,ResSeg
import nibabel as nib
import numpy as np
import torchvision.transforms as transforms
import torch
from torch.nn import functional as F

#import matplotlib.pyplot as plt
#from sklearn.metrics import roc_auc_score,average_precision_score


def get_data_transforms(image_size):
    img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return img_transform

def predict_folder_pixel_abs(input_folder, target_folder,model,device,scale):
    for m in model:
        m.eval()
    
    data_transform = get_data_transforms(scale)
    
    abnomal_score_array = []

    #test
    #groud_truth_array = []
    
    with torch.no_grad():

        for f in os.listdir(input_folder):

            source_file = os.path.join(input_folder, f)
            target_file = os.path.join(target_folder, f)

            #test
            #mask_folder = "/home/hanqiu/Desktop/FB2S/docker_example/scripts/dataset/abdom_mask"
            #mask_file = os.path.join(mask_folder, f)
            #mask = nib.load(mask_file)
            #mask_array = mask.get_fdata()
            #groud_truth_array.append(mask_array)
            

            nimg = nib.load(source_file)
            nimg_array = nimg.get_fdata().astype(np.float16)
            
            
            ax_len = nimg_array.shape

            each_score = []

            for ax in range(len(model)):
                print(ax)
                next_output = torch.tensor([]).to(device)
                for i in range(ax_len[ax]):
                    if ax==0:
                        img = nimg_array[i,:,:]
                    elif ax==1:
                        img = nimg_array[:,i,:]
                    else:
                        img = nimg_array[:,:,i]
                    img = Image.fromarray(np.uint8(img*255)).convert('L')
                    img = data_transform(img)
                    img = img.repeat(1, 3, 1, 1)
                    img = img.to(device)
                    output = model[ax](img)
                    output = F.softmax(output,1)[:,1,:,:]
                    output = F.interpolate(output.unsqueeze(0), scale_factor=2, mode='bilinear', align_corners=False)
                    next_output = torch.cat((next_output,output[0]))
                    #if i >=2:
                    #    print(next_output.permute(1,2,0).shape)
                    #    assert 1==2
                if ax==0:
                    each_score.append(next_output.cpu().numpy())
                elif ax==1:
                    each_score.append(next_output.permute(1,0,2).cpu().numpy())
                elif ax==2:
                    each_score.append(next_output.permute(1,2,0).cpu().numpy())
            #abnomal_score_array.append(np.mean(each_score,axis=0))
            #abnomal_score_array.append(each_score[-1])
            
            final_nimg = nib.Nifti1Image(np.mean(each_score,axis=0), affine=nimg.affine)
            nib.save(final_nimg, target_file)

        #del next_output
        #del each_score
        #del final_nimg
        #del nimg
        #del nimg_array
        #del model
        #print(np.array(abnomal_score_array).shape,np.array(groud_truth_array).shape)
        #plt.imshow(all_mask[1][:,:,100]*255,cmap=plt.cm.gray)
        #plt.show()
        #test
        #pred_list = np.array(abnomal_score_array).flatten()
        #print(pred_list.shape)
        #del abnomal_score_array
        #gdth_list = np.array(groud_truth_array).flatten()#.astype(int)

        #print(pred_list.shape,gdth_list.shape)
        
        #ap_smp = round(average_precision_score(gdth_list, pred_list),4)
        #print(ap_smp)
        

def predict_folder_sample_abs(input_folder, target_folder,model,device,scale):
    for m in model:
        m.eval()
    
    data_transform = get_data_transforms(scale)
    
    
    with torch.no_grad():
        
        abnomal_score_list = []
        
        file_list = os.listdir(input_folder)
        
    
        for f in file_list:

            source_file = os.path.join(input_folder, f)
            target_file = os.path.join(target_folder, f)

            nimg = nib.load(source_file)
            nimg_array = nimg.get_fdata().astype(np.float16)
            
            x,y,z = nimg_array.shape

            each_score = []

            for ax in range(len(model)):
            
                next_output = torch.tensor([]).to(device)
                for i in range(z):

                    if ax==0:
                        img = nimg_array[i,:,:]
                    elif ax==1:
                        img = nimg_array[:,i,:]
                    else:
                        img = nimg_array[:,:,i]
                
                    img = Image.fromarray(np.uint8(img*255)).convert('L')
                    img = data_transform(img)
                    img = img.repeat(1, 3, 1, 1)
                    img = img.to(device)
                    output = model[ax](img)
                
                
                    output = F.softmax(output,1)[:,1,:,:]
                
                
                    #output = F.interpolate(output.unsqueeze(0), scale_factor=2, mode='bilinear', align_corners=False)
                    next_output = torch.cat((next_output,output[0]))
                each_score.append(next_output.cpu().numpy())
            abnomal_score = np.sum(each_score)
            abnomal_score_list.append(abnomal_score)
        
        abnomal_score_list = (abnomal_score_list-np.min(abnomal_score_list))/(np.max(abnomal_score_list)-np.min(abnomal_score_list))
        
        for i in range(len(abnomal_score_list)):
            with open(os.path.join(target_folder, file_list[i] + ".txt"), "w") as write_file:
                write_file.write(str(abnomal_score_list[i]))


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-mode", type=str, default='pixel', help="can be either 'pixel' or 'sample'.", required=False)
    parser.add_argument("-data", type=str, default='brain', help="can be either 'brain' or 'abdom'.", required=False)
    parser.add_argument("-device", type=str, default='cpu', help="can be either 'cpu' or 'cuda'.", required=False)

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    mode = args.mode
    data = args.data
    device = args.device
    
    if data == 'brain':
        #checkpoint_x = '/workspace/checkpoints/brain_seg_x.pth'
        checkpoint_x = './checkpoints/brain_seg_x.pth'
        checkpoint_y = './checkpoints/brain_seg_y.pth'
        checkpoint_z = './checkpoints/brain_seg_z.pth'
        scale = 128
    elif data == 'abdom':
        checkpoint_x = '/workspace/checkpoints/abdom_seg_x.pth'
        checkpoint_y = '/workspace/checkpoints/abdom_seg_y.pth'
        checkpoint_z = '/workspace/checkpoints/abdom_seg_z.pth'
        scale = 256
    else:
        print("Mode not correctly defined. Either choose 'brain' oder 'abdom'")
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #backbone = resnet18(pretrained=False)
    #segmentor = DeepLabHeadV3Plus(512,64,2)
    resseg_x = ResSeg(resnet18(pretrained=False),DeepLabHeadV3Plus(512,64,2))
    resseg_y = ResSeg(resnet18(pretrained=False),DeepLabHeadV3Plus(512,64,2))
    resseg_z = ResSeg(resnet18(pretrained=False),DeepLabHeadV3Plus(512,64,2))
    resseg_x = resseg_x.to(device)
    resseg_y = resseg_y.to(device)
    resseg_z = resseg_z.to(device)
    resseg_x.load_state_dict(torch.load(checkpoint_x)['model'])
    resseg_y.load_state_dict(torch.load(checkpoint_y)['model'])
    resseg_z.load_state_dict(torch.load(checkpoint_z)['model'])

    if mode == "pixel":
        predict_folder_pixel_abs(input_dir, output_dir,[resseg_x,resseg_y,resseg_z],device,scale)
    elif mode == "sample":
        predict_folder_sample_abs(input_dir, output_dir,[resseg_x,resseg_y,resseg_z],device,scale)
    else:
        print("Mode not correctly defined. Either choose 'pixel' oder 'sample'")

    # predict_folder_sample_abs("/home/david/data/datasets_slow/mood_brain/toy", "/home/david/data/datasets_slow/mood_brain/target_sample")
