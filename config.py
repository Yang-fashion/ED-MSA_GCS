#  dataset dataset37 cell_division polyp_colon skin_lesion
dataset = "dataset37"
train_folder_path = "/root/projects/Datasets/%s/train" % dataset
test_folder_path = "/root/projects/Datasets/%s/test" % dataset
train_img_path = "/root/projects/Datasets/%s/train/imgs/" % dataset
train_mask_path = "/root/projects/Datasets/%s/train/masks/" % dataset

test_img_path = "/root/projects/Datasets/%s/test/imgs/" % dataset
test_mask_path = "/root/projects/Datasets/%s/test/masks/" % dataset

scale = 1

epoch = 200
start_epoch = 0
lr = 0.001
batch_size = 8


print("datasets is :", dataset)
