import os
import torch
import numpy as np
import nibabel as nib
from monai.data import write_nifti  # MONAI 0.4.0 has write_nifti but not read_nifti
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

def read_nifti_file(file_path):
    """
    Custom function to read NIfTI files since monai.data.read_nifti is not available in MONAI 0.4.0
    
    Args:
        file_path: Path to the NIfTI file
        
    Returns:
        numpy array containing the data
    """
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata()

def save_mask_as_nifti(
    mask,               # tensor or array of shape [N, C, D, H, W] where C can be 1 or 2
    output_dir,         # directory to save masks
    modality,           # 't1' or 't2', or None if multiple channels are present
    file_prefix="case", # prefix for filenames
    patient_ids=None,   # optional list of patient IDs
    reference_files=None # optional list of reference nifti files
):
    """
    Save prediction masks as NIfTI files, handling either single or dual-channel masks.
    
    Args:
        mask: Tensor or array of shape [N, C, D, H, W] where C can be 1 or 2
        output_dir: Directory to save the NIfTI files
        modality: Modality name ('t1' or 't2'), or None if multiple channels are present
        file_prefix: Prefix for the filenames
        patient_ids: Optional list of patient IDs instead of using indices
        reference_files: Optional list of reference NIfTI files to copy header info
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy if it's a torch tensor
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    num_patients = mask.shape[0]
    num_channels = mask.shape[1]
    
    # Define modality names based on channel count
    if num_channels == 1:
        modality_names = [modality]
    elif num_channels == 2:
        modality_names = ['t1', 't2']
    else:
        raise ValueError(f"Unexpected number of channels: {num_channels}, expected 1 or 2")
    
    # Process each patient
    for i in range(num_patients):
        # Determine patient ID
        if patient_ids and i < len(patient_ids):
            patient_id = patient_ids[i]
        else:
            patient_id = f"{file_prefix}_{i:03d}"
        
        # Get reference header and affine if provided
        affine = None
        if reference_files and i < len(reference_files):
            ref_img = nib.load(reference_files[i])
            affine = ref_img.affine
        
        # Process each channel
        for c in range(num_channels):
            # Extract single patient mask for this channel
            patient_mask = mask[i, c]  # [D, H, W]
            
            # Get modality name for this channel
            channel_modality = modality_names[c]
            
            # Filename
            filename = os.path.join(output_dir, f"{patient_id}_{channel_modality}_mask.nii.gz")
            
            # Use MONAI's write_nifti
            write_nifti(
                data=patient_mask,
                file_name=filename,
                affine=affine,
                resample=False
                # MONAI 0.4.0 doesn't have output_dtype parameter
                # output_dtype=np.uint8 if patient_mask.max() <= 1 else np.float32
            )
            print(f"Saved: {filename}")


import os
import logging
import numpy as np
from natsort import natsorted
from time import perf_counter
import glob
import csv
from time import strftime
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from matplotlib import pyplot as plt
import monai
from monai.transforms import (
    Compose,
    LoadNiftid,
    AddChanneld,
    NormalizeIntensityd,
    SpatialPadd,
    RandFlipd,
    RandSpatialCropd,
    Orientationd,
    ToTensord,
)
from monai.networks.layers import Norm
from monai.data import NiftiSaver
# from torchviz import make_dot
# import hiddenlayer as hl
from .networks.nets.unet2d5_spvPA import UNet2d5_spvPA
from .losses.dice_spvPA import Dice_spvPA
from monai.inferers import sliding_window_inference

monai.config.print_config()


class VSparams:
    def __init__(self, parser):
        parser.add_argument("--debug", dest="debug", action="store_true", help="activate debugging mode")
        parser.set_defaults(debug=False)

        parser.add_argument("--split", type=str, default="./params/split_TCIA.csv", help="path to CSV file that defines"
                                                                                         " training, validation and"
                                                                                         " test datasets")
        parser.add_argument("--dataset", type=str, default="T1", help='(string) use "T1" or "T2" to select dataset')
        parser.add_argument("--train_batch_size", type=int, default=1, help="batch size of the forward pass")
        parser.add_argument("--initial_learning_rate", type=float, default=1e-4, help="learning rate at first epoch")
        parser.add_argument(
            "--no_attention",
            dest="attention",
            action="store_false",
            help="disables the attention module in "
            "the network and the attention map "
            "weighting in the loss function",
        )
        parser.set_defaults(attention=True)
        parser.add_argument(
            "--no_hardness",
            dest="hardness",
            action="store_false",
            help="disables the hardness weighting in " "the loss function",
        )
        parser.set_defaults(hardness=True)
        parser.add_argument(
            "--results_folder_name", type=str, default="temp" + strftime("%Y%m%d%H%M%S"), help="name of results folder"
        )

        args = parser.parse_args()

        self.debug = args.debug
        self.dataset = args.dataset
        self.data_root = "./data/VS_defaced/"  # set path to data set
        self.split_csv = args.split
        if self.debug:
            self.split_csv = "./params/split_debug.csv"
        self.pad_crop_shape = [384, 384, 64]
        if self.debug:
            self.pad_crop_shape = [128, 128, 32]
        self.pad_crop_shape_test = [384, 384, 64]
        if self.debug:
            self.pad_crop_shape_test = [128, 128, 32]
        self.num_workers = 4
        self.torch_device_arg = "cuda:0"
        self.train_batch_size = args.train_batch_size
        self.initial_learning_rate = args.initial_learning_rate
        self.epochs_with_const_lr = 100
        if self.debug:
            self.epochs_with_const_lr = 3
        self.lr_divisor = 2.0
        self.weight_decay = 1e-7
        self.num_epochs = 300
        if self.debug:
            self.num_epochs = 10
        self.val_interval = 2  # determines how frequently validation is performed during training
        self.model = "UNet2d5_spvPA"
        self.sliding_window_inferer_roi_size = [384, 384, 64]
        if self.debug:
            self.sliding_window_inferer_roi_size = [128, 128, 32]
        self.attention = args.attention
        self.hardness = args.hardness
        self.export_inferred_segmentations = True

        # paths
        self.results_folder_path = os.path.join(self.data_root, "results", args.results_folder_name)
        if self.debug:
            self.results_folder_path = os.path.join(self.data_root, "results", "debug")
        self.logs_path = os.path.join(self.results_folder_path, "logs")
        self.model_path = os.path.join(self.results_folder_path, "model")
        self.figures_path = os.path.join(self.results_folder_path, "figures")

        #
        self.device = torch.device(self.torch_device_arg)

    def create_results_folders(self):
        # create results folders for logs, figures and model
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path, exist_ok=False)
            os.chmod(self.logs_path, 0o777)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path, exist_ok=False)
            os.chmod(self.model_path, 0o777)
        if not os.path.exists(self.figures_path):
            os.makedirs(self.figures_path, exist_ok=False)
            os.chmod(self.figures_path, 0o777)

    def set_up_logger(self, log_file_name):
        # logging settings
        self.logger = logging.getLogger()
        fileHandler = logging.FileHandler(os.path.join(self.logs_path, log_file_name), mode="w")
        consoleHandler = logging.StreamHandler()
        self.logger.addHandler(fileHandler)
        self.logger.addHandler(consoleHandler)
        formatter = logging.Formatter("%(asctime)s %(levelname)s        %(message)s")
        fileHandler.setFormatter(formatter)
        consoleHandler.setFormatter(formatter)
        self.logger.setLevel(logging.INFO)
        self.logger.info("Created " + log_file_name)
        return self.logger

    def log_parameters(self):
        logger = self.logger
        # write all parameters to log
        logger.info("-" * 10)
        logger.info("Parameters: ")
        logger.info("dataset =                          {}".format(self.dataset))
        logger.info("data_root =                        {}".format(self.data_root))
        logger.info("split_csv =                        {}".format(self.split_csv))
        logger.info("pad_crop_shape =                   {}".format(self.pad_crop_shape))
        logger.info("pad_crop_shape_test =              {}".format(self.pad_crop_shape_test))
        logger.info("num_workers =                      {}".format(self.num_workers))
        logger.info("torch_device_arg =                 {}".format(self.torch_device_arg))
        logger.info("train_batch_size =                 {}".format(self.train_batch_size))
        logger.info("initial_learning_rate =            {}".format(self.initial_learning_rate))
        logger.info("epochs_with_const_lr =             {}".format(self.epochs_with_const_lr))
        logger.info("lr_divisor =                       {}".format(self.lr_divisor))
        logger.info("weight_decay =                     {}".format(self.weight_decay))
        logger.info("num_epochs =                       {}".format(self.num_epochs))
        logger.info("val_interval =                     {}".format(self.val_interval))
        logger.info("model =                            {}".format(self.model))
        logger.info("sliding_window_inferer_roi_size =  {}".format(self.sliding_window_inferer_roi_size))

        logger.info("attention =                        {}".format(self.attention))
        logger.info("hardness =                         {}".format(self.hardness))

        logger.info("results_folder_path =              {}".format(self.results_folder_path))
        logger.info("export_inferred_segmentations =    {}".format(self.export_inferred_segmentations))
        logger.info("-" * 10)
    def load_samples(self, n=None,t="T1"):
        """
        Load T1 or T2 data files.
        
        Args:
            n (int, optional): Number of samples to return from each set. If None, return all data.
        
        Returns:
            tuple: (train_files, val_files, test_files) - Each containing at most n items
        """
        logger = self.logger
        train_files, val_files, test_files = [], [], []
        with open(self.split_csv) as csvfile:
            csvReader = csv.reader(csvfile)
            for row in csvReader:
                if t == "T1":
                    image_name = os.path.join(self.data_root, 'input_data', row[0], 'vs_gk_t1_refT1.nii.gz')
                    label_name = os.path.join(self.data_root, 'input_data', row[0], 'vs_gk_seg_refT1.nii.gz')
                elif t == "T2":
                    image_name = os.path.join(self.data_root, 'input_data', row[0], 'vs_gk_t2_refT2.nii.gz')
                    label_name = os.path.join(self.data_root, 'input_data', row[0], 'vs_gk_seg_refT2.nii.gz')
                if row[1] == "training":
                    train_files.append({"image": image_name, "label": label_name})
                elif row[1] == "validation":
                    val_files.append({"image": image_name, "label": label_name})
                elif row[1] == "test":
                    test_files.append({"image": image_name, "label": label_name})
        
        # Apply subset limit if n is specified
        if n is not None:
            train_files = train_files[:n]
            val_files = val_files[:n]
            test_files = test_files[:n]
        
        # check if all files exist
        for file_dict in train_files + val_files + test_files:
            assert (os.path.isfile(file_dict['image'])), f" {file_dict['image']} is not a file"
            assert (os.path.isfile(file_dict['label'])), f" {file_dict['label']} is not a file"
        
        logger.info("Number of images in training set   = {}".format(len(train_files)))
        logger.info("Number of images in validation set = {}".format(len(val_files)))
        logger.info("Number of images in test set       = {}".format(len(test_files)))
        logger.info("training set   = {}".format(train_files))
        logger.info("validation set = {}".format(val_files))
        logger.info("test set       = {}".format(test_files))
        # return as dictionaries of image/label pairs
        return train_files, val_files, test_files

    def load_T1_or_T2_data(self):
        logger = self.logger

        train_files, val_files, test_files = [], [], []

        with open(self.split_csv) as csvfile:
            csvReader = csv.reader(csvfile)
            for row in csvReader:
                if self.dataset == "T1":
                    image_name = os.path.join(self.data_root, 'input_data', row[0], 'vs_gk_t1_refT1.nii.gz')
                    label_name = os.path.join(self.data_root, 'input_data', row[0], 'vs_gk_seg_refT1.nii.gz')
                elif self.dataset == "T2":
                    image_name = os.path.join(self.data_root, 'input_data', row[0], 'vs_gk_t2_refT2.nii.gz')
                    label_name = os.path.join(self.data_root, 'input_data', row[0], 'vs_gk_seg_refT2.nii.gz')
                if row[1] == "training":
                    train_files.append({"image": image_name, "label": label_name})
                elif row[1] == "validation":
                    val_files.append({"image": image_name, "label": label_name})
                elif row[1] == "test":
                    test_files.append({"image": image_name, "label": label_name})

        # check if all files exist
        for file_dict in train_files + val_files + test_files:
            assert (os.path.isfile(file_dict['image'])), f" {file_dict['image']} is not a file"
            assert (os.path.isfile(file_dict['label'])), f" {file_dict['label']} is not a file"

        logger.info("Number of images in training set   = {}".format(len(train_files)))
        logger.info("Number of images in validation set = {}".format(len(val_files)))
        logger.info("Number of images in test set       = {}".format(len(test_files)))
        logger.info("training set   = {}".format(train_files))
        logger.info("validation set = {}".format(val_files))
        logger.info("test set       = {}".format(test_files))

        # return as dictionaries of image/label pairs
        return train_files, val_files, test_files

    def get_transforms(self):
        self.logger.info("Getting transforms...")
        # Setup transforms of data sets
        train_transforms = Compose(
            [
                LoadNiftid(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                NormalizeIntensityd(keys=["image"]),
                SpatialPadd(keys=["image", "label"], spatial_size=self.pad_crop_shape),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandSpatialCropd(
                    keys=["image", "label"], roi_size=self.pad_crop_shape, random_center=True, random_size=False
                ),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadNiftid(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                NormalizeIntensityd(keys=["image"]),
                SpatialPadd(keys=["image", "label"], spatial_size=self.pad_crop_shape),
                RandSpatialCropd(
                    keys=["image", "label"], roi_size=self.pad_crop_shape, random_center=True, random_size=False,
                ),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadNiftid(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                NormalizeIntensityd(keys=["image"]),
                ToTensord(keys=["image", "label"]),
            ]
        )

        return train_transforms, val_transforms, test_transforms

    @staticmethod
    def get_center_of_mass_slice(label):
        # calculate center of mass of label in through plan direction to select a slice that shows the tumour
        num_slices = label.shape[2]
        slice_masses = np.zeros(num_slices)
        for z in range(num_slices):
            slice_masses[z] = label[:, :, z].sum()

        if sum(slice_masses) == 0:  # if there is no label in the cropped image
            slice_weights = np.ones(num_slices) / num_slices  # give all slices equal weight
        else:
            slice_weights = slice_masses / sum(slice_masses)

        center_of_mass = sum(slice_weights * np.arange(num_slices))
        slice_closest_to_center_of_mass = int(center_of_mass.round())
        return slice_closest_to_center_of_mass

    def check_transforms_on_first_validation_image_and_label(self, val_files, val_transforms):
        logger = self.logger
        # check the transforms on the first validation set image
        check_ds = monai.data.Dataset(data=val_files, transform=val_transforms)  # apply transform
        check_loader = DataLoader(check_ds, batch_size=1)
        check_data = monai.utils.misc.first(check_loader)  # gets the first item from an input iterable
        image, label = (check_data["image"][0][0], check_data["label"][0][0])
        logger.info("-" * 10)
        logger.info("Check the transforms on the first validation set image and label")
        logger.info(
            "Length of check_data = {}".format(len(check_data))
        )  # this dictionary also contains all the nifti header info
        logger.info("check_data['image'].shape = {}".format(check_data["image"].shape))
        logger.info("Validation image shape = {}".format(image.shape))
        logger.info("Validation label shape = {}".format(label.shape))

        slice_idx = self.get_center_of_mass_slice(
            label
        )  # choose slice of selected validation set image volume for the figure

        logger.info("-" * 10)
        logger.info("Plot one slice of the image and the label")
        logger.info("image shape: {}, label shape: {}, slice = {}".format(image.shape, label.shape, slice_idx))
        # plot the slice [:, :, slice]
        plt.figure("check", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title("image")
        plt.imshow(image[:, :, slice_idx], cmap="gray", interpolation="none")
        plt.subplot(1, 2, 2)
        plt.title("label")
        plt.imshow(label[:, :, slice_idx], interpolation="none")
        plt.savefig(os.path.join(self.figures_path, "check_validation_image_and_label.png"))

    # Set different seed for workers of DataLoader
    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        worker_info.dataset.transform.set_random_state(worker_info.seed % (2 ** 32))

    def cache_transformed_train_data(self, train_files, train_transforms):
        self.logger.info("Caching training data set...")
        # Define CacheDataset and DataLoader for training and validation
        train_ds = monai.data.CacheDataset(
            data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=self.num_workers
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=monai.data.list_data_collate,
            worker_init_fn=self.worker_init_fn,
        )
        return train_loader

    def cache_transformed_val_data(self, val_files, val_transforms):
        self.logger.info("Caching validation data set...")
        val_ds = monai.data.CacheDataset(
            data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=self.num_workers
        )
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=self.num_workers)
        return val_loader

    def cache_transformed_test_data(self, test_files, test_transforms):
        self.logger.info("Caching test data set...")
        test_ds = monai.data.CacheDataset(
            data=test_files, transform=test_transforms, cache_rate=1.0, num_workers=self.num_workers
        )
        test_loader = DataLoader(test_ds, batch_size=1, num_workers=self.num_workers)
        return test_loader



    def set_and_get_model(self):
        logger = self.logger
        logger.info("Setting up the model type...")

        if self.model == "UNet2d5_spvPA":

            model = UNet2d5_spvPA(
                dimensions=3,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 48, 64, 80, 96),
                strides=(
                    (2, 2, 1),
                    (2, 2, 1),
                    (2, 2, 2),
                    (2, 2, 2),
                    (2, 2, 2),
                ),
                kernel_sizes=(
                    (3, 3, 1),
                    (3, 3, 1),
                    (3, 3, 3),
                    (3, 3, 3),
                    (3, 3, 3),
                    (3, 3, 3),
                ),
                sample_kernel_sizes=(
                    (3, 3, 1),
                    (3, 3, 1),
                    (3, 3, 3),
                    (3, 3, 3),
                    (3, 3, 3),
                ),
                num_res_units=2,
                norm=Norm.BATCH,
                dropout=0.1,
                attention_module=self.attention,
            ).to(self.device)
        else:
            raise Exception("Model not defined.")

        # hl.build_graph(model, torch.zeros(2, 1, 128, 128, 32).to(self.device)).save("model")
        return model
        
    def set_and_get_loss_function(self):
        self.logger.info("Setting up the loss function...")
        loss_function = Dice_spvPA(
            to_onehot_y=True, softmax=True, supervised_attention=self.attention, hardness_weighting=self.hardness
        )
        return loss_function

    def set_and_get_optimizer(self, model):
        self.logger.info("Setting up the optimizer...")
        optimizer = torch.optim.Adam(model.parameters(), lr=self.initial_learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def compute_dice_score(self, predicted_probabilities, label):
        n_classes = predicted_probabilities.shape[1]
        y_pred = torch.argmax(predicted_probabilities, dim=1, keepdim=True)  # pick larger value of 2 channels
        y_pred = monai.networks.utils.one_hot(y_pred, n_classes)  # make 2 channel one hot tensor
        dice_score = torch.tensor(
            [
                [
                    1
                    - monai.losses.DiceLoss(
                        include_background=False, to_onehot_y=True, softmax=False, reduction="mean"
                    ).forward(y_pred, label)
                ]
            ],
            device=self.device,
        )
        return dice_score

    def run_training_algorithm(self, model, loss_function, optimizer, train_loader, val_loader):
        logger = self.logger
        logger.info("Running the training loop...")
        # TensorBoard Writer will output to ./runs/ directory by default
        tb_writer = SummaryWriter()

        # add an image grid to tensorboard
        if self.debug:
            images_for_grid = []
            for batch_data in train_loader:
                images, labels = batch_data["image"], batch_data["label"]
                for image, label in zip(images, labels):
                    central_slice_number = self.get_center_of_mass_slice(np.squeeze(label[0, :, :, :]))
                    images_for_grid.append(image[..., central_slice_number])
                    images_for_grid.append(label[..., central_slice_number])
            image_grid = torchvision.utils.make_grid(images_for_grid, normalize=True, scale_each=True)
            tb_writer.add_image("images", image_grid, 0)

        epochs_with_const_lr = self.epochs_with_const_lr
        val_interval = self.val_interval  # validation every val_interval epochs

        # Execute training process
        best_metric = -1  # stores highest mean Dice score obtained during validation
        best_metric_epoch = -1  # stores the epoch number during which the highest mean Dice score was obtained
        epoch_loss_values = list()  # stores losses of every epoch
        metric_values = list()  # stores Dice scores of every val_interval epoch
        num_epochs = self.num_epochs
        start = perf_counter()
        for epoch in range(num_epochs):
            logger.info("-" * 10)
            logger.info("Epoch {}/{}".format(epoch + 1, num_epochs))
            if epoch == val_interval:
                stop = perf_counter()
                logger.info(
                    (
                        "Average duration of first {0:.0f} epochs = {1:.2f} s. "
                        + "Expected total training time = {2:.2f} h"
                    ).format(
                        val_interval, (stop - start) / val_interval, (stop - start) * num_epochs / val_interval / 3600
                    )
                )
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs, labels = batch_data["image"].to(self.device), batch_data["label"].to(self.device)
                optimizer.zero_grad()  # reset the optimizer gradient
                outputs = model(inputs)  # evaluate the model
                # make_dot(outputs.mean(), params=dict(model.named_parameters())).render("attached", format="png")
                loss = loss_function(outputs, labels)  # returns the mean loss over the batch by default
                loss.backward()  # computes the gradients
                optimizer.step()  # update the model weights
                epoch_loss += loss.item()
                if epoch == 0:
                    logger.info(
                        "{}/{}, train_loss: {:.4f}".format(step, len(train_loader) // train_loader.batch_size, loss.item())
                    )
            epoch_loss /= step  # calculate mean loss over current epoch
            epoch_loss_values.append(epoch_loss)
            logger.info("epoch {} average loss: {:.4f}".format(epoch + 1, epoch_loss))

            # validation
            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():  # turns of PyTorch's auto grad for better performance
                    metric_sum = 0.0
                    metric_count = 0  # counts number of images
                    epoch_loss_val = 0
                    step = 0  # counts number of batches
                    for val_data in val_loader:  # loop over images in validation set
                        step += 1
                        val_inputs, val_labels = val_data["image"].to(self.device), val_data["label"].to(self.device)

                        val_outputs = model(val_inputs)

                        dice_score = self.compute_dice_score(val_outputs[0], val_labels)

                        loss = loss_function(val_outputs, val_labels)

                        metric_count += len(dice_score)
                        metric_sum += dice_score.sum().item()
                        epoch_loss_val += loss.item()

                        metric_count += len(dice_score)
                        metric_sum += dice_score.sum().item()
                        epoch_loss_val += loss.item()

                    metric = metric_sum / metric_count  # calculate mean Dice score of current epoch for validation set
                    metric_values.append(metric)
                    epoch_loss_val /= step  # calculate mean loss over current epoch

                    tb_writer.add_scalars("Loss Train/Val", {"train": epoch_loss, "val": epoch_loss_val}, epoch)
                    tb_writer.add_scalar("Dice Score Val", metric, epoch)
                    if metric > best_metric:  # if it's the best Dice score so far, proceed to save
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        # save the current best model weights
                        torch.save(model.state_dict(), os.path.join(self.model_path, "best_metric_model.pth"))
                        logger.info("saved new best metric model")
                    logger.info(
                        "current epoch {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                            epoch + 1, metric, best_metric, best_metric_epoch
                        )
                    )

            # learning rate update
            if (epoch + 1) % epochs_with_const_lr == 0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = param_group["lr"] / self.lr_divisor
                    logger.info(
                        "Dividing learning rate by {}. "
                        "New learning rate is: lr = {}".format(self.lr_divisor, param_group["lr"])
                    )

        logger.info("Train completed, best_metric: {:.4f}  at epoch: {}".format(best_metric, best_metric_epoch))
        torch.save(model.state_dict(), os.path.join(self.model_path, "last_epoch_model.pth"))
        logger.info(f'Saved model of the last epoch at: {os.path.join(self.model_path, "last_epoch_model.pth")}')
        return epoch_loss_values, metric_values

    def plot_loss_curve_and_mean_dice(self, epoch_loss_values, metric_values):
        # Plot the loss and metric
        plt.figure("train", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Epoch Average Loss")
        x = [i + 1 for i in range(len(epoch_loss_values))]
        y = epoch_loss_values
        plt.xlabel("epoch")
        plt.plot(x, y)
        plt.subplot(1, 2, 2)
        plt.title("Val Mean Dice")
        x = [self.val_interval * (i + 1) for i in range(len(metric_values))]
        y = metric_values
        plt.xlabel("epoch")
        plt.plot(x, y)
        plt.savefig(os.path.join(self.figures_path, "epoch_average_loss_and_val_mean_dice.png"))

    def load_trained_state_of_model(self, model):
        # load the trained model and set it into evaluation mode
        #model.load_state_dict(torch.load(os.path.join(self.model_path, "best_metric_model.pth")))
        model.load_state_dict(torch.load(os.path.join(self.model_path, "best_metric_model.pth")),strict=False)
        return model

    def new_load_trained_state_of_model(self, model,model_path):
        # load the trained model and set it into evaluation mode
        model.load_state_dict(torch.load(os.path.join("/projectnb/cs585bp/students/econlin/VS_Seg-master/MODEL/", model_path)))
        return model

    def get_safe_bounds(self, c, margin, axis):
        window_size = 96
        half_window = window_size // 2
    
        # Try to center the window around `c`
        start = max(0, c - half_window)
        end = start + window_size
    
        # If window goes beyond axis, shift it left
        if end > axis:
            end = axis
            start = max(0, end - window_size)
    
        # Final clamp to ensure window is within bounds and has correct size
        start = max(0, min(start, axis - window_size))
        end = start + window_size
    
        return start, end - 1  # Make 'end' inclusive as in the original code



    def extract_patch(self,volume, center, margin=48):
        """
        Extract a 96x96x96 patch centered at (x, y, z) from a 3D volume.
        Pads with zeros if the patch goes out of bounds.
        
        Args:
            volume (np.ndarray): 3D array with shape (Z, Y, X)
            center (tuple): (x, y, z) center coordinate
    
        Returns:
            np.ndarray: Patch of shape (96, 96, 96)
        """
        x_max,y_max,z_max = volume.shape
        
        x,y,z = center
    
        (x1,x2) = self.get_safe_bounds(x,margin,x_max)
        (y1,y2) = self.get_safe_bounds(y,margin,y_max)
        (z1,z2) = self.get_safe_bounds(z,margin,z_max)
    
        bounds = ((x1,x2),(y1,y2),(z1,z2))
        return (volume[y1:y2+1,x1:x2+1,z1:z2+1],bounds)

    def crop_t2_data(self,data_loader,data_bounds):
        print(data_bounds.items())
        for i, data in enumerate(data_loader):
            bounds,_ = data_bounds[i]

            ((x1,x2),(y1,y2),(z1,z2)) = bounds

            volume = np.asarray(data["image"].cpu().numpy()[0][0])

            print(bounds)
            print(volume.shape)
            cropped_volume = volume[y1:y2+1,x1:x2+1,z1:z2+1]

            print("crop")
            print(cropped_volume.shape)
            print(bounds)
    def new_run_inference(self,model,data_loader,folder,create_labels=True,label_prefix=""):
        # run inference and create figures in figures folder
        model.eval()  # activate evaluation mode of model
        
        model_segmentation = lambda *args, **kwargs: model(*args, **kwargs)[0]
        data_bounds = {}
        with torch.no_grad():  # turns off PyTorch's auto grad for better performance
            for i, data in enumerate(data_loader):
            
                outputs = sliding_window_inference(
                    inputs=data["image"].to(self.device),
                    roi_size=self.sliding_window_inferer_roi_size,
                    sw_batch_size=1,
                    predictor=model_segmentation,
                    mode="gaussian",
                )
            
                
                if isinstance(outputs, torch.Tensor):
                    patient_mask = outputs.cpu().numpy()[0]
                else:
                    patient_mask = outputs[0].cpu().numpy()
                label = np.asarray(np.squeeze(data['label'].numpy()))
                patient_mask = np.asarray(patient_mask)
                print(label.shape)
                print(f"patient mask: {patient_mask.shape}")
                c = patient_mask[0]
                c[c<0]=0
                max_slice_idx = np.argmax(np.sum(c,axis=(0,1)),axis=-1)
                max_slice = c[:,:,max_slice_idx]

                max_idx = np.argmax(max_slice)

                # Convert flat index to 2D coordinates (row, column)
                y, x = np.unravel_index(max_idx, max_slice.shape)
                
                patch,bounds = self.extract_patch(c,(x,y,max_slice_idx))
                label_patch,label_bounds = self.extract_patch(label,(x,y,max_slice_idx))
                data_bounds[i] = (bounds,label_bounds)

                filename = os.path.join(folder, f"{i}_{x}_{y}_{max_slice_idx}_mask.nii.gz")
                write_nifti(
                    data=patch,
                    file_name=filename,
                    resample=False)
                if create_labels:
                    filename = os.path.join(label_prefix+"_label_masks", f"{i}_{x}_{y}_{max_slice_idx}_mask.nii.gz")
                    write_nifti(
                        data=label_patch,
                        file_name=filename,
                        resample=False)
                        
        return data_bounds


    def run_inference(self, model, data_loader):
        logger = self.logger
        logger.info('Running inference...')

        model.eval()  # activate evaluation mode of model
        dice_scores = np.zeros(len(data_loader))

        if self.model == "UNet2d5_spvPA":
            model_segmentation = lambda *args, **kwargs: model(*args, **kwargs)[0]
        else:
            model_segmentation = model

        with torch.no_grad():  # turns off PyTorch's auto grad for better performance
            for i, data in enumerate(data_loader):
                logger.info("starting image {}".format(i))

                outputs = sliding_window_inference(
                    inputs=data["image"].to(self.device),
                    roi_size=self.sliding_window_inferer_roi_size,
                    sw_batch_size=1,
                    predictor=model_segmentation,
                    mode="gaussian",
                )

                dice_score = self.compute_dice_score(outputs, data["label"].to(self.device))
                dice_scores[i] = dice_score.item()

                logger.info(f"dice_score = {dice_score.item()}")

                # export to nifti
                if self.export_inferred_segmentations:
                    logger.info(f"export to nifti...")

                    nifti_data_matrix = np.squeeze(torch.argmax(outputs, dim=1, keepdim=True))[None, :]
                    data['label_meta_dict']['filename_or_obj'] = data['label_meta_dict']['filename_or_obj'][0]
                    data['label_meta_dict']['affine'] = np.squeeze(data['label_meta_dict']['affine'])
                    data['label_meta_dict']['original_affine'] = np.squeeze(data['label_meta_dict']['original_affine'])

                    folder_name = os.path.basename(os.path.dirname(data['label_meta_dict']['filename_or_obj']))
                    saver = NiftiSaver(
                        output_dir=os.path.join(self.results_folder_path, 'inferred_segmentations_nifti', folder_name),
                        output_postfix='')
                    saver.save(nifti_data_matrix, meta_data=data['label_meta_dict'])

                # plot centre of mass slice of label
                label = torch.squeeze(data["label"][0, 0, :, :, :])
                slice_idx = self.get_center_of_mass_slice(
                    label
                )  # choose slice of selected validation set image volume for the figure
                plt.figure("check", (18, 6))
                plt.clf()
                plt.subplot(1, 3, 1)
                plt.title("image " + str(i) + ", slice = " + str(slice_idx))
                plt.imshow(data["image"][0, 0, :, :, slice_idx], cmap="gray", interpolation="none")
                plt.subplot(1, 3, 2)
                plt.title("label " + str(i))
                plt.imshow(data["label"][0, 0, :, :, slice_idx], interpolation="none")
                plt.subplot(1, 3, 3)
                plt.title("output " + str(i) + f", dice = {dice_score.item():.4}")
                plt.imshow(torch.argmax(outputs, dim=1).detach().cpu()[0, :, :, slice_idx], interpolation="none")
                plt.savefig(os.path.join(self.figures_path, "best_model_output_val" + str(i) + ".png"))

        plt.figure("dice score histogram")
        plt.hist(dice_scores, bins=np.arange(0, 1.01, 0.01))
        plt.savefig(os.path.join(self.figures_path, "best_model_output_dice_score_histogram.png"))

        logger.info(f"all_dice_scores = {dice_scores}")
        logger.info(f"mean_dice_score = {dice_scores.mean()} +- {dice_scores.std()}")
