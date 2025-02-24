import torch
import clip
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
from data.Waterbird import WaterbirdsDataset
from data.CelebA import CelebADataset
from utils import classify_images, accuracy_by_subgroup, print_per_class, orth_transforamtion_calculation, train_transformation
from data.SSL_text import get_SSL_dataset
from models import get_transformer
import wandb
from torch.utils.data import Subset
import torchvision.transforms.functional as F


def main(args):
    torch.manual_seed(args.seed)
    if args.wandb is not None:
        wandb.init(project='CLIPSB',name=args.wandb, config=args)

    #Load CLIP
    model, preprocess = clip.load(args.CLIP_model , device=args.device)


    # Load the full dataset, and download it if necessary
    if args.dataset == 'celeba':
        dataset = CelebADataset(download=True)
    elif args.dataset == 'waterbirds':
        dataset = WaterbirdsDataset(download=True)

    # Define preprocessing transformation
    transform = transforms.Compose([
        preprocess,  # Use CLIP's preprocess (resizes and normalizes the image)
    ])

    subset_indices = list(range(2000))

    # Manually assign the collate function from the original dataset
    



    # Get the training set
    test_data = dataset.get_subset(
        "test",
        transform=transform,
    )
    
    test_data_subset = Subset(test_data, subset_indices)
    test_data_subset.collate = test_data.collate
    # test_loader = get_train_loader("standard", test_data_subset, batch_size=args.batch_size, num_workers=0)

    # Prepare the standard data loader
    test_loader = get_train_loader("standard", test_data_subset, batch_size=args.batch_size, num_workers=0)

    if args.dataset == 'celeba':
        classes = ["a celebrity with dark hair", "a celebrity with blond hair"]
    elif args.dataset == 'waterbirds':
        classes = [ "Landbird", "Waterbird"]
    # Create prompts for classes
    def create_prompts(class_names):
        templates = [
            "a photo of a {}.",
            "a picture of a {}.",
        ]
        prompts = []
        for cls in class_names:
            for temp in templates:
                prompts.append(temp.format(cls))
        return prompts

    text_prompts = create_prompts(classes)
    print("The classification prompts are:", text_prompts)
    text_prompts2 = ["a photo of a CEO", "a photo of a nurse"]

    # Tokenize and encode text prompts
    text_tokens = clip.tokenize(text_prompts).to(args.device)
    text_tokens2 = clip.tokenize(text_prompts2).to(args.device)
    with torch.no_grad():
        text_embeddings = model.encode_text(text_tokens)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings2 = model.encode_text(text_tokens2)
        text_embeddings2 /= text_embeddings2.norm(dim=-1, keepdim=True)

    # Average embeddings per class
    text_embeddings = text_embeddings.view(len(classes), -1, text_embeddings.shape[-1])
    text_embeddings = text_embeddings.mean(dim=1)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings = text_embeddings.to(torch.float32)

    text_embeddings2 = text_embeddings2.view(len(classes), -1, text_embeddings2.shape[-1])
    text_embeddings2 = text_embeddings2.mean(dim=1)
    text_embeddings2 /= text_embeddings2.norm(dim=-1, keepdim=True)
    text_embeddings2 = text_embeddings2.to(torch.float32)

    transformer = None
    P = None

    if args.mitigation is not None:
        if args.mitigation == 'orth' or args.init_weight == 'orth':
            if args.dataset == 'celeba':
                spurious_words = ["Man", "Woman"]
            elif args.dataset == 'waterbirds':
                spurious_words = ["water", "land"]

            P = orth_transforamtion_calculation(args, model, spurious_words)

        if args.mitigation == 'train':
            text_loader = get_SSL_dataset(args)
            transformer = get_transformer(args)
            if args.init_weight == 'orth' and args.num_bases == 0:
                transformer.transformer.weight.data = P
            # transformer.load_state_dict(torch.load('transformer.pth'))
            train_transformation(args, model, text_loader, transformer)


    accuracy, misclassified_samples, [all_y, all_preds, all_metadata] = classify_images(args, model, text_embeddings, text_embeddings2, test_data_subset, test_loader, P=P, transformer=transformer, description="Classifying")


    eval_results = dataset.eval(all_preds.to('cpu'), all_y.to('cpu'), all_metadata.to('cpu'))
    # Assuming eval_results is a tuple with a dictionary and a string
    results_dict = eval_results[0]

    # Extract and print the desired metrics
    if args.dataset == 'celeba':
        adj_acc_avg = results_dict['acc_avg']
    elif args.dataset == 'waterbirds':
        adj_acc_avg = results_dict['adj_acc_avg']
    worst_group_acc = results_dict['acc_wg']

    print(f"Adjusted Average Accuracy: {adj_acc_avg:.3f}")
    print(f"Worst-Group Accuracy: {worst_group_acc:.3f}")
    if args.wandb is not None:
        wandb.log({"Adjusted Average Accuracy": adj_acc_avg, "Worst-Group Accuracy": worst_group_acc})

    if args.per_group:
        print_per_class(args, eval_results)
        accuracy_by_subgroup(list(all_preds.to('cpu').numpy()), list(all_y.to('cpu').numpy()), [x[0] for x in list(all_metadata.to('cpu').numpy())])


    # text_prompts = ["a photo of a CEO", "a photo of a nurse"]
    # text_tokens = clip.tokenize(text_prompts).to(args.device)

    # # Store image features and images
    # image_features_list = []
    # images_list = []
    # metadata_list = []

    # clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    # clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711])

    # def denormalize_clip(tensor):
    #     return tensor * clip_std.view(-1,1,1) + clip_mean.view(-1,1,1)


    # with torch.no_grad():
    #     text_features = model.encode_text(text_tokens).to(args.device)

    #     for i in range(len(test_data_subset)):
    #         image, label, metadata = test_data_subset[i]
    #     # for images, labels, metadata in tqdm(test_loader, desc="Classifying"):
    #     #     images = images.to(args.device)
    #     #     labels = labels.to(args.device)
    #         # Convert tensor to PIL Image
    #         # if isinstance(image, torch.Tensor):
    #         #     image = transforms.ToPILImage()(image)

    #         image_tensor = image.unsqueeze(0).to(args.device)
    #         image_features = model.encode_image(image_tensor)

    #         # Store features and images
    #         image_features_list.append(image_features)
    #         img_denorm = denormalize_clip(image)    # remove your pil_to_tensor line
    #         img_denorm = img_denorm.clamp(0,1)
    #         img_pil = F.to_pil_image(img_denorm)

    #         images_list.append(img_pil)
    #         metadata_list.append(metadata)

    # # Stack all image features
    # image_features_all = torch.cat(image_features_list, dim=0)

    # # Compute cosine similarities with text prompts
    # similarities = image_features_all @ text_features.T

    # # Retrieve top matching images for each prompt
    # num_results = 5  # Display top 5 matches
    # fig, axes = plt.subplots(2, num_results, figsize=(15, 6))

    # for idx, prompt in enumerate(text_prompts):
    #     top_indices = similarities[:, idx].topk(num_results).indices.cpu().numpy()
    #     # print(similarities[:, idx])

    #     for j, image_idx in enumerate(top_indices):
    #         axes[idx, j].imshow(images_list[image_idx])
    #         axes[idx, j].axis("off")

    #     axes[idx, 0].set_title(prompt, fontsize=12, loc="left")
    
    # plt.savefig('ourC1.png')
    # plt.show()


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument('--device', type=str.lower, default='cpu')
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--CLIP_model', type=str, default='ViT-L/14@336px',
                    help='CLIP model to use [ViT-B/32, RN50, RN101, RN50x4, ViT-B/16, ViT-L/14@224px, ViT-L/14@336px]')
    args.add_argument('--dataset', type=str.lower, default='waterbirds',
                        help='dataset to use [waterbirds, celeba]')
    args.add_argument('--epochs', type=int, default=1
                        , help='number of epochs to train the embedding transformer')
    args.add_argument('--per_group', type=bool, default=False
                        , help='whether to print accuracy per group')
    args.add_argument('--mitigation', type=str.lower, default=None
                        , help='What mitigation technique to use [None, orth, train]')
    args.add_argument('--num_bases', type=int, default=0
                        , help='Free transformation if zero otherwise number of bases for orthogonalization')
    args.add_argument('--wandb', type=str, default=None
                        , help='wandb run name')
    args.add_argument('--num_samples', type=int, default=100
                        , help='Number of samples to use for training the transformation')
    args.add_argument('--seed', type=int, default=42
                        , help='Random seed')
    args.add_argument('--lr', type=float, default=1e-1
                        , help='Learning rate for training the transformation')
    args.add_argument('--wd', type=float, default=0
                        , help='Weight decay for training the transformation')
    args.add_argument('--init_weight', type=str.lower, default='random'
                        , help='How to initialize the weights [random, orth]')

    args = args.parse_args()
    args.device = ['cuda' if torch.cuda.is_available() else 'cpu'][0]

    #print(args)
    print(args)
    print('*'*10)

    main(args)
