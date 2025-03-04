import torch
# import torch.nn.functional as F
from tqdm import tqdm
import clip
import matplotlib.pyplot as plt

import torch.nn.functional as Fnn      # For neural-net functionals (relu, conv, etc.)
import torchvision.transforms.functional as TF   # For image transformations (to_pil_image, etc.)


def orth_transforamtion_calculation(args, model, spurious_words):
    print("Orthogonalizing the embedding space w.r.t. {}".format(spurious_words))

    # Prepare spurious words and compute projection matrix
    spurious_tokens = clip.tokenize(spurious_words).to(args.device)
    with torch.no_grad():
        spurious_embeddings = model.encode_text(spurious_tokens)
        spurious_embeddings /= spurious_embeddings.norm(dim=-1, keepdim=True)

    # Compute projection matrix to remove spurious embeddings
    V = spurious_embeddings.T  # Shape: (embedding_dim, num_spurious_words)
    V = V.to(torch.float32)  # Ensure the dtype is float32 for inversion
    VtV = V.T @ V  # Shape: (num_spurious_words, num_spurious_words)
    VtV_inv = torch.inverse(VtV)  # Perform inversion in float32
    P = torch.eye(V.shape[0], device=args.device, dtype=torch.float32) - V @ VtV_inv @ V.T  # Projection matrix
    return P


def classify_images(args, model, text_embeddings, text_embeddings2, test_data_subset, test_loader, P=None, transformer = None, description="Classifying"):
    correct = 0
    total = 0
    misclassified_samples = []
    all_y = None
    all_preds = None
    all_metadata = None
    if transformer is not None:
        text_embeddings = transformer(text_embeddings.float())

    text_prompts = ["a photo of a CEO", "a photo of a nurse"]
    text_tokens = clip.tokenize(text_prompts).to(args.device)

    # Store image features and images
    image_features_list = []
    images_list = []
    metadata_list = []

    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711])

    def denormalize_clip(tensor):
        return tensor * clip_std.view(-1,1,1) + clip_mean.view(-1,1,1)
        
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).to(args.device)
        for images, labels, metadata in tqdm(test_loader, desc=description):
            images = images.to(args.device)
            labels = labels.to(args.device)

            # Encode images
            image_embeddings = model.encode_image(images)
            if transformer is not None:
                image_embeddings = transformer(image_embeddings.float())
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)

            # Ensure image_embeddings and text_embeddings have the same dtype
            image_embeddings = image_embeddings.to(text_embeddings.dtype)

            # Apply projection to image embeddings if P is not None
            if P is not None:
                P = P.to(image_embeddings.dtype)  # Ensure P is in the same dtype
                image_embeddings = image_embeddings @ P

            image_features_list.append(image_embeddings)

            # Compute cosine similarity
            similarity = image_embeddings @ text_embeddings.T

            # Predict the class with the highest similarity
            preds = similarity.argmax(dim=1)  

            # Record misclassified samples
            for i, (img, pred, label) in enumerate(zip(images, preds, labels)):
                if pred != label:
                    misclassified_samples.append((img.cpu(), label.cpu(), pred.cpu()))

            # Update correct predictions
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            if all_y is None:
                all_y = labels
                all_preds = preds
                all_metadata = metadata
            else:
                all_y = torch.cat((all_y, labels))
                all_preds = torch.cat((all_preds, preds))
                all_metadata = torch.cat((all_metadata, metadata))
          
        for i in range(2000):
            image, label, metadata = test_data_subset[i]
            img_denorm = denormalize_clip(image)    # remove your pil_to_tensor line
            img_denorm = img_denorm.clamp(0,1)
            img_pil = TF.to_pil_image(img_denorm)

            images_list.append(img_pil)
            metadata_list.append(metadata)

    image_features_all = torch.cat(image_features_list, dim=0)

    # Compute cosine similarities with text prompts
    similarities = image_features_all @ text_embeddings2.T    
    
    num_results = 5  # Display top 5 matches
    fig, axes = plt.subplots(2, num_results, figsize=(15, 6))

    for idx, prompt in enumerate(text_prompts):
        top_indices = similarity[:, idx].topk(num_results).indices.cpu().numpy()
        # print(similarities[:, idx])

        for j, image_idx in enumerate(top_indices):
            axes[idx, j].imshow(images_list[image_idx])
            axes[idx, j].axis("off")

        axes[idx, 0].set_title(prompt, fontsize=12, loc="left")
    
    plt.savefig('zeroB1.png')
    plt.show()

    # Debug accuracy calculation
    print(f"Total samples processed: {total}")
    print(f"Correct predictions: {correct}")
    accuracy = correct / total * 100
    return accuracy, misclassified_samples, [all_y, all_preds, all_metadata]

def accuracy_by_subgroup(pred, y, spurious):
    """
    pred: list or array of predicted labels (0 or 1)
    y:    list or array of true labels (0 or 1)
    spurious: list or array of spurious labels (0 or 1)

    Prints the accuracy for each of the 4 subgroup combinations:
        (y=0, s=0), (y=0, s=1), (y=1, s=0), (y=1, s=1)
    """
    # Ensure all lists are the same length
    assert len(pred) == len(y) == len(spurious), "All inputs must have the same length."

    # We'll store the correct counts and total counts for each subgroup
    counts_correct = {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}
    counts_total   = {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}
    
    # Go through each example
    for p, true_label, sp in zip(pred, y, spurious):
        # Subgroup key is (true_label, sp)
        key = (true_label, sp)
        counts_total[key] += 1
        if p == true_label:
            counts_correct[key] += 1
    
    # Calculate and print accuracy for each subgroup
    for subgroup in [(0,0), (0,1), (1,0), (1,1)]:
        corr = counts_correct[subgroup]
        tot = counts_total[subgroup]
        if tot > 0:
            acc = corr / tot
            print(f"Accuracy for y={subgroup[0]}, spurious={subgroup[1]}: {acc:.3f} "
                  f"({corr}/{tot} correct)")
        else:
            print(f"Accuracy for y={subgroup[0]}, spurious={subgroup[1]}: N/A (no samples)")

def print_per_class(args, eval_results):
    if args.dataset == 'celeba':
        print("Accuracy for subgroup")
        print(eval_results[0]['acc_y:notblond_male:0'])
        print(eval_results[0]['acc_y:notblond_male:1'])
        print(eval_results[0]['acc_y:blond_male:0'])
        print(eval_results[0]['acc_y:blond_male:1'])
    elif args.dataset == 'waterbirds':
        print("Accuracy for subgroup")
        print(eval_results[0]['acc_y:landbird_background:water'])
        print(eval_results[0]['acc_y:landbird_background:land'])
        print(eval_results[0]['acc_y:waterbird_background:water'])
        print(eval_results[0]['acc_y:waterbird_background:land'])



def contrastive_loss(embeddings, margin=0.5):
    """
    Compute contrastive loss using cosine similarity for given positive and negative pairs.
    
    Args:
        embeddings_neg1: First embedding in the first negative pair (shape: [batch, 1, 768]).
        embeddings_neg2: Second embedding in the first negative pair (shape: [batch, 1, 768]).
        embeddings_pos1: First embedding in the positive pair (shape: [batch, 1, 768]).
        embeddings_pos2: Second embedding in the positive pair (shape: [batch, 1, 768]).
        margin: Margin for dissimilarity (default: 0.5).
    
    Returns:
        loss: Contrastive loss value.
    """
    # Normalize embeddings along the last dimension
    embeddings_neg1 = Fnn.normalize(embeddings[0], dim=-1)
    embeddings_neg2 = Fnn.normalize(embeddings[1], dim=-1)
    embeddings_pos1 = Fnn.normalize(embeddings[2], dim=-1)
    embeddings_pos2 = Fnn.normalize(embeddings[3], dim=-1)

    # Compute cosine similarity
    sim_neg = Fnn.cosine_similarity(embeddings_neg1, embeddings_neg2, dim=-1)  # Shape: [batch, 1]
    sim_pos = Fnn.cosine_similarity(embeddings_pos1, embeddings_pos2, dim=-1)  # Shape: [batch, 1]

    # Loss components
    positive_loss = 1 - sim_pos  # Maximize similarity for positive pairs
    negative_loss = torch.relu(sim_neg - margin)  # Enforce margin for dissimilarity in negative pairs

    # Combine losses (mean across the batch)
    loss = positive_loss.mean() + negative_loss.mean()

    return loss


def sentence_list_to_embedding(args, model, sentences):
    st = [clip.tokenize(sentence).to(args.device) for sentence in sentences]
    se = [model.encode_text(token) for token in st]
    se = [embedding / embedding.norm(dim=-1, keepdim=True) for embedding in se]
    return torch.stack(se)

def train_transformation(args, model, textloader, transformer):
    optimizer = torch.optim.Adam(transformer.parameters(), lr=args.lr, weight_decay=args.wd)
    for epoch in range(args.epochs):
        total_loss = 0
        iter = 0

        for senetnces_list in tqdm(textloader):
            optimizer.zero_grad()

            embeddigs_list = [sentence_list_to_embedding(args, model, sentences) for sentences in senetnces_list]

            # froward pass
            transformed_embeddings = [transformer(embeddings.float()) for embeddings in embeddigs_list]
            
            # loss
            loss = contrastive_loss(transformed_embeddings)
            # loss = sim_loss(transformed_embeddings)

            # backward pass
            loss.backward()

            # update
            optimizer.step()
            total_loss += loss.item()
            iter += 1

        print(f"Epoch: {epoch}, Loss: {total_loss/iter}")