import torch
import torch.nn.functional as F
import random
import math

class RandomGrayscaleErasing(object):
    """ Randomly selects a rectangle region in an image and use grayscale image
        instead of its pixels.
        'Local Grayscale Transfomation' by Yunpeng Gong.
        See https://arxiv.org/pdf/2101.08533.pdf
    Args:
         probability: The probability that the Random Grayscale Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
    """

    def __init__(self, probability: float = 0.2, sl: float = 0.02, sh: float = 0.4, r1: float = 0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        """
        Args:
            img: after ToTensor() and Normalize([...]), img's type is Tensor
        """
        if random.uniform(0, 1) > self.probability:
            return img

        height, width = img.size()[-2], img.size()[-1]
        area = height * width

        for _ in range(100):

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)  # height / width

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < width and h < height:
                # tl
                x = random.randint(0, height - h)
                y = random.randint(0, width - w)
                # unbind channel dim
                r, g, b = img.unbind(dim=-3)
                # Weighted average method -> grayscale patch
                l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
                l_img = l_img.unsqueeze(dim=-3)  # rebind channel
                # erasing
                img[0, y:y + h, x:x + w] = l_img[0, y:y + h, x:x + w]
                img[1, y:y + h, x:x + w] = l_img[0, y:y + h, x:x + w]
                img[2, y:y + h, x:x + w] = l_img[0, y:y + h, x:x + w]

                return img

        return img


class DynamicColorAugmentationContrastiveLearning:
    def __init__(self, model, optimizer, p_0=0.2, alpha=0.1, tau=0.5, lambda_weight=0.5, initial_loss=1.0):
        self.model = model
        self.optimizer = optimizer
        self.p_0 = p_0
        self.p = p_0
        self.alpha = alpha
        self.tau = tau
        self.lambda_weight = lambda_weight
        self.loss_prev = initial_loss
        self.augmenter = RandomGrayscaleErasing(probability=self.p)

    def dynamic_color_augmentation(self, x):
        return self.augmenter(x)

    def contrastive_loss(self, z_i, z_j):
        batch_size = z_i.size(0)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        similarity_matrix = torch.matmul(z_i, z_j.T)
        positives = torch.diag(similarity_matrix)
        nominator = torch.exp(positives / self.tau)
        denominator = torch.sum(torch.exp(similarity_matrix / self.tau), dim=1)

        loss = -torch.log(nominator / denominator)
        return torch.mean(loss)

    def train_step(self, images, images_augmented, targets):
        # Zero the parameter gradients
        self.optimizer.zero_grad()

        # Forward pass
        features = self.model(images)
        augmented_features = self.model(images_augmented)

        # Calculate losses
        infoNCE_loss = self.contrastive_loss(features, features)
        color_invariance_loss = F.mse_loss(features, augmented_features)
        total_loss = infoNCE_loss + self.lambda_weight * color_invariance_loss

        # Backward pass and optimization
        total_loss.backward()
        self.optimizer.step()

        # Adjust color erasing probability
        loss_current = total_loss.item()
        delta_loss = (loss_current - self.loss_prev) / self.loss_prev
        self.p = self.p * (1 + self.alpha * delta_loss)
        self.loss_prev = loss_current
        self.augmenter.probability = self.p

        return total_loss.item()

# Example usage:
# model = YourModel()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# dynamic_learner = DynamicColorAugmentationContrastiveLearning(model, optimizer)

# for epoch in range(num_epochs):
#     for images, targets in dataloader:
#         images_augmented = dynamic_learner.dynamic_color_augmentation(images)
#         loss = dynamic_learner.train_step(images, images_augmented, targets)
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}")
