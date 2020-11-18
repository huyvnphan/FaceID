import matplotlib.pyplot as plt
import torch


def to_numpy(x):
    mean = torch.tensor([0.5, 0.5, 0.5, 0.5]).view(1, 4, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5, 0.5]).view(1, 4, 1, 1)
    # Scale back to range [0, 1]
    x = (x * std) + mean
    x = x.squeeze(0).permute(1, 2, 0)
    return x.numpy()


def display(faceid_model, x0, x1):

    faceid_model.eval()
    with torch.no_grad():
        embed_x0 = faceid_model(x0)
        embed_x1 = faceid_model(x1)

    cosine = round(torch.nn.CosineSimilarity()(embed_x0, embed_x1).item(), 4)

    x0 = to_numpy(x0)
    x1 = to_numpy(x1)

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(10, 10)

    axs[0, 0].imshow(x0[:, :, :3])
    axs[0, 0].set_title("X0 - RGB")
    axs[1, 0].imshow(x0[:, :, 3], cmap="RdYlBu")
    axs[1, 0].set_title("X0 - Depth")

    axs[0, 1].imshow(x1[:, :, :3])
    axs[0, 1].set_title("X1 - RGB \n Cosine to X0: " + str(cosine), color="b")
    axs[1, 1].imshow(x1[:, :, 3], cmap="RdYlBu")
    axs[1, 1].set_title("X1 - Depth")

    for i in range(2):
        for j in range(2):
            axs[i, j].axis("off")

    plt.show()
