import os
import random

import torch
import torch.nn.functional as F
import torch.utils.data as data
import PIL.Image as Image
import numpy as np
import json

from torchvision import transforms
from torchvision.datasets import ImageFolder


def PilLoaderRGB(imgPath):
    return Image.open(imgPath).convert('RGB')


class EpisodeDataset(data.Dataset):
    """
    Dataloader to sample a task/episode.
    In case of 5-way 1-shot: nSupport = 1, nCls = 5.

    :param string imgDir: image directory, each category is in a sub file;
    :param int nCls: number of classes in each episode;
    :param int nSupport: number of support examples;
    :param int nQuery: number of query examples;
    :param transform: image transformation/data augmentation;
    :param int inputW: input image size, dimension W;
    :param int inputH: input image size, dimension H;
    """

    def __init__(self, imgDir, nCls, nSupport, nQuery, transform, inputW, inputH, nEpisode=2000,
                 model=None, pre_select_classes=False):
        super().__init__()

        self.imgDir = imgDir
        self.clsList = os.listdir(imgDir)
        self.prototype_dict = {class_: None for class_ in self.clsList}
        self.nCls = nCls
        self.nSupport = nSupport
        self.nQuery = nQuery
        self.transform = transform
        self.nEpisode = nEpisode

        floatType = torch.FloatTensor

        self.tensorSupport = floatType(nCls * nSupport, 3, inputW, inputH)
        self.labelSupport = F.one_hot(torch.repeat_interleave(torch.arange(0, nCls), nSupport, dim=0))

        self.labelQuery = F.one_hot(torch.repeat_interleave(torch.arange(0, nCls), nQuery, dim=0))
        # Add zeros to define classes that don't belong to any labels
        if len(self.clsList) < nCls * 2:
            non_labels = torch.zeros(nQuery * (len(self.clsList) - nCls), nCls)
        else:
            non_labels = torch.zeros(nQuery * nCls, nCls)
        self.labelQuery = torch.concat((self.labelQuery, non_labels), dim=0)
        self.tensorQuery = floatType(self.labelQuery.shape[0], 3, inputW, inputH)

        self.imgTensor = floatType(3, inputW, inputH)

        self.pre_select_classes = pre_select_classes
        if self.pre_select_classes and model is not None:
            self.model = model
        elif self.pre_select_classes and model is None:
            raise Exception("Cannot pre-select classes without model")

    def update_prototypes(self, class_names, prototypes):
        class_names = np.asarray(class_names).transpose()
        for class_list, prototype_list in zip(class_names, prototypes):
            for class_, prototype in zip(class_list, prototype_list):
                self.prototype_dict[class_] = prototype

    def __len__(self):
        return self.nEpisode

    def __getitem__(self, idx):
        """
        Return an episode

        :return dict: {'SupportTensor': 1 x nSupport x 3 x H x W,
                       'SupportLabel': 1 x nSupport,
                       'QueryTensor': 1 x nQuery x 3 x H x W,
                       'QueryLabel': 1 x nQuery}
        """
        # select nCls from clsList
        temp_prototype_dict = self.prototype_dict.copy()
        episode_class = random.choice(list(temp_prototype_dict.keys()))
        episode_prototype = temp_prototype_dict.pop(episode_class)
        episode_classes = None
        if episode_prototype is not None and self.pre_select_classes is True:
            temp_prototype_dict_1 = {k: v for k, v in temp_prototype_dict.items() if v is not None}
            if len(temp_prototype_dict_1) == (len(self.prototype_dict) - 1):
                # Only compare prototypes when prototypes are calculated
                other_prototypes_tensor = torch.empty(size=(len(temp_prototype_dict), episode_prototype.shape[0]),
                                                      dtype=torch.float, device='cuda:0', requires_grad=False)
                for i, key in enumerate(temp_prototype_dict.keys()):
                    other_prototypes_tensor[i] = temp_prototype_dict[key]

                # Give input a batch size of 1. Cos Classifier expects batched input
                B = 1
                episode_prototype = episode_prototype.view(B, 1, episode_prototype.shape[0])
                other_prototypes_tensor = other_prototypes_tensor.view(B, other_prototypes_tensor.shape[0],
                                                                       other_prototypes_tensor.shape[1])

                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        class_similarity = self.model.cos_classifier(episode_prototype, other_prototypes_tensor)
                        _, top_indices = torch.topk(class_similarity.flatten(),
                                                    int(self.labelQuery.shape[0] / self.nQuery) - 1)

                episode_classes = [episode_class] + [list(temp_prototype_dict.keys())[idx] for idx in
                                                     top_indices.tolist()]
        if episode_classes is None:
            episode_classes = random.sample(list(temp_prototype_dict.keys()),
                                            int((self.labelQuery.shape[0] / self.nQuery)) - 1)
            episode_classes.append(episode_class)

        # Add one extra because you want a query class that's not in the support list
        for i, cls in enumerate(episode_classes):
            clsPath = os.path.join(self.imgDir, cls)
            imgList = os.listdir(clsPath)

            # in total nQuery+nSupport images from each class
            imgCls = np.random.choice(imgList, self.nQuery + self.nSupport, replace=False)

            if i < self.nCls:
                for j in range(self.nSupport):
                    img = imgCls[j]
                    imgPath = os.path.join(clsPath, img)
                    I = PilLoaderRGB(imgPath)
                    self.tensorSupport[i * self.nSupport + j] = self.imgTensor.copy_(self.transform(I))

            for j in range(self.nQuery):
                img = imgCls[j + self.nSupport]
                imgPath = os.path.join(clsPath, img)
                I = PilLoaderRGB(imgPath)
                self.tensorQuery[i * self.nQuery + j] = self.imgTensor.copy_(self.transform(I))

        ## Random permutation. Though this is not necessary in our approach
        permSupport = torch.randperm(self.nCls * self.nSupport)
        permQuery = torch.randperm((self.nCls + 1) * self.nQuery)

        return (self.tensorSupport[permSupport],
                self.labelSupport[permSupport],
                self.tensorQuery[permQuery],
                self.labelQuery[permQuery],
                episode_classes)


class EpisodeJSONDataset(data.Dataset):
    """
    To make validation results comparable, we fix 1000 episodes for validation. Clear.

    :param string episodeJson: ./data/Dataset/val1000Episode_K_way_N_shot.json
    :param string imgDir: image directory, each category is in a sub file;
    :param int inputW: input image size, dimension W;
    :param int inputH: input image size, dimension H;
    :param valTransform: image transformation/data augmentation;
    """

    def __init__(self, episodeJson, imgDir, inputW, inputH, valTransform):
        with open(episodeJson, 'r') as f:
            self.episodeInfo = json.load(f)
        # print(self.episodeInfo)

        self.imgDir = imgDir
        self.nEpisode = len(self.episodeInfo)
        self.nCls = len(self.episodeInfo[0]['Support'])
        self.nSupport = len(self.episodeInfo[0]['Support'][0])
        self.nQuery = len(self.episodeInfo[0]['Query'][0])
        self.transform = valTransform

        floatType = torch.FloatTensor
        intType = torch.LongTensor

        self.tensorSupport = floatType(self.nCls * self.nSupport, 3, inputW, inputH)
        self.labelSupport = intType(self.nCls * self.nSupport)
        self.tensorQuery = floatType(self.nCls * self.nQuery, 3, inputW, inputH)
        self.labelQuery = intType(self.nCls * self.nQuery)

        self.imgTensor = floatType(3, inputW, inputH)
        for i in range(self.nCls):
            self.labelSupport[i * self.nSupport: (i + 1) * self.nSupport] = i
            self.labelQuery[i * self.nQuery: (i + 1) * self.nQuery] = i

    def __getitem__(self, index):
        """
        Return an episode

        :param int index: index of data example
        :return dict: {'SupportTensor': 1 x nSupport x 3 x H x W,
                       'SupportLabel': 1 x nSupport,
                       'QueryTensor': 1 x nQuery x 3 x H x W,
                       'QueryLabel': 1 x nQuery}
        """
        for i in range(self.nCls):
            for j in range(self.nSupport):
                imgPath = os.path.join(self.imgDir, self.episodeInfo[index]['Support'][i][j])
                I = PilLoaderRGB(imgPath)
                self.tensorSupport[i * self.nSupport + j] = self.imgTensor.copy_(self.transform(I))

            for j in range(self.nQuery):
                imgPath = os.path.join(self.imgDir, self.episodeInfo[index]['Query'][i][j])
                I = PilLoaderRGB(imgPath)
                self.tensorQuery[i * self.nQuery + j] = self.imgTensor.copy_(self.transform(I))

        return (self.tensorSupport,
                self.labelSupport,
                self.tensorQuery,
                self.labelQuery)

    def __len__(self):
        """
        Number of episodes
        """
        return self.nEpisode


def ValLoader(episodeJson, imgDir, inputW, inputH, valTransform):
    dataloader = data.DataLoader(ValImageFolder(episodeJson, imgDir, inputW, inputH, valTransform),
                                 shuffle=False)
    return dataloader


def TrainLoader(batchSize, imgDir, trainTransform):
    dataloader = data.DataLoader(ImageFolder(imgDir, trainTransform),
                                 batch_size=batchSize, shuffle=True, drop_last=True)
    return dataloader


if __name__ == '__main__':
    import torchvision.transforms as transforms

    mean = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
    std = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
    normalize = transforms.Normalize(mean=mean, std=std)
    trainTransform = transforms.Compose([
        transforms.RandomCrop(32, padding=8),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x),
        transforms.ToTensor(),
        normalize
    ])

    TrainEpisodeSampler = EpisodeDataset(imgDir='../data/cifar-fs/train/',
                                         nCls=5,
                                         nSupport=5,
                                         nQuery=15,
                                         transform=trainTransform,
                                         inputW=32,
                                         inputH=32)
    data = TrainEpisodeSampler[0]
    print(data[1])
