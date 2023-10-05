from model.similarity_network import SimilarityNetwork
from model.feature_extractor import FeatureExtractor

dependencies = ["torch", "torchvision", "einops"]


def resnet50_LiMAC(dims=512):
    return FeatureExtractor['RESNET'].get_model(dims)


def s2vs_dns():
    return SimilarityNetwork['ViSiL'].get_model(pretrained='s2vs_dns')


def s2vs_vcdb():
    return SimilarityNetwork['ViSiL'].get_model(pretrained='s2vs_vcdb')
