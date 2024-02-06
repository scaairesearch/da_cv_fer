import torch.nn as nn
import torch
from torchvision.models import resnet18, resnet50, vgg16_bn, ResNet50_Weights, ResNet18_Weights,VGG16_BN_Weights
from torch.autograd import Function
import torch.nn.functional as F

# dict_pretrained_models = {
#     'resnet18': resnet18(pretrained=True,progress=True),
#     'resnet50': resnet50(weights = ResNet50_Weights.DEFAULT),
#     'vgg16': vgg16_bn(pretrained=True,progress=True)
#     }

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class DANN(nn.Module):
    def __init__(self, num_classes, backbone = 'resnet50', dropout = 0.1) -> None:
        super().__init__()

        # resnet50_base_pretrained = resnet50(weights = ResNet50_Weights.DEFAULT)
        # self.feature_extrator = nn.Sequential(*list(resnet50_base_pretrained.children())[:-2])
        #---------------------Feature Extractor Network------------------------#
        
        if backbone == 'resnet18':
            self.feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif backbone =='vgg16':
            self.feature_extractor = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        else:
            self.feature_extractor = resnet50(weights = ResNet50_Weights.DEFAULT)
        
        print(f'feature extractor backbone created using {backbone} model')
        
        # try:
        #     self.feature_extractor = dict_pretrained_models[backbone]
        # except:
        #     print(f'using default pretrained, problem loading {backbone}')
        #     self.feature_extractor = resnet50(weights = ResNet50_Weights.DEFAULT)

        #---------------------Class Classifier Network------------------------#
        self.class_classifier = nn.Sequential(nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(1000,100),
                                        nn.BatchNorm1d(100), # added batch norm to improve accuracy
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(100,num_classes))

        #---------------------Label Classifier Network------------------------#
        self.domain_classifier = nn.Sequential(nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(1000,100),
                                        nn.BatchNorm1d(100), # added batch norm to improve accuracy
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(100,2))
        
        return


    def forward(self, input_data, alpha = 0.0):
        features = self.feature_extractor(input_data)

        reverse_features = GradientReversalFn.apply(features,alpha)

        class_output = self.class_classifier(features)
        domain_output = self.domain_classifier(reverse_features)

        return class_output, domain_output, features
        # return class_output, domain_output, features, F.log_softmax(domain_output,dim=-1), F.softmax(domain_output,dim=-1)




if __name__ == '__main__':
    from torchsummary import summary  #for model summary and params
    from ds_sfew import DatasetSFEW


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    model = DANN(num_classes=7,backbone='vgg16').to(device)
    print(model)
    summary(model, input_size=(3,224,224))

    sfew = DatasetSFEW()
    sfew_train_loader, _ = sfew.get_dataloader()
    batch = next(iter(sfew_train_loader)) # image, label, image_name
    imgs, labels = batch[0], batch[1] # not using image name
    print(sfew.labels)
    print(imgs.shape,labels.shape)

    x_labels, x_domains, x_features = model(imgs.to(device))
    print(x_labels.shape, x_domains.shape, x_features.shape)
    print("input_labels\n", labels)
    print("x_output_labels\n", x_labels)
    print("x_output_domains\n",x_domains)
    
    ## For 5 return outputs
        
    # sfew = DatasetSFEW()
    # sfew_train_loader, _ = sfew.get_dataloader()
    # batch = next(iter(sfew_train_loader)) # image, label, image_name
    # imgs, labels = batch[0], batch[1] # not using image name
    # print(sfew.labels)
    # print(imgs.shape,labels.shape)

    # x_labels, x_domains, x_features, log_softmax_domains, softmax_domains = model(imgs.to(device))
    # print(x_labels.shape, x_domains.shape, x_features.shape,log_softmax_domains.shape, softmax_domains.shape)
    # print("input_labels\n", labels)
    # print("x_output_labels\n", x_labels)
    # print("x_output_domains\n",x_domains)
    # print("log_softmax_domains\n", log_softmax_domains)
    # print("softmax_domains\n", softmax_domains)
