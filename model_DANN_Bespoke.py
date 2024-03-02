import torch.nn as nn
import torch
from torchvision.models import resnet18, resnet50, vgg16_bn, ResNet50_Weights, ResNet18_Weights,VGG16_BN_Weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
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

class SelfAttention(nn.Module):
    def __init__(self, in_features):
        super(SelfAttention, self).__init__()

        # Linear layers for query, key, and value
        self.query = nn.Linear(in_features, in_features)
        self.key = nn.Linear(in_features, in_features)
        self.value = nn.Linear(in_features, in_features)

    def forward(self, x):
        # Apply linear transformations to obtain query, key, and value
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float32))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute the weighted sum using attention weights
        attended_values = torch.matmul(attention_weights, value)

        return attended_values




class DANNBespoke(nn.Module):
    def __init__(self, num_classes, 
                 backbone = 'resnet50', 
                 attention = False,
                 dropout = 0.1) -> None:
        super().__init__()

        # resnet50_base_pretrained = resnet50(weights = ResNet50_Weights.DEFAULT)
        # self.feature_extrator = nn.Sequential(*list(resnet50_base_pretrained.children())[:-2])
        #---------------------Feature Extractor Network------------------------#
        
        if backbone == 'resnet18':
            self.feature_extractor = resnet18(weights = ResNet18_Weights.DEFAULT)
        elif backbone =='vgg16':
            self.feature_extractor = vgg16_bn(weights = VGG16_BN_Weights.DEFAULT)
        elif backbone =="resnet50":
            self.feature_extractor = resnet50(weights = ResNet50_Weights.DEFAULT)
        else:
            self.a = mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT)

            # Recursively replace all convolutional layers with dropout-convolutional sequence
            self._replace_conv_layers(self.a, dropout_prob = dropout)

            self.feature_extractor = torch.nn.Sequential(*(list(self.a.children())[:-1]))


            # Add a Global Average Pooling (GAP) layer
            self.feature_extractor.add_module('GlobalAvgPool', torch.nn.AdaptiveAvgPool2d(1))

            # Print the modified MobileNetV2 architecture
            # print(self.feature_extractor)



        
        print(f'feature extractor backbone created using {backbone} model')

        self.attention = attention
        if self.attention:
            print(" Attention is being used for this model")
        
        # try:
        #     self.feature_extractor = dict_pretrained_models[backbone]
        # except:
        #     print(f'using default pretrained, problem loading {backbone}')
        #     self.feature_extractor = resnet50(weights = ResNet50_Weights.DEFAULT)

        #---------------------Class (Task) Classifier Network------------------------#
        
        class_classifier_layers = [nn.ReLU(),
                                   nn.Dropout(p=dropout),
                                   nn.Linear(1280,100),
                                #    nn.Linear(1000,100),
                                   nn.ReLU(),
                                   nn.Dropout(p=dropout)]
        if self.attention:
            class_classifier_layers.append(SelfAttention(in_features=100))
        
        class_classifier_layers.extend([nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(100,num_classes)])
        
        self.class_classifier = nn.Sequential(*class_classifier_layers)
            
        # self.class_classifier = nn.Sequential(nn.ReLU(),
        #                                 nn.Dropout(p=dropout),
        #                                 nn.Linear(1000,100),
        #                                 # nn.BatchNorm1d(100), # added batch norm to improve accuracy
        #                                 nn.ReLU(),
        #                                 nn.Dropout(p=dropout),
        #                                 nn.Linear(100,num_classes))
        
        

        #---------------------Domain Classifier Network------------------------#
        self.domain_classifier = nn.Sequential(nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(1280,100),
                                        # nn.Linear(1000,100),
                                        # nn.BatchNorm1d(100), # added batch norm to improve accuracy
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(100,2))
        
        return

    def _replace_conv_layers(self, module, dropout_prob):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(module, name, nn.Sequential(
                    child,
                    nn.Dropout2d(dropout_prob)
                ))
            else:
                self._replace_conv_layers(child, dropout_prob) # recursion

    def forward(self, input_data, alpha = 0.0):
        features = self.feature_extractor(input_data)
        # print("features.shape......", features.shape)
        features = features.view(-1,features.size(1))
        # print("features.shape after view......", features.shape)

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

    model = DANNBespoke(num_classes=7,backbone='mobilenet',attention=False).to(device)
    print(model)
    summary(model, input_size=(3,224,224))

    # sfew = DatasetSFEW()
    # sfew_train_loader, _ = sfew.get_dataloader()
    # batch = next(iter(sfew_train_loader)) # image, label, image_name
    # imgs, labels = batch[0], batch[1] # not using image name
    # print(sfew.labels)
    # print(imgs.shape,labels.shape)

    # x_labels, x_domains, x_features = model(imgs.to(device))
    # print(x_labels.shape, x_domains.shape, x_features.shape)
    # print("input_labels\n", labels)
    # print("x_output_labels\n", x_labels)
    # print("x_output_domains\n",x_domains)
    
    ## For 5 return outputs
        
    sfew = DatasetSFEW()
    sfew_train_loader, _ = sfew.get_dataloader()
    batch = next(iter(sfew_train_loader)) # image, label, image_name
    imgs, labels = batch[0], batch[1] # not using image name
    print(sfew.labels)
    print(imgs.shape,labels.shape)

    # x_labels, x_domains, x_features, log_softmax_domains, softmax_domains = model(imgs.to(device))
    # print(x_labels.shape, x_domains.shape, x_features.shape,log_softmax_domains.shape, softmax_domains.shape)
    # print("input_labels\n", labels)
    # print("x_output_labels\n", x_labels)
    # print("x_output_domains\n",x_domains)
    # print("log_softmax_domains\n", log_softmax_domains)
    # print("softmax_domains\n", softmax_domains)

    # linear_output = torch.randn(1, 100)  # Assuming output from a linear layer with 100 units
    # self_attention = SelfAttention(in_features=100)

    # output = self_attention(linear_output)
    # print(output.shape)

