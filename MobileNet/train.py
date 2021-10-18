import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, utils
from tqdm import tqdm
from mobilenet import MobileNet
from model_v2 import MobileNetV2
from model_v3 import mobilenet_v3_large


def main():
    #识别是否有可用的GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 数据预处理
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    #data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    #image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    #assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    #train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
    #                                     transform=data_transform["train"])
    #train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    #flower_list = train_dataset.class_to_idx
    #cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    #json_str = json.dumps(cla_dict, indent=4)
    #with open('class_indices.json', 'w') as json_file:
    #    json_file.write(json_str)

    #nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    #print('Using {} dataloader workers every process'.format(nw))

    #train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                           batch_size=batch_size, shuffle=True,
    #                                           num_workers=nw)

    #validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
    #                                        transform=data_transform["val"])
    #val_num = len(validate_dataset)
    #validate_loader = torch.utils.data.DataLoader(validate_dataset,
    #                                              batch_size=batch_size, shuffle=False,
    #                                              num_workers=nw)

    #print("using {} images for training, {} images for validation.".format(train_num,
    #                                                                       val_num))


    #调用torchvision的dataset中自带的cifar10数据集
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=80, shuffle=True, num_workers=12)
    val_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transform["val"])
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=80, shuffle=False, num_workers=0)
    val_num = len(val_set)


    # create model
    net = mobilenet_v3_large(num_classes=10)

    #如果需要使用预训练模型采用下面的代码
    # load pretrain weights
    # download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    #model_weight_path = "./mobilenet_v2.pth"
    #assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    #pre_weights = torch.load(model_weight_path, map_location=device)
    # delete classifier weights
    #pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    #missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

    # freeze features weights
    #for param in net.features.parameters():
        #param.requires_grad = False


    #从头开始训练，定义训练相关参数
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    # construct an optimizer
    #params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    epochs = 150
    #model_name=MobileNetV2
    save_path = './MobileNetV3.pth'
    train_steps = len(train_loader)
    best_acc=0.0
    best_ephoch=0
    for epoch in range(epochs):
        # 训练
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        #测试
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        # 当新的测试精度大于原来的时候，保存新的模型
        if val_accurate > best_acc:
            best_acc = val_accurate
            best_ephoch = epoch + 1
            torch.save(net.state_dict(), save_path)
        print('best_accuracy:{:.3f}'.format(best_acc))
        print('best_epoch: {}'.format(best_ephoch))

    #结果输出
    print('best_accuracy:{:.3f}'.format(best_acc))
    print('best_epoch: {}'.format(best_ephoch))
    print('Finished Training')


if __name__ == '__main__':
    main()
