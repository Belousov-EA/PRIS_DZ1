import streamlit as st
import torch
from torchvision import transforms, models
import torchvision.io

CLASSES = ['Это утка', "Это гипопотам", "Крыса!!!"]

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device('cpu')

MODEL_PATH = 'res_net.pth'

st.title('ПРИС ДЗ1')
st.header('ИУ5-23 Белоусов')

image = st.file_uploader('Pick an image')

if(image):
    st.image(image)

    with open('image/some/image.jpg', 'wb') as file:
        file.write(image.getvalue())

    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, 3, bias=True)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    val_dataset = torchvision.datasets.ImageFolder('image', val_transform)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False)
    result_label = -1
    for inputs, label in val_dataloader:
        preds = model(inputs)
        result_label = preds.argmax(dim=1)
    if result_label != -1:
        st.header(CLASSES[result_label])