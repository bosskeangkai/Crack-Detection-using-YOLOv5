import streamlit as st
import torch
import torchvision.transforms as T 

# config model files
cfg_model_path="models/best.pt"

MODEL_PATH='best.pt'
LABELS_PATH='model_class.txt'

# import image to local storage and displaying it
def load_image():
    uploaded_file=st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data=uploaded_file.getvalue()
        st.image(image_data)

# Loading model 
def load_model(model_path):
    model=torch.load(model_path, map_location='cpu')
    model.eval()
    return model

# loading Label
def load_labels(labels_file):
    with open(labels_file,"r") as f:
        category=[s.strip() for s in f.readlines()]
        return category

# Predicting model 
def predict(model, category, image):
    preprocess=T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    input_tensor=preprocess(image)
    input_batch=input_tensor.unsqueeze(0)

    with torch.no_grad():
        output=model(input_batch)

    probabilities=torch.nn.functional.softmax(output[0], dim=0)

    all_prob, all_catid=torch.topk(probabilities, len(category))
    for i in range(all_prob.size(0)):
        st.write(category[all_catid[i]], all_prob[i].item())

# main function 
def main():
    st.title('Upload Crack Surface Image')
    model=load_model(MODEL_PATH)
    category=load_labels(LABELS_PATH)
    image=load_image()
    result=st.button('Run on image')
    if result:
        st.write('Calculating results...')
        predict(model, category, image)

if __name__ == '__main__':
    main()  
