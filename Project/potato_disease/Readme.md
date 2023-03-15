# 1. Deep learning project - potato disease classification

Issues: potato disease classification - train with plant village dataset (leaf)  

Potato disease classification project: Healthy, early blight, late blight

1. data cleaning & preprocessing: tf dataset , Data Augmentation(more training sample)
2. Model Building: CNN 
3. Export the train model on to PC
    1. tf serving: http:localhost: 8501 - FastAPI: [http://localhost:8000](http://localhost:8000) → Website - React JS: drop the image → this will label them
    2. Quantization → Tf lite model → google cloud functions(GCP) → mobile application (React Native)

