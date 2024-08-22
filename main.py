import streamlit as st

import tensorflow as tf

import numpy as np

import time
#tensorflow model prediction

def model_prediction(test_image):
    model=tf.keras.models.load_model("trained_model1.keras")

    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(225,225))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])  #convert single image to a batch
    print(input_arr.shape)

    prediction=model.predict(input_arr)
    result_index=np.argmax(prediction)
    return result_index


#sidebar
st.sidebar.title("Dashboard")
app_mode=st.sidebar.selectbox("select page",["Home","About","Tumor Recognition"])


#home page
if(app_mode=="Home"):
    st.header("AMRITAhealth")
    st.subheader("BRAIN TUMOR RECOGNITION SYSTEM")

    image_path="brain.png"
    st.image(image_path,use_column_width=True)
    st.markdown("""
 Welcome to the Brain Tumor Recognition System! 
    
    Our mission is to help in identifying brain tumors efficiently. Upload an image of a brain mri scan, and our system will analyze it to detect any signs of diseases. 

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a mri scan with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential tumors.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Brain Tumor Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.


  """)
    
#about page

elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
#### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.
                

                """)
    
#prediction  page

elif(app_mode=="Tumor Recognition"):
    st.header("Tumor Recognition")
    test_image=st.file_uploader("Choose an Image:")

    if(st.button("show image")):
        st.image(test_image,use_column_width=True)
    #predict button
    if(st.button("Predict")):
        with st.spinner("please wait..."):
            time.sleep(5)
            st.write("Our Prediction is")
            result_index=model_prediction(test_image)
            
            class_name=['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
            st.success("model prediction is {}".format(class_name[result_index]))


       
