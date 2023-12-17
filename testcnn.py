import cv2
import os
import numpy as np

# Specify the path to the video file
video_file_path = os.listdir('baseline')[0] #CHANGE THIS PATH IF YOU WANT TO USE A DIFERENT VIDEO

# Specify the output directory for images
output_directory = "outputfolder"


def split(video_path, output_directory):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Read the frames from the video
    success, frame = video_capture.read()
    count = 0

    # Loop through all frames in the video
    while success:
        # Save the frame as an image
        image_path = os.path.join(output_directory, f"frame_{count:04d}.jpg")
        cv2.imwrite(image_path, frame)

        # Read the next frame
        success, frame = video_capture.read()
        count += 1

    # Release the video capture object
    video_capture.release()



from keras.models import load_model  # TensorFlow is required for Keras to work


def classify(image_directory):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # CAMERA can be 0 or 1 based on default camera of your computer
    IsCount = 0
    IsNotCount = 0
    count = 0

    for i in os.listdir(image_directory):
        # Grab the webcamera's image.
        image = cv2.imread(output_directory+"/"+i)

        # Resize the raw image into (224-height,224-width) pixels
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        # Show the image in a window


        # Make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image = (image / 127.5) - 1

        # Predicts the model
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
       
        if class_name == class_names[1]:
            IsCount += 1
        elif class_name == class_names[0]:
            IsNotCount += 1
    
    percentageIs = (round((IsCount/(IsNotCount + IsCount)) * 100,1))
    percentageIsNot = (round((IsNotCount/(IsNotCount + IsCount)) * 100,1))

        
    print("I am ",percentageIs," percent sure this video is Ai-Generated and ",percentageIsNot," percent sure it is not.")
    
        



        # Print prediction and confidence score
       # print("Class:", class_name[2:], end="")
        #print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")






if __name__ == "__main__":
    dir = os.listdir(output_directory) 
    if len(dir) == 0: 
        split(video_file_path, output_directory)
        classify(output_directory)
        
    else: 
        print("Not empty directory")
        
    
    print("DONE")

    # Call the function to split the video into images
    
