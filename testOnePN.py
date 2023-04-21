import os
from PIL import Image

from mmocr.apis import TextRecInferencer
inferencer = TextRecInferencer(model='SATRN', weights=r'C:\Users\User\mmocr\best_IC15_recog_word_acc_epoch_77.pth')
avg_val = []

# Set directory path where the images are located
directory = r'C:\Users\User\mmocr'

filename = r"C:\Users\User\mmocr\iMAGE.jpg"

print()

filepath = os.path.join(directory, filename)

# Open image file and resize it to a smaller size
# image = Image.open(filepath)
# width, height = image.size
# new_width = 640  # Replace with desired width
# new_height = int(height / (width / new_width))
# image = image.resize((new_width, new_height))

# result = inferencer(image, print_result=True)
# compres = result['predictions'][0]['text']
 
result = inferencer(filepath,print_result=True)
compres = result['predictions'][0]['text']

print('Prediction: ',compres)








# img = cv2.imread(r"C:\Users\User\mmocr\iMAGE.jpg")
# img_resized = cv2.resize(img, (width, height))
# img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
# # img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# # Pass preprocessed image to OCR model
# result = inferencer(img_gray, print_result=True)
# text = result['predictions'][0]['text']

# # Print OCR results
# print(text)
