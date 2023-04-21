import os
from PIL import Image

from mmocr.apis import TextRecInferencer
inferencer = TextRecInferencer(model='SATRN', weights=r'C:\Users\User\mmocr\best_IC15_recog_word_acc_epoch_77.pth')
avg_val = []

# Set directory path where the images are located
directory = r'C:\Users\User\mmocr\val'

# Loop through all files in the directory
for filename in os.listdir(directory):
    print()
    file = filename[:-4]
    print(file)
    parts = file.split('_')[2:]
    # print(parts)
    new_string = '_'.join(parts)
    # print(new_string)
    original = new_string.replace(" ","")
    # print(original)

    filepath = os.path.join(directory, filename)
    # print(filepath)
    image = Image.open(filepath)
    width, height = image.size
    new_width = 640  # Replace with desired width
    new_height = int(height / (width / new_width))
    image = image.resize((new_width, new_height))

    # print(filepath)
    result = inferencer(image, print_result=True)
    compres = result['predictions'][0]['text']
    
    max_len = max(len(original), len(compres))
    matching_chars = sum(1 for a, b in zip(original, compres) if a == b)
    percentage_matching = (matching_chars / max_len) * 100
    percentage_matching = round(percentage_matching,1)
    avg_val.append(percentage_matching)
    print('Original: ',original)
    print('Prediction: ',compres)
    print('Accuracy: ',percentage_matching)

average = sum(avg_val) / len(avg_val)
average = round(average,1)
print('Average Accuracy of model: ', average)