from PIL import Image 
import os 

def Create(input_path, output_path, gif_name):
    image_folder = input_path 
    output_gif = output_path+gif_name+'.gif' 

    images = [] 

    for filename in sorted(os.listdir(image_folder)): 
        if filename.endswith('.png') or filename.endswith('.png'): 
            file_path = os.path.join(image_folder, filename) 
            img = Image.open(file_path) 
            images.append(img) 
        
    if images: 
        images[0].save(output_gif, save_all=True, append_images=images[1:], duration=200, loop=0) 