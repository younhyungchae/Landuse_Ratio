import numpy as np
import gradio as gr
from PIL import Image
from gradio_image_prompter import ImagePrompter

Image.MAX_IMAGE_PIXELS = None

CLASSES = [ 'Residential', 
            'Agricultural', 
            'Commercial', 
            'Factory', 
            'Mining', 
            'Power station', 
            'Transportation', 
            'Sports', 
            'Religious', 
            'Water', 
            'Others']

image = Image.open('../plots/HU_full.png')
#label = None
label = np.array(Image.open(f'../plots/ratio_label/HU/Agricultural_image.png').convert('RGBA'))
label = (label * 255).astype(np.uint8)
alpha = 100  # 투명도 값 (0 = 완전 투명, 255 = 완전 불투명)
label[..., 3] = alpha

def create_expanded_array(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    
    cropped = image[y_min:y_max, x_min:x_max]
    bbox_height, bbox_width = cropped.shape[:2]
    expanded_tiles = np.zeros((bbox_height * 256, bbox_width * 256, image.shape[2]), dtype=image.dtype)
    
    for i in range(bbox_height):
        for j in range(bbox_width):
            pixel_value = cropped[i, j]
            # 해당 픽셀을 256x256 영역에 복사
            expanded_tiles[i*256:(i+1)*256, j*256:(j+1)*256] = pixel_value
    
    return expanded_tiles

with gr.Blocks() as demo:
    current_country = gr.State()
    current_class = gr.State()
    current_label = gr.Image(visible=False)

    def get_map_image(prompts):
        bbox = [prompts['points'][-1][0], prompts['points'][-1][1], prompts['points'][-1][3], prompts['points'][-1][4]]
        #expanded = create_expanded_array(label, bbox)
        #cropped_image = base_image.crop([val*256 for val in bbox]).convert('RGBA')
        #expanded = Image.fromarray(expanded, mode="RGBA")
        #return Image.alpha_composite(cropped_image, expanded)
        return image.crop([val*256 for val in bbox])
    
    def update_image(current_country, current_class):
        return {'image':f'../plots/ratio_label/{current_country}/{current_class}_image.png', 'points':[]}

    with gr.Row():
        country_selector = gr.Dropdown(
            choices=["PL", "SK", "HU", "CZ"],
            label="국가 선택",
            value="None"  # 기본값
        )

        class_selector = gr.Dropdown(
            choices=CLASSES,
            label="Landuse 선택",
            value="None"  # 기본값
        )
    
    with gr.Row():
        update_button = gr.Button("이미지 업데이트")

    with gr.Row():
        image_prompter = ImagePrompter(show_label=False)
        gr.Interface(get_map_image,
                    image_prompter,
                    gr.Image(interactive=False))

    country_selector.change(lambda x:x, inputs=country_selector, outputs=current_country)
    class_selector.change(lambda x:x, inputs=class_selector, outputs=current_class)
    update_button.click(update_image, inputs=[current_country, current_class], outputs=image_prompter)
    
demo.launch(share=True)