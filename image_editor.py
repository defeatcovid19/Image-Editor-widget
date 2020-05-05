import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import ipywidgets as widgets
from IPython.display import display,clear_output

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    '''change contrast and brightness'''

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def image_editor_widget(input_folder, output_folder, color_map = None):
    ''' for color_map use any of matplotlib.cm '''
    
    # load images list
    image_name_list = sorted([i for i in os.listdir(input_folder) if not i.endswith('ipynb_checkpoints')])
    image_list = {i: plt.imread(os.path.join(input_folder, i)) for i in image_name_list}
    shape_list = {k: v.shape for k, v in image_list.items()}

    # define widget elements
    select_image = widgets.Select(
        options=image_name_list,
        rows=min(20,len(image_name_list)),
        description='Image:'
    )
    select_image.layout.width='450px'
    invert_color = widgets.Checkbox(
        value=False,
        description='Invert color'
    )
    resize_x = widgets.IntSlider(
        value=shape_list[image_name_list[0]][1],
        min=0,
        max=shape_list[image_name_list[0]][1],
        step=1,
        description=' ' # if description changes, change update_current_size accordingly
    )
    original_size = widgets.Label(
        value = 'Original: ' + str(shape_list[image_name_list[0]][1]) + ' x ' + str(shape_list[image_name_list[0]][0])
    )
    current_size = widgets.Label(
        value = 'Current: ' + str(shape_list[image_name_list[0]][1]) + ' x ' + str(shape_list[image_name_list[0]][0])
    )
    change_bright = widgets.IntSlider(
        value=0,
        min=-255,
        max=255,
        step=1,
        description='Brightness:'
    )
    change_contrast = widgets.IntSlider(
        value=0,
        min=-255,
        max=255,
        step=1,
        description='Contrast:'
    )
    crop_x=widgets.IntRangeSlider(
        value=[0, 10000],
        min=0,
        max=shape_list[image_name_list[0]][1],
        step=1,
        description='Crop X:'
    )
    crop_y=widgets.IntRangeSlider(
        value=[0, 10000],
        min=0,
        max=shape_list[image_name_list[0]][0],
        step=1,
        description='Crop Y:'
    )
    save_name=widgets.Text(
        value=image_name_list[0],
        placeholder='Type something',
        description='Save name:',
        disabled=False   
    )
    save_name.layout.width='450px'
    save_button = widgets.Button(description='Save')
    def save_image(b):
        im_path = os.path.join(output_folder, save_name.value)
        cv2.imwrite(im_path, im)
        with out:
            print('Saved in:', im_path)
    save_button.on_click(save_image)
    reset_button = widgets.Button(description='Reset')
    def reset_param(b):
        change_contrast.value = 0
        change_bright.value = 0
        crop_x.value = (0, im_orig.shape[1])
        crop_y.value = (0, im_orig.shape[0])
        resize_x.value = im_orig.shape[1]
        save_name.value = select_image.value
        current_size.value = 'Current: ' + str(im.shape[1]) + ' x ' + str(im.shape[0])
    reset_button.on_click(reset_param)
    def update_crop(change):
        crop_x.max = shape_list[change.new][1]
        crop_x.value = (0, shape_list[change.new][1])
        crop_y.max = shape_list[change.new][0]
        crop_y.value = (0, shape_list[change.new][0])
    def update_resize(change):
        resize_x.max = shape_list[select_image.value][1]#im.shape[1]#shape_list[change.new][1]
        resize_x.value = shape_list[select_image.value][1]#im.shape[1]#shape_list[change.new][1]
    def update_resize_after_crop(change):
        resize_x.max = crop_x.value[1] - crop_x.value[0]
        resize_x.value = crop_x.value[1] - crop_x.value[0]
    def update_label(change):
        save_name.value = select_image.value
        current_size.value = 'Current: ' + str(im_orig.shape[1]) + ' x ' + str(im_orig.shape[0])
        original_size.value = 'Original: ' + str(im_orig.shape[1]) + ' x ' + str(im_orig.shape[0])
    select_image.observe(update_crop, 'value')
    select_image.observe(update_resize, 'value')   
    select_image.observe(update_label, 'value')
    crop_x.observe(update_resize_after_crop, 'value')
    crop_y.observe(update_resize_after_crop, 'value')
    def update_current_size(change):
        x_size = (crop_x.value[1] - crop_x.value[0])
        y_size = (crop_y.value[1] - crop_y.value[0])
        if (change.owner.description == ' '):  # resize is applied
            x_size = change.new
            ratio = x_size / y_size
            y_size = ratio * x_size
        current_size.value = 'Current: ' + str(x_size) + ' x ' + str(round(y_size))
    resize_x.observe(update_current_size, 'value')
    crop_x.observe(update_current_size, 'value')
    crop_y.observe(update_current_size, 'value')

    def interactive_function(select_image, invert_color, resize_x, change_bright, change_contrast, crop_x, crop_y):
        global im_orig, im
        
        print('Showing image:', select_image)
        im_orig = image_list[select_image]
        # RGBA to RGB (4th channel for png is Alpha)
        if len(im_orig.shape) > 2 and im_orig.shape[2] == 4:
            im_orig = cv2.cvtColor(im_orig, cv2.COLOR_BGRA2BGR)
        # normalize input to 0-255
        im_orig = cv2.normalize(im_orig, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype('uint8')
        # invert color
        if invert_color:
            im_orig = 255 - im_orig
        # change brightness and contrast
        im = apply_brightness_contrast(im_orig, brightness = change_bright, contrast = change_contrast)
        # crop
        im_crop = im[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]
        # resize
        ratio = im_crop.shape[0] / im_crop.shape[1]  # y / x
        im = cv2.resize(im_crop,
                        dsize=(resize_x, round(resize_x * ratio)), # size is (x, y)
                        interpolation=cv2.INTER_NEAREST)
        # plot image
        plt.figure(figsize = (7, 7))
        plt.imshow(im, cmap=color_map, vmin=0, vmax=255)

    out = widgets.interactive_output(interactive_function, {'select_image': select_image, 'invert_color': invert_color,
                                                            'resize_x': resize_x, 'crop_x': crop_x, 'crop_y': crop_y,
                                                            'change_bright': change_bright, 'change_contrast': change_contrast})
    # place widget elements
    final_widget = widgets.HBox([widgets.VBox([select_image,
                                               change_bright,
                                               change_contrast,
                                               invert_color,
                                               crop_x,
                                               crop_y,
                                               widgets.HTML('<p>Resize <b>x-axis</b> (px), keep aspect ratio:</p>'),
                                               original_size,
                                               current_size,
                                               resize_x,
                                               reset_button,
                                               save_name,
                                               save_button]), out])

    return(final_widget)
