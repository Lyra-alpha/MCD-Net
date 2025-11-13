#----------------------------------------------------#
#   将单张图片预测和文件夹批量预测功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#----------------------------------------------------#
import os
import time
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image

from mcdnet_predictor import MCDNetPredictor

if __name__ == "__main__":
    #-------------------------------------------------------------------------#
    #   如果想要修改对应种类的颜色，到__init__函数里修改self.colors即可
    #-------------------------------------------------------------------------#
    mcdnet = MCDNetPredictor()
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹
    #----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"
    #-------------------------------------------------------------------------#
    #   count               指定了是否进行目标的像素点计数（即面积）与比例计算
    #   name_classes        区分的种类，和json_to_dataset里面的一样，用于打印种类和数量
    #
    #   count、name_classes仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    count           = False
    name_classes    = ["background","Moraine"]
    #-------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #   
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "dataset/Moraine_dataset/JPEGImages"
    dir_save_path   = "img_out/"

    if mode == "predict":
        '''
        predict模式有几个注意点
        1、该代码无法直接进行批量预测，如果想要批量预测，请使用dir_predict模式。
        2、如果想要保存，利用r_image.save("img.jpg")即可保存。
        3、如果想要原图和分割图不混合，可以把mix_type参数设置成1。
        4、如果想根据mask获取对应的区域，可以参考detect_image函数中，利用预测结果绘图的部分。
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = mcdnet.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()
                
                # 询问是否保存图片
                save = input('Save image? (y/n): ')
                if save.lower() == 'y':
                    save_path = input('Enter save path (default: output.jpg): ') or 'output.jpg'
                    r_image.save(save_path)
                    print(f'Image saved to: {save_path}')

    elif mode == "dir_predict":
        '''
        dir_predict模式用于批量预测文件夹中的所有图片
        '''
        import os
        from tqdm import tqdm

        # 创建保存目录
        if not os.path.exists(dir_save_path):
            os.makedirs(dir_save_path)

        # 获取图片文件列表
        img_names = [name for name in os.listdir(dir_origin_path) 
                    if name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))]
        
        if not img_names:
            print(f"No images found in {dir_origin_path}")
            exit()

        print(f"Found {len(img_names)} images in {dir_origin_path}")
        print(f"Output directory: {dir_save_path}")

        # 批量处理图片
        for img_name in tqdm(img_names, desc="Processing images"):
            image_path = os.path.join(dir_origin_path, img_name)
            try:
                image = Image.open(image_path)
                r_image = mcdnet.detect_image(image)
                
                # 保存结果图片
                output_path = os.path.join(dir_save_path, img_name)
                r_image.save(output_path)
                
            except Exception as e:
                print(f"Error processing {img_name}: {str(e)}")
                continue

        print(f"All images processed. Results saved to: {dir_save_path}")

    else:
        raise AssertionError("Please specify the correct mode: 'predict' or 'dir_predict'.")