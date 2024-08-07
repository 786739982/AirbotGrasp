import cv2
import torch
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics.models.sam import Predictor as SAMPredictor
from torchvision.ops import masks_to_boxes


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    ax.imshow(mask_image)


def show_bbox(bbox, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    x1, y1 = bbox[0], bbox[1]
    x2, y2 = bbox[2], bbox[3]
    w = x2 - x1
    h = y2 - y1
    rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    plt.draw()


def show_points(coords, labels, ax, marker_size=25):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='o', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o', s=marker_size, edgecolor='white',
               linewidth=1.25)


def select_point(event):
    global predictor, img_raster, img_origin, input_point, input_label, masks, bboxes
    # Mouse click envet
    if event.button == 1: # click the left mouse
        print("[%s] Start Segmentation......" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        x, y = int(event.xdata), int(event.ydata)
        input_point.append([x, y])
        input_label.append(1)
        # segment anything
        masks, scores, logits = predictor.predict(
            point_coords=np.array(input_point),
            point_labels=np.array(input_label),
            multimask_output=True
        )
        bboxes = masks_to_boxes(torch.from_numpy(masks))
        bboxes = [box.numpy() for box in bboxes]

        # visualize object
        img_raster = np.zeros((img_origin.shape[0], img_origin.shape[1]))
        plt.clf()  # clean label image
        plt.imshow(img_origin)  # show origin image
        plt.axis('off')  # no image coordinate
        show_mask(masks[0], plt.gca())
        show_bbox(bboxes[0], plt.gca())
        img_raster[masks[0]] = 1
        show_points(np.array(input_point), np.array(input_label), plt.gca())
        print("[%s] Finish Segmentation......" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    if event.button == 3:  # click the right mouse
        print("[%s] Start Segmentation......" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        x, y = int(event.xdata), int(event.ydata)
        input_point.append([x, y])
        input_label.append(0)
        # segment anything
        masks, scores, logits = predictor.predict(
            point_coords=np.array(input_point),
            point_labels=np.array(input_label),
            multimask_output=True
        )
        bboxes = masks_to_boxes(torch.from_numpy(masks))
        bboxes = [box.numpy() for box in bboxes]
        # visualize object
        img_raster = np.zeros((img_origin.shape[0], img_origin.shape[1]))
        plt.clf()  # clean label image
        plt.imshow(img_origin)  # show origin image
        plt.axis('off')  # no image coordinate
        show_mask(masks[0], plt.gca())
        show_bbox(bboxes[0], plt.gca())
        img_raster[masks[0]] = 1
        show_points(np.array(input_point), np.array(input_label), plt.gca())
        print("[%s] Finish Segmentation......" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    
    if event.button == 2:  # click the mouse wheel     
        if len(input_point) > 1 and len(input_label) > 1:
            print("[%s]Clean Segmentation ......" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            input_point.pop()
            input_label.pop()
            # segment anything
            masks, scores, logits = predictor.predict(
                point_coords=np.array(input_point),
                point_labels=np.array(input_label),
                multimask_output=False
            )
            bboxes = masks_to_boxes(torch.from_numpy(masks))
            bboxes = [box.numpy() for box in bboxes]
            # visualize object
            img_raster = np.zeros((img_origin.shape[0], img_origin.shape[1]))
            plt.clf()  # clean label image
            plt.imshow(img_origin)  # show origin image
            plt.axis('off')  # no image coordinate
            show_mask(masks[0], plt.gca())
            show_bbox(bboxes[0], plt.gca())
            img_raster[masks[0]] = 1
            show_points(np.array(input_point), np.array(input_label), plt.gca())
            print("[%s] Finish Segmentation......" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        elif len(input_point) == 1 and len(input_label) == 1:
            print("[%s]Finish Segmentation ......" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            input_point.pop()
            input_label.pop()
            img_raster = np.zeros((img_origin.shape[0], img_origin.shape[1]))
            plt.clf()  # clean label image
            plt.imshow(img_origin)  # show origin image
            plt.axis('off')  # no image coordinate         
    plt.draw()  # update UI



if __name__ == "__main__":
    global predictor, img_raster, img_origin, input_point, input_label, masks, bboxes  # 声明全局变量
    input_point, input_label = [], []
    model_path = './checkpoint/sam_b.pt'
    image_path = 'GraspNet/doc/example_data/color.png'
    print("==Select Grasp Object==")
    print("[%s]正在读取图片......" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    img_origin = cv2.imread(image_path)  # 读取的图像以NumPy数组的形式存储在变量image中
    print("[%s]正在转换图片格式......" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    img_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)  # 将图像从BGR颜色空间转换为RGB颜色空间，还原图片色彩（图像处理库所认同的格式）
    print("[%s]正在初始化模型参数......" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    img_raster = np.zeros((img_origin.shape[0], img_origin.shape[1]))
    # 创建一个二维数组，用于保存掩膜做栅格转面
    fit = plt.figure(figsize=(10, 10))
    plt.imshow(img_origin)

    from segment_anything import sam_model_registry, SamPredictor
    sam_checkpoint = './checkpoint/sam_vit_b.pth'
    model_type = 'vit_b'
    device = 'cuda'
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(img_origin)


    # 调用`SamPredictor.set_image`来处理图像以产生一个图像嵌入。`SamPredictor`会记住这个嵌入，并将其用于随后的掩码预测
    print("【单点分割阶段】")
    print("[%s]正在分割图片......" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    fit.canvas.mpl_connect('button_press_event', select_point)
    # 窗口点击事件
    plt.axis('off')
    plt.show()
    print("【结果保存阶段】")
    print("[%s]正在保存分割数据......" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


