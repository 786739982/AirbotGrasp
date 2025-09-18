import sys
import numpy as np
import cv2
from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel, QWidget
from PyQt5.QtGui import QPixmap, QMouseEvent, QImage
import torch
from torchvision.ops import masks_to_boxes

app = QApplication(sys.argv)  # 在模块级别创建 QApplication 实例

from AirbotSegment import AirbotSegment


class AirbotInterface(QWidget):

    def __init__(self, type_predictor='SAM'):
        super().__init__()
        self.label = QLabel(self)
        self.init_ui()
        self.type_predictor = type_predictor
        self.init_predictor()
        self.image = None
        self.input_point = []
        self.input_label = []
        self.bbox = None
        self.mask = np.zeros(shape=[720, 1280], dtype=bool)

    def init_ui(self):
        self.setWindowTitle('Image Display')
        self.label.setAlignment(Qt.AlignCenter)
        self.resize(1280, 720)

    def init_predictor(self):
        if self.type_predictor == 'SAM':
            Predictor = AirbotSegment()
            self.predictor = Predictor.get_model()
        elif self.type_predictor == 'Yolo-World': # You can use your own predictor model
            pass

    def update_image(self, image):
        # Get the shape of the image
        height, width, channel = image.shape
        # Create QImage from the RGB image
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        # Convert QImage to QPixmap
        pixmap = QPixmap.fromImage(q_image)
        # Set the pixmap to the label
        self.label.setPixmap(pixmap)
        self.label.setAlignment(Qt.AlignCenter)
        # Resize the window to fit the image
        self.resize(pixmap.width() + 20, pixmap.height() + 20)
    
    def mousePressEvent(self, event: QMouseEvent):
        if self.label.underMouse():
            pos = event.pos()
            global_pos = self.mapToGlobal(pos)
            label_pos = self.label.mapFromGlobal(global_pos)

            if event.button() == Qt.LeftButton:
                print(f"Left Clicked at ({label_pos.x()}, {label_pos.y()})")
                self.input_point.append([label_pos.x(), label_pos.y()])
                self.input_label.append(1)
                cv2.circle(self.image, (label_pos.x(), label_pos.y()), 4, (255,0,0), -1)
                self.update_image(self.image)

                print("[%s] Start Segmentation......" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                
                # segment anything
                masks, scores, logits = self.predictor.predict(
                    point_coords=np.array(self.input_point),
                    point_labels=np.array(self.input_label),
                    multimask_output=False
                )
                bboxes = masks_to_boxes(torch.from_numpy(masks))
                bbox = bboxes[0]
                top_left = int(bbox[0]), int(bbox[1])
                right_bottom = int(bbox[2]), int(bbox[3])
                image_copy = self.image.copy()
                cv2.rectangle(image_copy, top_left, right_bottom, (0,0,255), 2)
                image_copy[masks[0]] = [191, 214, 238]

                self.mask = masks[0]
                self.bbox = bbox
                self.update_image(image_copy)

                # print("[%s] Finish Segmentation......" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            elif event.button() == Qt.RightButton:
                print(f"Right Clicked at ({label_pos.x()}, {label_pos.y()})")
                self.input_point.append([label_pos.x(), label_pos.y()])
                self.input_label.append(0)
                
    def segment(self, image, window_title='Image Display', save=True):
        self.input_point = []
        self.input_label = []
        self.bbox = []
        self.image = image
        self.predictor.set_image(image)

        self.update_image(image)
        self.setWindowTitle(window_title)
        self.show()
        app.exec_()
        if save:
            cv2.imwrite('mask.png', self.mask*255)
        print('Show over')
        return self.mask, self.bbox


# if not hasattr(sys.modules[__name__], "window"):

#     window = AirbotInterface()

# def segment(image, window_title='Image Display', save=True):
#     global window
#     window.input_point = []
#     window.input_label = []
#     window.bbox = []
#     window.image = image
#     window.predictor.set_image(image)
    
#     window.update_image(image)
#     window.setWindowTitle(window_title)
#     window.show()
#     app.exec_()
#     if save:
#         cv2.imwrite('mask.png', window.mask*255)
#     print('Show over')
#     return window.mask, window.bbox



if __name__ == '__main__':
    # 图片路径列表
    images = ['color.png', 'depth.png']
    interface = AirbotInterface()
    for image_path in images:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        interface.segment(image)