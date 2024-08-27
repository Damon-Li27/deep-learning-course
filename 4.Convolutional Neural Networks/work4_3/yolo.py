# OpenCV是一个跨平台的计算机视觉库，主要用于图片和视频的处理
import cv2
import numpy as np

'''
    读取YOLOv2的权重文件（yolov2.weights）和配置文件（yolov2.cfg），进行目标检测的基本流程。
    包括加载模型、预处理图像、前向推理、后处理（如NMS）和显示结果。
'''

# 加载 YOLOv2 模型的配置文件和权重文件
# cv2.dnn.readNet()是 OpenCV 中用于读取深度学习网络模型的函数。
net = cv2.dnn.readNet('model_data/yolov2.weights', 'model_data/yolov2.cfg')

# 读取 COCO 数据集的类别名称（80类）
with open('model_data/coco_classes.txt', 'r') as f:
    # line.strip() 对于文件中的每一行，去除行两端的空白字符（如空格、换行符等）。
    # if line.strip()：这是一个过滤条件，只有当去除空白字符后的行不为空时，才会将该行包含在最终的列表中。
    # classes变量保存一个类别列表，如['person', 'bicycle',...]
    classes = [line.strip() for line in f.readlines() if line.strip()]


# 读取输入图像-测试图片
image = cv2.imread('images/test1.jpg')
# 获取输入图像的尺寸，用于后续边界框转换。
height, width, _ = image.shape

# blobFromImage将图片调整为 YOLO 所需的输入格式：缩放系数0.00392（1/255），将像素值归一化到[0,1]。
# 输入尺寸 (416, 416)是YOLOv2的默认输入大小。(0, 0, 0)是均值，默认减去均值进行归一化。True表示交换通道顺序，将BGR转为RGB。
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#将输入数据设置为网络输入
net.setInput(blob)

# 获取yolo网络所有层名称，返回一个包含所有层名称的列表
layer_names = net.getLayerNames()
# net.getUnconnectedOutLayers()返回一个未连接的输出层的缩引（通常YOLO的检测层）。
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 前向传播，通过 YOLO 网络获取预测结果。outs包含每个输出层的检测结果。
outs = net.forward(output_layers)

# 解析YOLO的输出，提取边界框、置信度和类
class_ids = []
confidences = []
boxes = []
# 遍历每个输出层，并解析每个检测到的目标
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        # 设定置信度阈值，保留大于0.5的检测结果
        if confidence > 0.5:
            # 计算边界框的实际位置和尺寸
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            # 保存所有检测到的边界框、置信度和类别
            confidences.append(float(confidence))
            class_ids.append(class_id)

# 使用非极大值抑制去除多余的框，0.5是置信度阈值，0.4是NMS阈值
# cv2.dnn.NMSBoxes是 OpenCV 中用于执行非极大值抑制（Non-Maximum Suppression，NMS）操作的函数。
# 参数：bboxes：一个包含检测框坐标的列表或数组。通常每个检测框由四个值表示，如 [x1, y1, x2, y2]，分别表示检测框的左上角坐标和右下角坐标。
# scores：一个与检测框对应的置信度分数列表或数组。  score_threshold：置信度分数的阈值。  nms_threshold：非极大值抑制的阈值。
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# 绘制剩余的有效边界框和标签
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f'{label} {int(confidence * 100)}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 显示输出图像，直到用户按下任意键关闭窗口
cv2.imshow("YOLOv2 Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()