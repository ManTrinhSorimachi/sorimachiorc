import numpy as np
import cv2
from scipy import ndimage
from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt 
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from paddleocr import PPStructure
import base64
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
import math
import os
import datetime
import uvicorn
from fastapi.templating import Jinja2Templates

def processing_image(img):
  # load image
  # img = cv2.imread(img)
  # convert to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  factor = 3
  img = Image.fromarray(img)
  enhancer = ImageEnhance.Sharpness(img).enhance(factor)
  if gray.std() < 30:
     enhancer = ImageEnhance.Contrast(enhancer).enhance(factor)
  enhanced = np.array(enhancer)
  # blur
  blur = cv2.GaussianBlur(enhanced, (35,35), sigmaX=33, sigmaY=33)
  # divide
  divide = cv2.divide(enhanced, blur, scale=255)
  # otsu threshold
  divide = cv2.cvtColor(divide, cv2.COLOR_BGR2GRAY)
  # thresh = cv2.adaptiveThreshold(divide, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=201, C=20)
  thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
  # apply morphology
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
  morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel,iterations=1)
  # display the result
  return morph
def rotate_image(img):
    # img_before = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = 255 - img_bin

    gradient = cv2.morphologyEx(img_bin, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))
    # Apply dilation
    dilation = cv2.dilate(gradient, np.ones((8,8), np.uint8), iterations=1)
    erosion = cv2.erode(dilation, np.ones((8,8), np.uint8), iterations=1)
    lines = cv2.HoughLinesP(erosion, rho=1, theta=np.pi/180, threshold=255, minLineLength=180, maxLineGap=8)

    angles = []

    for [[x1, y1, x2, y2]] in lines:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    median_angle = np.median(angles)
    img_rotated = ndimage.rotate(img, median_angle)
    
    return img_rotated
class SortBoxes(object):
    def __init__(self, result_PPStructure):
        self.results = []
        self.__sort__(result_PPStructure)

    def __sort__(self, result):
        results = []
        for region in result:
            if region['res']['boxes'] is None:
                continue

            result_len = len(region['res']['boxes'])
            for i in range(result_len):
                ocrResult = OcrResult()
                bbox = region['res']['boxes'][i]
                ocrResult.boxes = [bbox[0], bbox[1], bbox[2], bbox[1], bbox[2], bbox[3], bbox[0], bbox[3]]
                results.append(ocrResult)

        results = sorted(results, key=lambda x: (x.boxes[1], x.boxes[0]))

        tmpRow = []
        for i in range(0, len(results)):
            if tmpRow:
                if (results[i].boxes[1] - tmpRow[-1].boxes[1] < max(results[i].boxes[5] - results[i].boxes[1],
                                                                    tmpRow[-1].boxes[5] - tmpRow[-1].boxes[1]) / 2):
                    tmpRow.append(results[i])
                else:
                    self.results.extend(sorted(tmpRow, key=lambda x: (x.boxes[0])))
                    tmpRow = []
                    tmpRow.append(results[i])
            else:
                tmpRow.append(results[i])

        if tmpRow:
            self.results.extend(sorted(tmpRow, key=lambda x: (x.boxes[0])))


class OcrResult(object):
    def __init__(self):
        self.boxes = []
app = FastAPI()

# Load VietOCR model
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'data\\transformerocr10.pth'
config['cnn']['pretrained'] = False
config['device'] = 'cpu'
predictor = Predictor(config)

# Load PaddleOCR model
table_engine = PPStructure(use_pdserving=False, use_gpu=True, lang='en', layout=False, show_log=False)

@app.route('/api', methods=['POST'])
async def ocr_api(request: Request):
     # Load image
    image_file = await request.form()
    image = cv2.imdecode(np.frombuffer(await image_file["file"].read(), np.uint8), cv2.IMREAD_COLOR)
    # processed = processing_image(image)
    rotate = rotate_image(image)
    
    import time
    time_old = time.time()
    result = table_engine(rotate, return_ocr_result_in_table=True)
    print('boxes detect in', time.time() - time_old)
    
    sortBoxes = SortBoxes(result)

    time_old = time.time()
    # Extract text from OCR results
    texts = []
    cropped_images = []
    for box in sortBoxes.results:
        # Convert box coordinates to integers
        box = list(map(int, box.boxes))

        # Crop image to box region
        cropped_img = rotate[box[1]:box[5], box[0]:box[2]]
        
        # Convert cropped image to PIL Image
        crop_process = processing_image(cropped_img)
        pil_img = Image.fromarray(cv2.cvtColor(crop_process, cv2.COLOR_BGR2RGB))
        
        # Run VietOCR on cropped image
        text = predictor.predict(pil_img)
        
        # Convert cropped image to base64 string
        base64_img = base64.b64encode(cv2.imencode('.png', crop_process)[1]).decode()
        
        # Append cropped image and text to lists
        cropped_images.append(crop_process)
        texts.append(text)

    # Convert the images to base64 strings for the JSON response
    data = []
    for i, (text, cropped_img) in enumerate(zip(texts, cropped_images)):
        base64_img = base64.b64encode(cv2.imencode('.png', cropped_img)[1]).decode()
        data.append({'text': text, 'image': base64_img})
    
    print('texts detect in', time.time() - time_old)
    # Return the extracted text and images as a JSON response
    return JSONResponse({'data': data})


@app.route('/post_data', methods=['POST'])
async def ocr_api(request: Request):
    form_data = await request.json()
    time_now = datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
    # Tạo folder
    if os.path.exists('outputs/' + time_now):
        return JSONResponse('folder đã tồn tại')
    
    os.makedirs('outputs/' + time_now)
    
    lines = []
    for i, obj in enumerate(form_data):
        text = obj.get('text')
        img_b64 = obj.get('img')
        img = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_GRAYSCALE)
        
        path_img = f'outputs/{time_now}/{time_now}-{i}.png'
        filename = os.path.basename(path_img)
        lines.append(filename + ' ' + text)
        
        cv2.imwrite(path_img, img)
    open(f'outputs/{time_now}/labels.txt', 'a', encoding='utf-8').write('\n'.join(lines))
        
    
    return JSONResponse(f'Đã lưu {len(lines)} data')

# import shutil
# # Đường dẫn đến thư mục chứa các thư mục con chứa file ảnh và file txt
# root_dir = 'outputs'

# # Tạo thư mục mới để lưu trữ tất cả các file ảnh
# image_dir = 'all_images'
# os.makedirs(image_dir, exist_ok=True)

# # Di chuyển tất cả các file ảnh đến thư mục mới
# for dirpath, dirnames, filenames in os.walk(root_dir):
#     for filename in filenames:
#         if filename.endswith('.png'):
#             shutil.move(os.path.join(dirpath, filename), os.path.join(image_dir, filename))

# output_file = 'all_labels.txt'
# all_content = ''

# for dirpath, dirnames, filenames in os.walk('outputs'):
#     for filename in filenames:
#         if filename.endswith('.txt'):
#             filepath = os.path.join(dirpath, filename)
#             # Extract the image name from the file path
#             img_name = os.path.splitext(os.path.basename(filename))[0]
#             with open(filepath, 'r', encoding='utf-8') as f:
#                 for line in f.readlines():
#                     image_file = line.split()[0]  # Get the image file name from the line
#                     all_content += f"InkData_line_processed/{image_file}\t{line[len(image_file)+1:].strip()}\n"

# with open(output_file, 'w', encoding='utf-8') as f:
#     f.write(all_content.rstrip())

### Render web ###
# Khai báo thư mục templates (chứa file html)
templates = Jinja2Templates(directory="templates")

# Trang chủ
@app.route("/")
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})