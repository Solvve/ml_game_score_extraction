import cv2
import albumentations as albu
import numpy as np

def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
    ]
    
    return albu.Compose(_transform)

def preprocess_image(image, preprocess_input):
    return get_preprocessing(preprocess_input)(image=image)['image']

def preprocess_mask(mask, threshold=0.3):
    return (mask > threshold).astype(np.uint8)

def approx_polygon(image):
    contours,hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0: return []
    
    countour = contours[0]
    # find contour with the biggest area
    if len(contours) > 1:
        max_area = cv2.contourArea(countour)
        for c in contours[1:]:
            area = cv2.contourArea(c)
            if area > max_area:
                max_area = area
                countour = c
                
    hull = cv2.convexHull(countour, clockwise=True, returnPoints=True)
    hull_ = np.array(hull).reshape(-1,2)
    corners = np.zeros((4, 2), dtype = "int32")
    s = hull_.sum(axis = 1)
    corners[0] = hull_[np.argmin(s)]
    corners[2] = hull_[np.argmax(s)]

    diff = np.diff(hull_, axis = 1)
    corners[1] = hull_[np.argmin(diff)]
    corners[3] = hull_[np.argmax(diff)]

    return corners

def four_point_transform(image, pts):
    ordered_points = pts.astype("float32")
    (tl, tr, br, bl) = ordered_points
    
    widthTop = np.linalg.norm(tr-tl)
    widthBottom = np.linalg.norm(br-bl)
    maxWidth = max(int(widthTop), int(widthBottom))

    heightLeft = np.linalg.norm(tl-bl)
    heightRight = np.linalg.norm(tr-br)
    maxHeight = max(int(heightLeft), int(heightRight))

    w = maxWidth
    h = maxHeight
    dst = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]], dtype = "float32")
    
    M = cv2.getPerspectiveTransform(ordered_points, dst)
    warped = cv2.warpPerspective(image, M, (w, h))

    return warped

def get_min_rectangle(mask):
    coords = np.where(mask > 0)
    coords = np.vstack((coords[1], coords[0]))
    coords = coords.transpose(1,0)

    rect = cv2.minAreaRect(coords)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return rect, box

def crop_segment(image, image_mask):
    crop_segment = cv2.bitwise_and(image, image, mask = image_mask)
    segment_gray = cv2.cvtColor(crop_segment, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(segment_gray, 127, 255, cv2.THRESH_BINARY)

    rect, box = get_min_rectangle(mask)
    angle = rect[-1]

    moments = cv2.moments(box)
    if moments["m00"] == 0: return image
    
    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])
    center = (cX, cY)
    
    if angle < -45:
        angle = 90 + angle
        
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated_mask = cv2.warpAffine(mask, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    _, box = get_min_rectangle(mask)
    (x,y,w,h) = cv2.boundingRect(box)
    
    padding = 20
    return rotated[y:y+h+padding, x-padding:x+w+padding]

def extract_segment(model, preprocess_input, image, segment_idx, threshold, to_rgb=False):
    image_r = cv2.resize(image, (224,224))
    
    if to_rgb:
        image_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2RGB)
    
    image_r = np.expand_dims(preprocess_image(image_r, preprocess_input), axis=0)

    mask = preprocess_mask(np.squeeze(model.predict(image_r)), threshold)
    mask = cv2.resize(mask, (image.shape[1],image.shape[0]))
    mask = mask[..., segment_idx]
    
    points = approx_polygon(mask)
    segment = {'data': [], 'points': points, 'mask': mask}
    
    if(len(points) == 4):
        solid_mask = np.zeros((image.shape[0],image.shape[1]))
        solid_mask = cv2.fillPoly(solid_mask, np.int32([points]), color=255).astype(np.uint8)
        segment['data'] = crop_segment(image, solid_mask)
            
    return segment

