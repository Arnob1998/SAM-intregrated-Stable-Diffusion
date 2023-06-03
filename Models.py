import os,sys
import numpy as np
import cv2
import supervision as sv
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import base64, io
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import functools
import operator
from tqdm import tqdm

class SAM_Pipeline:

    HOME = os.getcwd()
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"
    CHECKPOINT_PATH = os.path.join(HOME, "sam_vit_h_4b8939.pth")
    IMAGE_PATH = "./static/uploaded_image.jpg"

    def __init__(self) -> None:
        print("LOG: Loading SAM")
        self.load_sam()
        self.intregrity_check()

    def intregrity_check(self):
        if not os.path.isfile(self.CHECKPOINT_PATH):
            print(f"Model not found in : {self.CHECKPOINT_PATH}")
            sys.exit(1)
        
        if not os.path.isfile(self.IMAGE_PATH):
            print(f"Image not found in : {self.CHECKPOINT_PATH}")
            sys.exit(1)

    def load_sam(self):
        self.sam = sam_model_registry[self.MODEL_TYPE](checkpoint=self.CHECKPOINT_PATH).to(device=self.DEVICE)

    def to_encode(self, np_arr):
        if np_arr.shape[0] <= 3: # masks have dimension of (c,h,w)
            image = Image.fromarray(np_arr[0])
            image.save("./static/converted_mask.jpg")
        else:
            image = Image.fromarray(cv2.cvtColor(np_arr, cv2.COLOR_BGR2RGB))
        image_data = io.BytesIO()
        image.save(image_data, format='PNG')
        image_data.seek(0)
        return base64.b64encode(image_data.read()).decode('utf-8')


    def bbox_segmentation(self, bbox_data):
        mask_predictor = SamPredictor(self.sam)
        box = np.array([
            bbox_data['x'], 
            bbox_data['y'], 
            bbox_data['x'] + bbox_data['width'], 
            bbox_data['y'] + bbox_data['height']
        ])

        image_bgr = cv2.imread(self.IMAGE_PATH)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask_predictor.set_image(image_rgb)
        print("LOG: Segmenting with SAM")
        masks, scores, logits = mask_predictor.predict(
            box=box,
            multimask_output=False
        )

        box_annotator = sv.BoxAnnotator(color=sv.Color.red())
        mask_annotator = sv.MaskAnnotator(color=sv.Color.red())
        print("LOG: Annotating")
        detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks
        )
        detections = detections[detections.area == np.max(detections.area)]

        bbox_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections, skip_label=True)
        segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

        return {"bbox_encode" : self.to_encode(bbox_image), "segment_encode": self.to_encode(segmented_image), "mask_encode":self.to_encode(masks)}
    
    def auto_segmentation(self):
        image_bgr = cv2.imread(self.IMAGE_PATH)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        print("LOG: Setting Up SamAutomaticMaskGenerator")
        mask_generator = SamAutomaticMaskGenerator(self.sam)
        print("LOG: Segmentating Full Image")
        sam_result = mask_generator.generate(image_rgb)
        print("LOG: Generating Mask")
        mask_segs = [seg["segmentation"] for seg in tqdm(sam_result)]
        return mask_segs

class Diffusion_Pipeline:

    def __init__(self) -> None:
        self.DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.SIZE = 512

    def b64output(self, generated_image_pil):
        image_data = io.BytesIO()
        generated_image_pil.save(image_data, format='PNG')
        image_data.seek(0)
        return base64.b64encode(image_data.read()).decode('utf-8')  

    def generate_content(self, prompt, pil_image, pil_mask):
        print("LOG: Loading diffusion")
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting"
        # "stabilityai/stable-diffusion-2-inpainting"
        # torch_dtype=torch.float16,
        )
        pipeline = pipeline.to(self.DEVICE)
        print("LOG: Generating Image")
        gen_image_pil = pipeline(prompt=prompt, image=pil_image.resize((self.SIZE, self.SIZE)), mask_image=pil_mask.resize((self.SIZE, self.SIZE))).images[0]

        # gen_image_pil.resize(pil_image.size)
        # image_data = io.BytesIO()
        # gen_image_pil.save(image_data, format='PNG')
        # image_data.seek(0)
        return self.b64output(gen_image_pil.resize(pil_image.size))
        # gen_image_pil.save("test_output.jpg")


class Annotator_Pipeline:
    def __init__(self) -> None:
        # self.entity_masks = np.load("SAM_all_masks.npy")
        pass

    def anti_aliasing(self,entity_masks):
        entity_masks_gaussian = []
        for mask in entity_masks:
            mask_gaussian = mask.astype(np.uint8)
            # blurred_mask = cv2.GaussianBlur(mask_gaussian, (3, 3), 0)
            mask_gaussian= cv2.medianBlur(mask_gaussian,9)
            entity_masks_gaussian.append(mask_gaussian)
        print("Edges Smoothened")
        return entity_masks_gaussian
    
    def edge_detechtion_annots(self, annots, img_size):
        all_edge_coords = []
        print("Detechting annotations for edges")
        for index in tqdm(range(len(annots))):
            annoted_img = np.zeros((img_size))
            for i,j in annots[index]:
                annoted_img[i,j] = 255  
            # Apply Canny edge detection without Gaussian blur
            image_entity = np.uint8(annoted_img)
            edges = cv2.Canny(image_entity, 100, 200)
            # Get the coordinates of the edges
            edge_coords = np.argwhere(edges != 0)
            all_edge_coords.append(edge_coords)
        return all_edge_coords
    
    def extract_annots(self,entity_masks):
        annotations = []
        for i in range(len(entity_masks)):
            annotations.append(np.argwhere(entity_masks[i] != False)) 

        return annotations

    def extract_coordinates_from_mask(self,mask_image, focus):
        # Load the mask image
        mask_image = np.array(mask_image).astype(np.uint8) * 255
        coordinates_all = []

        for i in range(len(mask_image)):
            mask_cords = self.extract_coordinates_from_single_mask(mask_image[i])
            if focus:
                coordinates_all.append(mask_cords[0]) # NOTE : Remove [0] to get mask of scinario instead of single entities
            else:
                coordinates_all.append(mask_cords)
        
        return coordinates_all
    
    def extract_coordinates_from_single_mask(self,mask_image):
        # Threshold the mask image to convert it into a binary image
        _, binary_image = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract the coordinates for each contour separately
        coordinates = []
        for contour in contours:
            contour_coordinates = []
            for point in contour:
                x, y = point[0]
                contour_coordinates.append(x)
                contour_coordinates.append(y)
            coordinates.append(contour_coordinates)

        return coordinates

    def annot2coords(self,annots):
        coords = []
        for i in range(len(annots)):
            flat = functools.reduce(operator.iconcat, annots[i], [])
            flat.reverse()
            coords.append(str(flat).strip("[]"))

        return coords
    
    def simple_encode(self, np_arr):
        image = Image.fromarray(np_arr)
        image_data = io.BytesIO()
        image.save(image_data, format='PNG')
        image_data.seek(0)
        return base64.b64encode(image_data.read()).decode('utf-8')