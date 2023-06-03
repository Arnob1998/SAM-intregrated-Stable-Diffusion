from flask import Flask, render_template, request, url_for, json, jsonify
import Models
from PIL import Image
import base64
from PIL import Image
import io
import pickle

app = Flask(__name__)

# sam_pipe = Models.SAM_Pipeline()

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        image = request.files['image']
        image.save('static/uploaded_image.jpg')
        return render_template('segmentation-mode.html')
    
    return render_template('file-upload.html')

@app.route('/manual_segmentation_maker')
def manual_segmentation_maker():
    image_url = url_for('static', filename='uploaded_image.jpg')
    return render_template('manual-segmentation-maker.html', image_url=image_url)

@app.route('/bbox_postprocess')
def bbox_postprocess():
    data = request.args.get('data')
    data_dict = json.loads(data)

    # Use the extracted values on the next page
    # sam_pipe = Models.SAM_Pipeline()
    img_dict = sam_pipe.bbox_segmentation(data_dict)

    return render_template('bbox-postprocess.html', img_encoding = img_dict)

@app.route("/auto_postprocess")
def auto_postprocess():   
    # sam_pipe = Models.SAM_Pipeline()

    entity_masks = sam_pipe.auto_segmentation()

    annot_pipe = Models.Annotator_Pipeline()
    # entity_masks = annot_pipe.anti_aliasing(entity_masks)
    coords = annot_pipe.extract_coordinates_from_mask(entity_masks, focus=False)
    coords_focus = annot_pipe.extract_coordinates_from_mask(entity_masks, focus=True)
    # entity_annots = annot_pipe.edge_detechtion_annots(entity_annots, entity_masks[0].shape)
    # coords = annot_pipe.annot2coords(entity_annots)

    data = {}
    data_focus = {}

    for i in range(len(entity_masks)):
        data[i] = {"mask": annot_pipe.simple_encode(entity_masks[i]), "coordination": coords[i]}
        data_focus[i] = {"mask": annot_pipe.simple_encode(entity_masks[i]), "coordination": coords_focus[i]}
    
    return render_template('auto-postprocess.html', data = data, data_focus=data_focus)


@app.route('/save_image', methods=['POST'])
def save_image():
    data = request.json['imageBase64']  # Get the Base64 encoded image data from the AJAX request
    image_data = base64.b64decode(data)
    image = Image.open(io.BytesIO(image_data))
    image.save("./static/converted_mask.jpg")
    return jsonify(status='success')

@app.route("/diffusion_generation", methods=['POST'])
def diffusion_generation():
    prompt = request.form.get('data')
    diffusion_pipe = Models.Diffusion_Pipeline()

    img_pil = Image.open("./static/uploaded_image.jpg")
    mask_pil = Image.open("./static/converted_mask.jpg")

    img_encoding = diffusion_pipe.generate_content(prompt,img_pil,mask_pil)

    return render_template('diffusion-output.html', generated_img = img_encoding, caption = prompt)


if __name__ == '__main__':
    global sam_pipe
    sam_pipe = Models.SAM_Pipeline()
    app.run(debug=True)