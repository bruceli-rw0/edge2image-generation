import sys
import re
from base64 import b64encode, b64decode
from PIL import Image
from io import BytesIO

from torchvision import transforms
from generator.options.pix2pixHD_opt import Pix2PixHDOptions
from generator.models.pix2pixHD_model import Pix2PixHD

from flask import Flask, render_template, request, jsonify

sys.argv.extend(['--config', '_checkpoints/pix2pixHD-2021-08-13-04-02-59/config.yaml'])
sys.argv.extend(['--model', 'pix2pixHD'])

args, _ = Pix2PixHDOptions().parse()
args.model_id = "-2021-08-13-04-02-59"
args.load_epoch = 100

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

model = Pix2PixHD(args)
model.load()

app = Flask(__name__)

@app.route('/transform', methods=['POST'])
def transformEdges():
    image_data = re.sub('^data:image/.+;base64,', '', request.json)
    edge = Image.open(BytesIO(b64decode(image_data))).convert('RGB')
    
    input = transform(edge).unsqueeze(0)
    image = model.inference(input)
    image = ((image[0].detach().cpu().squeeze().numpy().transpose(1,2,0)+1) / 2 * 255).astype('uint8')
    pil_image = Image.fromarray(image)

    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_base64 = u"data:image/png;base64," + b64encode(buffered.getvalue()).decode("ascii")
    return jsonify(result=img_base64)

@app.route('/')
def home():
    return render_template('index.html')

if __name__=="__main__":
    app.run()
