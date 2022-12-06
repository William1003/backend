import torch
from CNN import CNN
from flask import Flask, request
from flask_cors import CORS
import json
import torch
import xml.etree.ElementTree as ET

app = Flask(__name__)
CORS(app, supports_credentials=True)

@app.route("/")

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    data = request.get_data()
    input_data = json.loads(data)
    input_data = torch.tensor(input_data).resize(1, 28, 28)


    input_data = torch.unsqueeze(input_data, dim=1).type(torch.FloatTensor) / 255

    cnn = CNN()
    cnn.load_state_dict(torch.load('cnn2.pkl'))
    cnn.eval()

    output = cnn(input_data)
    
    predict = torch.max(output, 1)[1].data.numpy()

    return str(predict[0])

@app.route("/get_musicxml", methods=['GET', 'POST'])
def get_musicxml():
    data = request.get_data()
    score = json.loads(data)
    note_list = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    file_path = './default.xml'
    tree = ET.parse(file_path)
    root = tree.getroot()
    measure_node = root.find('.//measure')
    for i in score:
        measure_node.append(create_new_note(note_list[i - 1]))
    head = b"<?xml version='1.0' encoding='utf-8'?>\n"
    return head + ET.tostring(root)

def create_new_note(note):
    note_node = ET.Element('note')
    pitch_node = ET.Element('pitch')
    step_node = ET.Element('step')
    step_node.text = note
    octave_node = ET.Element('octave')
    octave_node.text = '4'
    duration_node = ET.Element('duration')
    duration_node.text = '4'
    type_node = ET.Element('type')
    type_node.text = 'quarter'
    pitch_node.append(step_node)
    pitch_node.append(octave_node)
    note_node.append(pitch_node)
    note_node.append(duration_node)
    note_node.append(type_node)
    return note_node

if __name__ == "__main__":
    # get_musicxml()
    app.run(port=5005)