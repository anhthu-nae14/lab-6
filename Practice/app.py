from flask import Flask, render_template_string
import torch
import torch.nn.functional as F

app = Flask(__name__)

# Công thức tính CrossEntropy Loss
def crossEntropyLoss(output, target):
    return F.cross_entropy(output, target)

# Công thức tính Mean Square Error
def meanSquareError(output, target):
    return F.mse_loss(output, target)

# Công thức tính BinaryEntropy Loss
def binaryEntropyLoss(output, target, n):
    return F.binary_cross_entropy(output, target)

# Hàm activation function
def sigmoid(x: torch.tensor):
    return 1 / (1 + torch.exp(-x))

def relu(x: torch.tensor):
    return torch.maximum(x, torch.tensor(0.0))

def softmax(zi: torch.tensor):
    exp_zi = torch.exp(zi - torch.max(zi))
    return exp_zi / exp_zi.sum()

def tanh(x: torch.tensor):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

@app.route('/')
def index():
    # Tính toán các hàm loss function
    inputs = torch.tensor([0.1, 0.3, 0.6, 0.7])
    target = torch.tensor([0.31, 0.32, 0.8, 0.2])
    n = len(inputs)
    mse = meanSquareError(inputs, target)
    binary_loss = binaryEntropyLoss(inputs, target, n)
    cross_loss = crossEntropyLoss(inputs.unsqueeze(0), torch.tensor([2]))

    # Tính toán các hàm activation function
    x = torch.tensor([1, 5, -4, 3, -2])
    f_sigmoid = sigmoid(x)
    f_relu = relu(x)
    f_softmax = softmax(x)
    f_tanh = tanh(x)

    # HTML template với CSS để trang trí và thêm thông tin cá nhân
    html = '''
    <html>
    <head>
        <title>Kết quả từ Flask</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                background-color: #f4f6f7; 
                color: #2c3e50; 
                text-align: center; 
            }
            h1 { 
                color: #3498db; 
            }
            p { 
                font-size: 1.1em; 
                margin: 10px 0; 
            }
            .container { 
                max-width: 600px; 
                margin: auto; 
                background: #ecf0f1; 
                padding: 20px; 
                border-radius: 10px; 
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); 
            }
            .info {
                font-weight: bold; 
                font-size: 1.2em;
                color: #2980b9;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Kết quả tính toán</h1>
            <div class="info">Họ tên: Phan Anh Thư</div>
            <div class="info">MSSV: 2274802010872</div>
            <h2>Kết quả các hàm Loss Function</h2>
            <p>Mean Square Error: {{ mse }}</p>
            <p>Binary Entropy Loss: {{ binary_loss }}</p>
            <p>Cross Entropy Loss: {{ cross_loss }}</p>
            
            <h2>Kết quả các hàm Activation Function</h2>
            <p>Sigmoid: {{ f_sigmoid }}</p>
            <p>ReLU: {{ f_relu }}</p>
            <p>Softmax: {{ f_softmax }}</p>
            <p>Tanh: {{ f_tanh }}</p>
        </div>
    </body>
    </html>
    '''

    return render_template_string(html, mse=mse.item(), binary_loss=binary_loss.item(), cross_loss=cross_loss.item(),
                                  f_sigmoid=f_sigmoid, f_relu=f_relu, f_softmax=f_softmax, f_tanh=f_tanh)

if __name__ == '__main__':
    app.run(debug=True)
