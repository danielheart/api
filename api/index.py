from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    response = jsonify({'message': 'Hello, World!'})
    # response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/sum', methods=['POST'])
def sum_numbers():
    data = request.get_json()
    numbers = data.get('numbers')
    result = sum(numbers)
    return jsonify({'result': result})


@app.route('/callback', methods=['GET'])
def callback():
    code = request.args.get('code')  # 提取授权码参数
    print('code', code)
    return 'Authorization code: {}'.format(code)


@app.route('/transform', methods=['POST'])
def transform():
    data = request.get_json()
    movingMatrixData = data.get('moving_object_list')
    movingData = data.get('moving_list')
    fixedData = data.get('fixed_list')
    result = execute(movingMatrixData, fixedData, movingData)
    response = jsonify(result)
    # response.headers.add('Access-Control-Allow-Origin', '*')
    return response


def execute(movingMatrixData, fixedData, movingData):
    # receive data, build matrix
    movingArr = np.array(movingMatrixData)
    movingMatrix = movingArr.reshape(4, 4)
    transformationRoughT0 = np.copy(movingMatrix)

    fixedArray = np.array(fixedData)
    fixedArray = fixedArray.reshape(4, 3)
    movingArray = np.array(movingData)
    movingArray = movingArray.reshape(4, 3)

    # calculate centroids
    fixedCentroid = np.mean(fixedArray, axis=0)
    movingCentroid = np.mean(movingArray, axis=0)

    # move arrays to origin
    fixedOrigin = fixedArray - fixedCentroid
    movingOrigin = movingArray - movingCentroid

    # calculate sum of squares
    fixedSumSquared = np.sum(fixedOrigin ** 2)
    movingSumSquared = np.sum(movingOrigin ** 2)

    # normalize arrays
    fixedNormalized = np.sqrt(fixedSumSquared)
    fixedNormOrigin = fixedOrigin / fixedNormalized
    movingNormalized = np.sqrt(movingSumSquared)
    movingNormOrigin = movingOrigin / movingNormalized

    # singular value decomposition
    covMatrix = np.matrix.transpose(movingNormOrigin) @ fixedNormOrigin
    U, s, Vt = np.linalg.svd(covMatrix)
    V = Vt.T
    rotation3x3 = V @ U.T

    # prevent reflection
    if np.linalg.det(rotation3x3) < 0:
        V[:, -1] *= -1
        s[-1] *= -1
        rotation3x3 = V @ U.T

    # scaling
    scalingFactor = np.sum(s) * fixedNormalized / movingNormalized
    scalingMatrix = np.eye(4)
    for i in range(3):
        scalingMatrix[i, i] *= scalingFactor
    normMatrix = np.eye(4)
    normMatrix[0:3, 3] = -np.matrix.transpose(movingCentroid)
    movingMatrix = normMatrix @ movingMatrix
    movingMatrix = scalingMatrix @ movingMatrix
    normMatrix[0:3, 3] = -normMatrix[0:3, 3]
    movingMatrix = normMatrix @ movingMatrix

    # rotation
    rotationMatrix = np.eye(4)
    rotationMatrix[0:3, 0:3] = rotation3x3
    movingMatrix = rotationMatrix @ movingMatrix

    # translation
    translationMatrix = np.eye(4)
    translationMatrix[0:3, 3] = np.matrix.transpose(
        fixedCentroid - rotation3x3 @ movingCentroid)
    movingMatrix = translationMatrix @ movingMatrix

    # compute transformation matrix
    transformationRough = movingMatrix @ np.linalg.inv(
        transformationRoughT0)
    print("transformationRough:")
    print(transformationRough)
    return transformationRough.flatten().tolist()


if __name__ == '__main__':
    app.run()
