import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np


def execute(movingMatrixData, movingData, fixedData):
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


class transform(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode())
        movingMatrixData, movingData, fixedData = data[
            'moving_object_list'], data['moving_list'], data['fixed_list']

        result = execute(movingMatrixData, movingData, fixedData)
        self.wfile.write(json.dumps(result).encode())


if __name__ == '__main__':
    server_address = ('', 8001)
    httpd = HTTPServer(server_address, transform)
    print('Starting server...')
    httpd.serve_forever()
