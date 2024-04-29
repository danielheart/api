from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS
import os
from notion_client import Client

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


# 初始化 Notion 客户端
NOTION_TOKEN = os.environ.get("NOTION_TOKEN")
notion = Client(auth=NOTION_TOKEN)

append_block_data = {
    "parent": {
        "database_id": "0a7d58a4923e486bb61d0121f5209f15"
    },
    "properties": {
        "Word": {
            "title": [
                {
                    "text": {
                        "content": "integration"
                    }
                }
            ]
        },
        "Phonetics": {
            "rich_text": [
                {
                    "text": {
                        "content": "/ˌɪntɪˈɡreɪʃn/"
                    }
                }
            ]
        },
        "Meaning": {
            "rich_text": [
                {
                    "text": {
                        "content": "the action or process of integrating."
                    }
                }
            ]
        },
        "Example": {
            "rich_text": [
                {
                    "text": {
                        "content": "\"economic and political integration\""
                    }
                }
            ]
        },
        "State": {
            "select": {
                "name": "New"
            }
        },
        "Source": {
            "rich_text": [
                {
                    "text": {
                        "content": "Authorization",
                        "link": {
                            "url": "https://developers.notion.com/docs/authorization#step-2-notion-redirects-the-user-to-the-integrations-redirect-uri-and-includes-a-code-parameter"
                        }
                    }
                }
            ]
        }
    },
    "icon": {
        "type": "emoji",
        "emoji": "🐣"
    },
    "children": [
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "text": [
                    {
                        "type": "text",
                        "text": {
                            "content": "noun"
                        },
                        "annotations": {
                            "italic": True,
                            "color": "gray"
                        }
                    }
                ]
            }
        },
        {
            "object": "block",
            "type": "numbered_list_item",
            "numbered_list_item": {
                "text": [
                    {
                        "type": "text",
                        "text": {
                            "content": "the action or process of integrating.\n"
                        }
                    },
                    {
                        "type": "text",
                        "text": {
                            "content": "\"economic and political integration\""
                        },
                        "annotations": {
                            "color": "gray"
                        }
                    }
                ]
            }
        },
        {
            "object": "block",
            "type": "numbered_list_item",
            "numbered_list_item": {
                "text": [
                    {
                        "type": "text",
                        "text": {
                            "content": "the finding of an integral or integrals.\n"
                        }
                    },
                    {
                        "type": "text",
                        "text": {
                            "content": "\"integration of an ordinary differential equation\""
                        },
                        "annotations": {
                            "color": "gray"
                        }
                    }
                ]
            }
        },
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "text": [
                    {
                        "type": "text",
                        "text": {
                            "content": ""
                        }
                    }
                ]
            }
        },
        {
            "object": "block",
            "type": "audio",
            "audio": {
                "external": {
                    "url": "https://ssl.gstatic.com/dictionary/static/sounds/20220808/integration--_gb_1.mp3"
                }
            }
        }
    ]
}
# append_block_data = jsonify(data)


@app.route('/savetonotion', methods=['POST'])
def savetonotion():
    try:
        # 从请求中获取数据（这里假设你有一个名为 appendBlockData 的 JSON 数据）
        append_block_data = request.json

        # 发送请求到 Notion API
        response = notion.pages.create(
            parent={"database_id": "YOUR_DATABASE_ID"}, properties=append_block_data)

        if response.get('id'):
            return jsonify({"message": "success!"}), 200
        else:
            return jsonify({"message": "error"}), 500
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500


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
